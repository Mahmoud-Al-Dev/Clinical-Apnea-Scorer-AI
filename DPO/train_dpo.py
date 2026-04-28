import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# ==========================================
# --- PATH RESOLUTION & IMPORTS ---
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from train_lstm import ConvLSTM
    print("✅ Successfully imported ConvLSTM from the parent directory.")
except ImportError as e:
    print(f"❌ Could not find train_lstm.py. Error: {e}")
    sys.exit()

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'OSA'

PREF_DATA_PATHS = [
    f'dpo_automined_{TARGET_TYPE}_pairs.npy',
]

SFT_WEIGHTS_PATH = os.path.join(parent_dir, f'penta_lstm_{TARGET_TYPE}_weights.pth')
DPO_WEIGHTS_PATH = os.path.join(parent_dir, f'penta_lstm_{TARGET_TYPE}_DPO_weights.pth')

PAIR_WEIGHTS = {
    'detection': 0.40,   # Idea 1: Fix Missed Apneas
    'suppression': 0.15, # Idea 2: Fix Hallucinations
    'boundary': 0.05     # The original boundary jitter
}

EPOCHS = 5            
BATCH_SIZE = 16        # Increased from 4 to 16 for faster processing
LEARNING_RATE = 2e-6   
BETA = 0.05      
# ==========================================

def get_boundary_mask(mask, edge_width_samples=32):
    padded = np.pad(mask, (1, 1), 'constant')
    diffs = np.diff(padded)
    boundary_mask = np.ones_like(mask, dtype=float)
    change_indices = np.where(diffs != 0)[0]
    for idx in change_indices:
        start_idx = max(0, idx - edge_width_samples)
        end_idx = min(len(mask), idx + edge_width_samples)
        boundary_mask[start_idx:end_idx] = 3.0 
    return torch.tensor(boundary_mask, dtype=torch.float32)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, batch_weights, beta=0.1):
    """
    Updated DPO Loss that applies Claude's specific weights to each pair.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    
    # Calculate base loss
    base_loss = -torch.nn.functional.logsigmoid(beta * pi_logratios)
    
    # Apply dynamic weights
    weighted_loss = base_loss * batch_weights
    
    return weighted_loss.mean()

def calculate_log_prob(logits, target_mask, boundary_weights):
    log_probs = torch.log_softmax(logits, dim=1) 
    target_mask_expanded = target_mask.unsqueeze(1).long() 
    selected_log_probs = torch.gather(log_probs, 1, target_mask_expanded).squeeze(1) 
    weighted_log_probs = selected_log_probs * boundary_weights
    sequence_log_prob = weighted_log_probs.sum(dim=1) 
    return sequence_log_prob

def train_dpo():
    print(f"--- Starting Weighted Direct Preference Optimization (DPO) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load and Combine Data
    raw_data = []
    for path in PREF_DATA_PATHS:
        try:
            data_chunk = np.load(path, allow_pickle=True)
            raw_data.extend(data_chunk)
            print(f"✅ Loaded {len(data_chunk)} pairs from {path}")
        except FileNotFoundError:
            print(f"⚠️ Warning: Could not find {path}, skipping...")
            
    if len(raw_data) == 0:
        print("❌ Error: No preference pairs loaded. Exiting.")
        sys.exit()
        
    print(f"\nTotal Unified Dataset: {len(raw_data)} preference pairs.")

    # 2. Load Model
    model = ConvLSTM().to(device)
    model.load_state_dict(torch.load(SFT_WEIGHTS_PATH, map_location=device, weights_only=True))
    
    print("\nApplying Freezing Guardrails...")
    for name, param in model.named_parameters():
        if "cnn" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True 
            
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 3. Training Loop
    model.train()
    num_pairs = len(raw_data)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        np.random.shuffle(raw_data)
        
        # Track stats for logging
        epoch_stats = {'detection': 0, 'suppression': 0, 'boundary': 0}
        
        for i in range(0, num_pairs, BATCH_SIZE):
            batch_data = raw_data[i:i+BATCH_SIZE]
            
            contexts = [item['context_signal'] for item in batch_data]
            chosen_masks = [item['chosen_mask'] for item in batch_data]
            rejected_masks = [item['rejected_mask'] for item in batch_data]
            
            # --- DYNAMIC PAIR CLASSIFICATION ---
            batch_weights = []
            for c_mask, r_mask in zip(chosen_masks, rejected_masks):
                c_sum = np.sum(c_mask)
                r_sum = np.sum(r_mask)
                
                if c_sum > 0 and r_sum == 0:
                    batch_weights.append(PAIR_WEIGHTS['detection'])
                    epoch_stats['detection'] += 1
                elif c_sum == 0 and r_sum > 0:
                    batch_weights.append(PAIR_WEIGHTS['suppression'])
                    epoch_stats['suppression'] += 1
                else:
                    batch_weights.append(PAIR_WEIGHTS['boundary'])
                    epoch_stats['boundary'] += 1
            
            w_tensor = torch.tensor(batch_weights, dtype=torch.float32).to(device)
            
            x_batch = torch.tensor(np.array(contexts), dtype=torch.float32).to(device)
            y_chosen = torch.tensor(np.array(chosen_masks), dtype=torch.float32).to(device)
            y_rejected = torch.tensor(np.array(rejected_masks), dtype=torch.float32).to(device)
            b_weights = torch.stack([get_boundary_mask(m) for m in chosen_masks]).to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            
            chosen_logps = calculate_log_prob(logits, y_chosen, b_weights)
            rejected_logps = calculate_log_prob(logits, y_rejected, b_weights)
            
            # Pass our dynamic weights into the loss function
            loss = dpo_loss(chosen_logps, rejected_logps, w_tensor, BETA)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / (num_pairs / BATCH_SIZE)
        print(f"Epoch {epoch+1}/{EPOCHS} | DPO Loss: {avg_loss:.4f}")
        if epoch == 0:
            print(f"   -> Batch Composition: {epoch_stats['detection']} Detections | {epoch_stats['suppression']} Suppressions | {epoch_stats['boundary']} Boundaries")

    # 4. Save Weights
    torch.save(model.state_dict(), DPO_WEIGHTS_PATH)
    print(f"\n✅ DPO Training Complete! Saved new weights to {DPO_WEIGHTS_PATH}")

if __name__ == "__main__":
    train_dpo()