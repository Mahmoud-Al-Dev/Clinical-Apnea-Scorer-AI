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
NIGHT_ID = 12 # The night you collected data from
PREF_DATA_PATH = f'dpo_preferences_n{NIGHT_ID}.npy'
SFT_WEIGHTS_PATH = os.path.join(parent_dir, f'penta_lstm_{TARGET_TYPE}_weights.pth')
DPO_WEIGHTS_PATH = os.path.join(parent_dir, f'penta_lstm_{TARGET_TYPE}_DPO_weights.pth')

EPOCHS = 15            # Keep this very low to prevent overfitting on 41 pairs
BATCH_SIZE = 4         # Small batch size for stability with tiny datasets
LEARNING_RATE = 2e-6   # Extremely small LR (as Claude advised)
BETA = 0.05             # The DPO temperature hyperparameter (0.1 is standard)
# ==========================================

def get_boundary_mask(mask, edge_width_samples=32):
    """
    Identifies the boundaries (starts and ends) of events.
    Multiplies the loss importance by 3.0 in these regions.
    """
    padded = np.pad(mask, (1, 1), 'constant')
    diffs = np.diff(padded)
    
    boundary_mask = np.ones_like(mask, dtype=float)
    
    # Where mask changes from 0->1 or 1->0
    change_indices = np.where(diffs != 0)[0]
    
    for idx in change_indices:
        start_idx = max(0, idx - edge_width_samples)
        end_idx = min(len(mask), idx + edge_width_samples)
        boundary_mask[start_idx:end_idx] = 3.0 # Apply the 3x penalty multiplier
        
    return torch.tensor(boundary_mask, dtype=torch.float32)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, beta=0.1):
    """
    The core Direct Preference Optimization math.
    Loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    loss = -torch.nn.functional.logsigmoid(beta * pi_logratios).mean()
    return loss

def calculate_log_prob(logits, target_mask, boundary_weights):
    """
    Calculates the boundary-weighted log probability of a specific mask given the model's logits.
    """
    # Logits shape: [Batch, 2, 960]
    # We want the log probability of class 1 (Apnea) where target_mask == 1, 
    # and class 0 (Normal) where target_mask == 0.
    
    log_probs = torch.log_softmax(logits, dim=1) # [Batch, 2, 960]
    
    # Gather the log probabilities corresponding to the target mask
    # target_mask needs to be shape [Batch, 1, 960] to use gather
    target_mask_expanded = target_mask.unsqueeze(1).long() 
    
    # This grabs the log_prob of the chosen class for every single frame
    selected_log_probs = torch.gather(log_probs, 1, target_mask_expanded).squeeze(1) # [Batch, 960]
    
    # Apply Claude's boundary weighting trick
    weighted_log_probs = selected_log_probs * boundary_weights
    
    # Sum across the sequence (960 frames) to get the total log probability for the whole box
    sequence_log_prob = weighted_log_probs.sum(dim=1) 
    
    return sequence_log_prob

def train_dpo():
    print(f"--- Starting Direct Preference Optimization (DPO) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        raw_data = np.load(PREF_DATA_PATH, allow_pickle=True)
        print(f"Loaded {len(raw_data)} preference pairs from {PREF_DATA_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {PREF_DATA_PATH}")
        sys.exit()

    # We need a reference model (frozen SFT) and a policy model (the one we update)
    # But because our dataset is so small, we can simplify: we just calculate log_probs directly on the policy.
    
    # 2. Load Model & Apply Freezing Guardrails
    model = ConvLSTM().to(device)
    model.load_state_dict(torch.load(SFT_WEIGHTS_PATH, map_location=device, weights_only=True))
    
    print("\nApplying Freezing Guardrails...")
    for name, param in model.named_parameters():
        if "cnn" in name:
            param.requires_grad = False
            print(f"  ❄️ Froze CNN layer: {name}")
        else:
            param.requires_grad = True # Update LSTM and output head
            
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 3. Training Loop
    model.train()
    
    num_pairs = len(raw_data)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        # Shuffle data each epoch
        np.random.shuffle(raw_data)
        
        for i in range(0, num_pairs, BATCH_SIZE):
            batch_data = raw_data[i:i+BATCH_SIZE]
            
            # Extract lists
            contexts = [item['context_signal'] for item in batch_data]
            chosen_masks = [item['chosen_mask'] for item in batch_data]
            rejected_masks = [item['rejected_mask'] for item in batch_data]
            
            # Convert to tensors
            x_batch = torch.tensor(np.array(contexts), dtype=torch.float32).to(device)
            y_chosen = torch.tensor(np.array(chosen_masks), dtype=torch.float32).to(device)
            y_rejected = torch.tensor(np.array(rejected_masks), dtype=torch.float32).to(device)
            
            # Create boundary weights based on the chosen mask
            b_weights = torch.stack([get_boundary_mask(m) for m in chosen_masks]).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x_batch)
            
            # Calculate Log Probs
            chosen_logps = calculate_log_prob(logits, y_chosen, b_weights)
            rejected_logps = calculate_log_prob(logits, y_rejected, b_weights)
            
            # Calculate DPO Loss
            loss = dpo_loss(chosen_logps, rejected_logps, BETA)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (num_pairs / BATCH_SIZE)
        print(f"Epoch {epoch+1}/{EPOCHS} | DPO Loss: {avg_loss:.4f}")

    # 4. Save the new DPO weights
    torch.save(model.state_dict(), DPO_WEIGHTS_PATH)
    print(f"\n✅ DPO Training Complete! Saved new weights to {DPO_WEIGHTS_PATH}")
    print("Next step: Run evaluate_clinical_metrics_sft.py using these new weights on your Clean Validation Nights.")

if __name__ == "__main__":
    train_dpo()