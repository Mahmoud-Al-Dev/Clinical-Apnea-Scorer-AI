import torch
import numpy as np
import os
import cleanlab
from cleanlab.filter import find_label_issues

# CHANGED: Import your ConvLSTM 
from train_lstm import ConvLSTM

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'CA'    
CLEAN_TEACHER_WEIGHTS = f'penta_lstm_{TARGET_TYPE}_weights.pth' # Your newly trained pristine model
NOISY_NIGHT_ID = 3 # The night we want to clean next
# ==========================================

def run_cleanlab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Cleanlab Noise Detection for Night {NOISY_NIGHT_ID} ---")
    
    # 1. Load the Noisy Data
    X_noisy = np.load(f'X_{NOISY_NIGHT_ID}.npy')
    Y_noisy_raw = np.load(f'Y_{TARGET_TYPE}_{NOISY_NIGHT_ID}.npy')
    
    ai_indices = [0, 3, 4, 5, 6, 7]
    
    # 2. Convert 960-timestep labels to a single Segment Label (0 = Normal, 1 = Apnea)
    # If the doctor flagged ANY apnea in this window, the segment is labeled 1.
    Y_noisy_segment = np.any(Y_noisy_raw == 1, axis=1).astype(int).flatten()
    
    # 3. Load the Pristine Teacher Model
    agent = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    agent.load_state_dict(torch.load(CLEAN_TEACHER_WEIGHTS, map_location=device, weights_only=True))
    agent.eval()
    
    # 4. Gather Teacher Predictions
    print("Generating Pristine Teacher probabilities...")
    pred_probs = []
    
    with torch.no_grad():
        for i in range(len(X_noisy)):
            obs = X_noisy[i][:, ai_indices].astype(np.float32)
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            
            # --- THE FIX: ConvLSTM logic ---
            # SFT model only returns logits (not a tuple)
            logits = agent(obs_tensor) 
            
            # ConvLSTM shape is (Batch, Classes, Timesteps) -> (1, 2, 960)
            # We softmax on dim=1 to get probabilities
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy() # Shape: (2, 960)
            
            # We want the Segment Probability. 
            # We take the MAXIMUM probability of Apnea (class index 1) across the 960 timesteps.
            prob_apnea = np.max(probs[1, :]) 
            prob_normal = 1.0 - prob_apnea
            
            pred_probs.append([prob_normal, prob_apnea])
            
    pred_probs = np.array(pred_probs)

    # 5. ENTER CLEANLAB
    print("\nRunning Confident Learning (Cleanlab)...")
    
    # Cleanlab mathematically calculates the exact indices where the doctor's label
    # contradicts the confident predictions of the Teacher model.
    ranked_label_issues = find_label_issues(
        labels=Y_noisy_segment,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    print(f"\n🚨 Cleanlab found {len(ranked_label_issues)} highly suspicious clinical labels in Night {NOISY_NIGHT_ID}!")
    
    # Let's break down the errors so you know what you're looking at:
    false_negatives = 0
    false_positives = 0
    
    for idx in ranked_label_issues:
        if Y_noisy_segment[idx] == 0:
            false_negatives += 1 # Doc says Normal, AI is sure it's Apnea
        else:
            false_positives += 1 # Doc says Apnea, AI is sure it's Normal
            
    print(f"   - Suspicious 'Normal' labels (Missed Apneas): {false_negatives}")
    print(f"   - Suspicious 'Apnea' labels (False Alarms): {false_positives}")
    
    # 6. Save the suspicious indices for review
    np.save(f'cleanlab_flags_{TARGET_TYPE}_n{NOISY_NIGHT_ID}.npy', ranked_label_issues)
    print("\nSaved suspicious segment indices. You only need to review these specific segments!")

if __name__ == "__main__":
    run_cleanlab()