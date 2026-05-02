import torch
import numpy as np
import os
import cleanlab
from cleanlab.filter import find_label_issues
from scipy.ndimage import label 
from train_lstm import ConvLSTM
import gc

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'OSA'    
MODEL_TYPE = 'DPO'

if MODEL_TYPE == 'SFT':
    CLEAN_TEACHER_WEIGHTS = f'penta_lstm_{TARGET_TYPE}_weights.pth' 
else:
    CLEAN_TEACHER_WEIGHTS = f'penta_lstm_{TARGET_TYPE}_{MODEL_TYPE}_weights.pth'

NOISY_NIGHTS = [4,10,11,12] 
# ==========================================

def process_single_night(night_id, agent, device):
    print(f"\n==================================================")
    print(f"--- Running Noise Detection for Night {night_id} ---")
    print(f"==================================================")
    
    try:
        # Reverted mmap_mode to prevent I/O thrashing. Loads straight to RAM.
        X_noisy = np.load(f'Nights/X_{night_id}.npy')
        Y_noisy_raw = np.load(f'Nights/Y_{TARGET_TYPE}_{night_id}.npy')
    except FileNotFoundError as e:
        print(f"❌ Error loading data for Night {night_id}: {e}")
        return

    ai_indices = [0, 3, 4, 5, 6, 7]
    Y_noisy_segment = np.any(Y_noisy_raw == 1, axis=1).astype(int).flatten()
    
    num_segments = len(X_noisy)
    batch_size = 64
    
    # Preallocate arrays to prevent memory fragmentation
    pred_probs = np.zeros((num_segments, 2), dtype=np.float32)
    meets_10s_rule = np.zeros(num_segments, dtype=bool)
    
    print("Generating AI probabilities with 10-second duration filter (Batched)...")
    
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            end_idx = min(i + batch_size, num_segments)
            
            # Extract batch and move to GPU
            batch_x = X_noisy[i:end_idx, :, ai_indices].astype(np.float32)
            obs_tensor = torch.tensor(batch_x).to(device)
            
            # Batch Inference
            logits = agent(obs_tensor) 
            probs = torch.softmax(logits, dim=1).cpu().numpy() # Shape: [Batch, 2, 960]
            
            # Process each segment in the batch
            for b_idx in range(end_idx - i):
                global_idx = i + b_idx
                prob_array = probs[b_idx, 1, :] 
                prob_apnea = np.max(prob_array) 
                prob_normal = 1.0 - prob_apnea
                
                pred_probs[global_idx] = [prob_normal, prob_apnea]
                
                # --- APPLY 10-SECOND RULE TO PREDICTIONS ---
                binary_preds = (prob_array > 0.5).astype(int)
                labeled_array, num_features = label(binary_preds)
                
                has_10s_event = False
                for j in range(1, num_features + 1):
                    if np.sum(labeled_array == j) >= 320: 
                        has_10s_event = True
                        break
                        
                meets_10s_rule[global_idx] = has_10s_event

    num_doctor_events = np.sum(Y_noisy_segment)
    
    if num_doctor_events == 0:
        print(f"\n⚠️ WARNING: The doctor labeled EXACTLY ZERO events for Night {night_id}.")
        print("Bypassing Cleanlab -> Directly flagging AI predictions using 10s Rule...")
        
        suspicious_mask = (pred_probs[:, 1] > 0.9) & meets_10s_rule
        ranked_label_issues = np.where(suspicious_mask)[0]
        
        if len(ranked_label_issues) > 0:
            ranked_label_issues = ranked_label_issues[np.argsort(-pred_probs[ranked_label_issues, 1])]
        
        print(f"🚨 Direct AI Thresholding found {len(ranked_label_issues)} highly suspicious segments!")
        
    else:
        print("\nRunning Confident Learning (Cleanlab)...")
        ranked_label_issues = find_label_issues(
            labels=Y_noisy_segment,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence',
            n_jobs=1
        )
        
        filtered_issues = []
        for idx in ranked_label_issues:
            if Y_noisy_segment[idx] == 0: 
                if meets_10s_rule[idx]:
                    filtered_issues.append(idx)
            else: 
                filtered_issues.append(idx)
                
        ranked_label_issues = np.array(filtered_issues)
        print(f"🚨 Cleanlab found {len(ranked_label_issues)} highly suspicious clinical labels (filtered for 10s rules)!")
    
    output_filename = f'cleanlab_flags_{TARGET_TYPE}_n{night_id}.npy'
    np.save(output_filename, ranked_label_issues)
    print(f"✅ Saved suspicious segment indices to {output_filename}")
    
    # ==========================================
    # --- AGGRESSIVE MEMORY CLEARING ---
    # ==========================================
    print("🧹 Sweeping memory before next night...")
    del X_noisy
    del Y_noisy_raw
    del pred_probs
    del meets_10s_rule
    del ranked_label_issues
    
    # Force Python to delete variables from RAM
    gc.collect()
    
    # Force GPU to dump unused tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_multi_night_cleanlab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Multi-Night Cleanlab Pipeline on {device}...")
    
    agent = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    try:
        agent.load_state_dict(torch.load(CLEAN_TEACHER_WEIGHTS, map_location=device, weights_only=True))
        print(f"Loaded weights from {CLEAN_TEACHER_WEIGHTS}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {CLEAN_TEACHER_WEIGHTS}.")
        return
        
    agent.eval()
    
    for night in NOISY_NIGHTS:
        process_single_night(night, agent, device)
        
    print("\n🎉 All nights processed successfully!")

if __name__ == "__main__":
    run_multi_night_cleanlab()