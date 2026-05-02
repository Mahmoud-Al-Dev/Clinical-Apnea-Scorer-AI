import torch
import numpy as np
import os
import sys
import gc

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
CLEAN_TRAIN_NIGHTS = [1,2,5,6,7,9,10,11,12,13,15,17,18,19,20] 

SFT_WEIGHTS_PATH = os.path.join(parent_dir, f'penta_lstm_{TARGET_TYPE}_weights.pth')
OUTPUT_FILE = f'dpo_automined_{TARGET_TYPE}_pairs.npy'

SAMPLING_RATE = 32
AI_CHANNELS = [0, 3, 4, 5, 6, 7] 
# ==========================================

def extract_events(mask):
    padded = np.pad(mask, (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))

def enforce_10s_rule(mask, sampling_rate=32):
    min_samples = 10 * sampling_rate
    cleaned_mask = np.copy(mask)
    events = extract_events(cleaned_mask)
    
    for start, end in events:
        if (end - start) < min_samples:
            cleaned_mask[start:end] = 0
    return cleaned_mask

def run_auto_miner():
    print(f"--- Starting DPO Auto-Miner for {TARGET_TYPE} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConvLSTM().to(device)
    try:
        model.load_state_dict(torch.load(SFT_WEIGHTS_PATH, map_location=device, weights_only=True))
        print(f"✅ Loaded SFT weights from {SFT_WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find weights at {SFT_WEIGHTS_PATH}")
        sys.exit()
        
    model.eval()
    
    preference_dataset = []
    total_missed_mined = 0
    total_hallucinations_mined = 0
    
    for night_id in CLEAN_TRAIN_NIGHTS:
        print(f"\n🔍 Scanning Night {night_id}...")
        
        # Load Data
        try:
            X_night = np.load(os.path.join(parent_dir, 'Nights', f'X_{night_id}.npy'))
            
            # Smart load: Prioritize Silver labels if they exist
            silver_path = os.path.join(parent_dir, 'Nights', f'Y_{TARGET_TYPE}_{night_id}.npy')
            Y_true = np.load(silver_path)
                
        except FileNotFoundError as e:
            print(f"⚠️ Skipping Night {night_id} - Data missing: {e}")
            continue

        num_segments = len(X_night)
        batch_size = 64
        
        night_missed = 0
        night_hallucinated = 0

        with torch.no_grad():
            for i in range(0, num_segments, batch_size):
                end_idx = min(i + batch_size, num_segments)
                
                # Extract batch and move to GPU
                batch_x = X_night[i:end_idx, :, AI_CHANNELS].astype(np.float32)
                obs_tensor = torch.tensor(batch_x).to(device)
                
                # Batch Inference
                logits = model(obs_tensor) 
                probs = torch.softmax(logits, dim=1).cpu().numpy() # Shape: [Batch, 2, 960]
                
                for b_idx in range(end_idx - i):
                    global_idx = i + b_idx
                    
                    true_mask = Y_true[global_idx].flatten()
                    prob_array = probs[b_idx, 1, :]
                    
                    raw_pred_mask = (prob_array > 0.5).astype(int)
                    pred_mask = enforce_10s_rule(raw_pred_mask, SAMPLING_RATE)
                    
                    # --- THE MINING LOGIC ---
                    
                    # 1. Check for Missed Apnea (Idea 1)
                    # The doctor/silver label has a real event, but AI predicted absolutely nothing
                    if np.sum(true_mask) >= 320 and np.sum(pred_mask) == 0:
                        preference_dataset.append({
                            'context_signal': batch_x[b_idx], 
                            'chosen_mask': true_mask,     # Reward detection
                            'rejected_mask': pred_mask    # Punish silence
                        })
                        night_missed += 1
                        total_missed_mined += 1
                        
                    # 2. Check for Hallucination (Idea 2)
                    # The true label is perfectly empty, but AI drew a 10s+ box
                    elif np.sum(true_mask) == 0 and np.sum(pred_mask) >= 320:
                        preference_dataset.append({
                            'context_signal': batch_x[b_idx], 
                            'chosen_mask': true_mask,     # Reward silence
                            'rejected_mask': pred_mask    # Punish hallucination
                        })
                        night_hallucinated += 1
                        total_hallucinations_mined += 1
                        
        print(f"  -> Found {night_missed} missed apneas and {night_hallucinated} hallucinations.")
        
        # Memory cleanup per night
        del X_night, Y_true
        gc.collect()

    # Save the Dataset
    if len(preference_dataset) > 0:
        np.save(OUTPUT_FILE, preference_dataset)
        print(f"\n==================================================")
        print(f"✅ AUTO-MINING COMPLETE!")
        print(f"Total Missed Apneas Extracted:    {total_missed_mined}")
        print(f"Total Hallucinations Extracted:   {total_hallucinations_mined}")
        print(f"Total Pairs Saved to {OUTPUT_FILE}: {len(preference_dataset)}")
        print(f"==================================================")
    else:
        print("\n⚠️ No preference pairs found. Your SFT model might be perfectly aligned with these nights!")

if __name__ == "__main__":
    run_auto_miner()