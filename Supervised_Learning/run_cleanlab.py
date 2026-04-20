import torch
import numpy as np
import os
import cleanlab
from cleanlab.filter import find_label_issues
from scipy.ndimage import label # <-- IMPORTED FOR 10s RULE
from train_lstm import ConvLSTM

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TARGET_TYPE = 'OSA'    
CLEAN_TEACHER_WEIGHTS = f'penta_lstm_{TARGET_TYPE}_weights.pth' 
NOISY_NIGHT_ID = 9
# ==========================================

def run_cleanlab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Noise Detection for Night {NOISY_NIGHT_ID} ---")
    
    X_noisy = np.load(f'Nights/X_{NOISY_NIGHT_ID}.npy')
    Y_noisy_raw = np.load(f'Nights/Y_{TARGET_TYPE}_{NOISY_NIGHT_ID}.npy')
    
    ai_indices = [0, 3, 4, 5, 6, 7]
    
    Y_noisy_segment = np.any(Y_noisy_raw == 1, axis=1).astype(int).flatten()
    
    agent = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
    agent.load_state_dict(torch.load(CLEAN_TEACHER_WEIGHTS, map_location=device, weights_only=True))
    agent.eval()
    
    print("Generating AI probabilities with 10-second duration filter...")
    pred_probs = []
    meets_10s_rule = [] # NEW: Track which segments actually have a 10s event
    
    with torch.no_grad():
        for i in range(len(X_noisy)):
            obs = X_noisy[i][:, ai_indices].astype(np.float32)
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            
            logits = agent(obs_tensor) 
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy() 
            
            prob_array = probs[1, :] # 960 frames
            prob_apnea = np.max(prob_array) 
            prob_normal = 1.0 - prob_apnea
            
            pred_probs.append([prob_normal, prob_apnea])
            
            # --- NEW: APPLY 10-SECOND RULE TO PREDICTIONS ---
            binary_preds = (prob_array > 0.5).astype(int)
            labeled_array, num_features = label(binary_preds)
            
            has_10s_event = False
            for j in range(1, num_features + 1):
                if np.sum(labeled_array == j) >= 320: # 320 frames = 10 seconds
                    has_10s_event = True
                    break
                    
            meets_10s_rule.append(has_10s_event)
            
    pred_probs = np.array(pred_probs)
    meets_10s_rule = np.array(meets_10s_rule)

    num_doctor_events = np.sum(Y_noisy_segment)
    
    if num_doctor_events == 0:
        print("\n⚠️ WARNING: The doctor labeled EXACTLY ZERO events for this night.")
        print("Bypassing Cleanlab -> Directly flagging AI predictions using 10s Rule...")
        
        # Find where AI has >50% confidence AND the event is >= 10 seconds
        suspicious_mask = (pred_probs[:, 1] > 0.9) & meets_10s_rule
        ranked_label_issues = np.where(suspicious_mask)[0]
        
        # Sort them by AI confidence
        if len(ranked_label_issues) > 0:
            ranked_label_issues = ranked_label_issues[np.argsort(-pred_probs[ranked_label_issues, 1])]
        
        print(f"\n🚨 Direct AI Thresholding found {len(ranked_label_issues)} highly suspicious segments!")
        print(f"   - Suspicious 'Normal' labels (Missed Apneas): {len(ranked_label_issues)}")
        print(f"   - Suspicious 'Apnea' labels (False Alarms): 0")
        
    else:
        print("\nRunning Confident Learning (Cleanlab)...")
        ranked_label_issues = find_label_issues(
            labels=Y_noisy_segment,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )
        
        # We can also apply the 10s rule to cleanlab's false negative flags to reduce noise!
        filtered_issues = []
        false_negatives = 0
        false_positives = 0
        
        for idx in ranked_label_issues:
            if Y_noisy_segment[idx] == 0: # Doctor missed it
                # Only keep it if the AI actually found a full 10-second event
                if meets_10s_rule[idx]:
                    filtered_issues.append(idx)
                    false_negatives += 1
            else: # Doctor False Alarm
                filtered_issues.append(idx)
                false_positives += 1 
                
        ranked_label_issues = np.array(filtered_issues)
                
        print(f"\n🚨 Cleanlab found {len(ranked_label_issues)} highly suspicious clinical labels (filtered for 10s rules)!")
        print(f"   - Suspicious 'Normal' labels (Missed Apneas): {false_negatives}")
        print(f"   - Suspicious 'Apnea' labels (False Alarms): {false_positives}")
    
    np.save(f'cleanlab_flags_{TARGET_TYPE}_n{NOISY_NIGHT_ID}.npy', ranked_label_issues)
    print("\nSaved suspicious segment indices. Ready for review!")

if __name__ == "__main__":
    run_cleanlab()