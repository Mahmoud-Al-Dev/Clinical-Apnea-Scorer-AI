import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os
from train_lstm import ConvLSTM

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TYPE = 'OSA'  
NIGHT_ID = 20
# ==========================================

print(f"--- Launching Manual AI Reviewer for {TARGET_TYPE} Night {NIGHT_ID} ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data from 'Nights' folder
X = np.load(os.path.join('Nights', f'X_{NIGHT_ID}.npy'))
segment_times = np.load(os.path.join('Nights', f'segment_times_n{NIGHT_ID}.npy'))

original_y_path = os.path.join('Nights', f'Y_{TARGET_TYPE}_{NIGHT_ID}.npy')
silver_y_path = os.path.join('Nights', f'Y_{TARGET_TYPE}_{NIGHT_ID}_SILVER.npy')

# Resume from Silver if it exists, otherwise copy original
if os.path.exists(silver_y_path):
    print("Found existing SILVER standard. Resuming...")
    Y_working = np.load(silver_y_path)
else:
    print("Starting fresh from original clinical labels...")
    Y_working = np.load(original_y_path)
    
Y_original = np.load(original_y_path) # Keep for reference

# 2. Load SFT Model
model = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
weights_path = f'penta_lstm_{TARGET_TYPE}_weights.pth'
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

ai_indices = [0, 3, 4, 5, 6, 7]
human_indices = [0, 1, 2, 5] 

new_discoveries_count = 0
segments_reviewed = 0

print("\nScanning night for AI discoveries...")

for seg_idx in range(len(X)):
    
    # We only want to review segments where the Doctor originally found NOTHING
    doc_has_event = np.any(Y_original[seg_idx] == 1)
    if doc_has_event:
        continue # Skip, doctor already handled this one
        
    raw_segment = X[seg_idx]
    ai_obs = torch.tensor(raw_segment[:, ai_indices], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(ai_obs)
        probs = torch.softmax(logits, dim=1)[0, 1, :].cpu().numpy()
        
        # Apply 10s rule
        binary_preds = (probs > 0.5).astype(int)
        labeled_array, num_features = label(binary_preds)
        
        has_valid_ai_event = False
        for j in range(1, num_features + 1):
            if np.sum(labeled_array == j) >= 320: # 10 seconds
                has_valid_ai_event = True
            else:
                binary_preds[labeled_array == j] = 0 # Erase micro-flickers
                
    # If the AI found a solid event and the doctor missed it, we review it!
    if has_valid_ai_event:
        # Check if we already approved it in a previous session
        if np.any(Y_working[seg_idx] == 1):
            continue 
            
        segments_reviewed += 1
        time_axis = segment_times[seg_idx]
        vis_signals = raw_segment[:, human_indices]
        ratio_signal = raw_segment[:, 6]
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"AI Discovery | Segment {seg_idx} | Doctor originally said NORMAL", fontsize=14)
        
        # 1. PFlow
        axes[0].plot(time_axis, vis_signals[:, 0], color='blue', alpha=0.7)
        axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                             where=(binary_preds == 1), color='red', alpha=0.4, label='AI Prediction')
        axes[0].set_ylabel('PFlow_Clean')
        axes[0].legend(loc='upper right')

        # 2. Thorax
        axes[1].plot(time_axis, vis_signals[:, 1], color='green', alpha=0.7)
        axes[1].set_ylabel('Thorax_Clean')

        # 3. Abdomen
        axes[2].plot(time_axis, vis_signals[:, 2], color='purple', alpha=0.7)
        axes[2].set_ylabel('Abdomen_Clean')

        # 4. Effort-Flow Ratio
        axes[3].plot(time_axis, ratio_signal, color='darkred', alpha=0.8)
        axes[3].set_ylabel('Ratio')

        # 5. SaO2
        axes[4].plot(time_axis, vis_signals[:, 3], color='magenta', alpha=0.7)
        axes[4].set_ylabel('SaO2_Smooth')

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.xlabel("Real Elapsed Time (Seconds)")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        print(f"\n--- Segment {seg_idx} ---")
        print("Options: [y] Accept AI | [n] Reject AI | [q] Save & Quit")
        user_input = input("Choice: ").strip().lower()
        plt.close(fig)
        
        if user_input == 'y':
            print("✅ Accepted! Merging AI boundary into Silver Standard.")
            Y_working[seg_idx] = binary_preds.reshape(960, 1)
            new_discoveries_count += 1
        elif user_input == 'q':
            print("\n💾 Quitting early and saving progress...")
            break
        else:
            print("❌ Rejected. Moving to next...")

# Final Save
np.save(silver_y_path, Y_working)
print(f"\n=========================================")
print(f"Session Complete!")
print(f"Segments Reviewed: {segments_reviewed}")
print(f"New {TARGET_TYPE} Events Accepted: {new_discoveries_count}")
print(f"Saved to: {silver_y_path}")
print(f"=========================================")