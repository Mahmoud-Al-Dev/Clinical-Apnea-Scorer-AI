import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os
from train_lstm import ConvLSTM 

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TYPE = 'CA'  
NIGHT_TO_TEST = 25

# Review Selection Controls
REVIEW_ALL = True             # Set to True to review EVERY flag Cleanlab found
REVIEW_TOP_CONFIDENT = 100        # "Arrogant AI Mistakes" (High Confidence)
REVIEW_BOTTOM_UNCERTAIN = 0     # "Confused Edge Cases" (Low Confidence / ~50%)
# =========================================y=n

print(f"--- Launching Interactive Cleanlab Reviewer for Night {NIGHT_TO_TEST} ({TARGET_TYPE}) ---")

# 1. Load Data
X = np.load(os.path.join('Nights', f'X_{NIGHT_TO_TEST}.npy'))
Y_original = np.load(os.path.join('Nights', f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}.npy'))
segment_times = np.load(os.path.join('Nights', f'segment_times_n{NIGHT_TO_TEST}.npy'))

# Load Cleanlab Flags
flags_path = f'cleanlab_flags_{TARGET_TYPE}_n{NIGHT_TO_TEST}.npy'
try:
    flags = np.load(flags_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find {flags_path}. Did you run run_cleanlab.py first?")
    exit()

total_flags = len(flags)

# --- NEW SELECTION LOGIC ---
if REVIEW_ALL:
    print(f"🚨 REVIEW_ALL is True. Reviewing all {total_flags} flagged segments...")
    indices_to_review = flags
else:
    top_limit = min(REVIEW_TOP_CONFIDENT, total_flags)
    bottom_limit = min(REVIEW_BOTTOM_UNCERTAIN, max(0, total_flags - top_limit))
    
    top_flags = flags[:top_limit]
    bottom_flags = flags[-bottom_limit:] if bottom_limit > 0 else []
    
    # Combine and remove duplicates just in case
    indices_to_review = np.unique(np.concatenate((top_flags, bottom_flags))).astype(int)
    print(f"🚨 Flagged {total_flags} segments. Reviewing {len(indices_to_review)} (Top {top_limit} Confident, Bottom {bottom_limit} Uncertain)...")

# 2. Setup Silver Standard Working Array
silver_y_path = os.path.join('Nights', f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}_SILVER.npy')
if os.path.exists(silver_y_path):
    print("📂 Found existing SILVER standard. Resuming session...")
    Y_working = np.load(silver_y_path)
else:
    print("📄 Starting fresh from original clinical labels...")
    Y_working = np.copy(Y_original)

# 3. Load SFT Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Loading AI Model to show exact predictions...")
model = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
model.load_state_dict(torch.load(f'penta_lstm_{TARGET_TYPE}_weights.pth', map_location=device, weights_only=True))
model.eval()

human_indices = [0, 1, 2, 5] 
ai_indices = [0, 3, 4, 5, 6, 7] 

rlhf_queue = []
corrections_made = 0

for i, seg_idx in enumerate(indices_to_review):
    raw_segment = X[seg_idx]
    
    time_axis = segment_times[seg_idx]
    vis_signals = raw_segment[:, human_indices]
    ratio_signal = raw_segment[:, 6]
    doc_label = Y_original[seg_idx].flatten()
    
    # --- GET AI PREDICTIONS & APPLY 10s FILTER ---
    ai_obs = torch.tensor(raw_segment[:, ai_indices], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(ai_obs)
        probs = torch.softmax(logits, dim=1)[0, 1, :].cpu().numpy()
        
        # 10-Second (320 frame) Rule
        binary_preds = (probs > 0.5).astype(int)
        labeled_array, num_features = label(binary_preds)
        for j in range(1, num_features + 1):
            if np.sum(labeled_array == j) < 320: 
                binary_preds[labeled_array == j] = 0 
                
        ai_preds = binary_preds
    
    doc_has_event = np.any(doc_label == 1)
    mistake_type = "Doctor False Alarm" if doc_has_event else "Doctor Missed Event"
    
    print(f"\n--- Reviewing Flag {i+1}/{len(indices_to_review)} (Segment {seg_idx}) ---")
    print(f"Cleanlab Flag Type: {mistake_type}")
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Cleanlab Flag {i+1} | Segment {seg_idx} | {mistake_type}", fontsize=14)
    
    axes[0].plot(time_axis, vis_signals[:, 0], color='blue', alpha=0.7)
    axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                         where=(doc_label == 1), color='cyan', alpha=0.4, label='Doctor Label')
    axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                         where=(ai_preds == 1), color='red', alpha=0.4, label='AI Prediction')
    axes[0].set_ylabel('PFlow_Clean')
    axes[0].legend(loc='upper right')

    axes[1].plot(time_axis, vis_signals[:, 1], color='green', alpha=0.7)
    axes[1].set_ylabel('Thorax_Clean')

    axes[2].plot(time_axis, vis_signals[:, 2], color='purple', alpha=0.7)
    axes[2].set_ylabel('Abdomen_Clean')

    axes[3].plot(time_axis, ratio_signal, color='darkred', alpha=0.8)
    axes[3].set_ylabel('Ratio')

    axes[4].plot(time_axis, vis_signals[:, 3], color='magenta', alpha=0.7)
    axes[4].set_ylabel('SaO2_Smooth')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.xlabel("Real Elapsed Time (Seconds)")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    # --- INTERACTIVE MENU ---
    print("Options:")
    print("  [y] Accept AI's Red Label (Exactly as is)")
    print("  [c] Custom Edit (Click START and END points directly on the graph)")
    print("  [n] Reject AI (Keeps Doctor's original label)")
    print("  [r] Add to RLHF Queue (For ambiguous cases)")
    print("  [q] Save & Quit")
    
    user_input = input("Choice: ").strip().lower()
    
    # NOTE: We DO NOT close the plot yet, because we might need it for clicks!
    
    if user_input == 'c':
        print("🖱️  Waiting for clicks... Please click 2 points on the graph (Start and End of the event).")
        try:
            # ginput(2) waits for exactly 2 mouse clicks on the active Matplotlib window
            clicks = plt.ginput(2, timeout=-1) 
            
            if len(clicks) == 2:
                # Extract the X coordinates (Time in seconds) from the clicks
                x1, _ = clicks[0]
                x2, _ = clicks[1]
                start_time, end_time = min(x1, x2), max(x1, x2)
                
                # Convert clicked seconds back into 960-frame indices
                start_idx = np.argmin(np.abs(time_axis - start_time))
                end_idx = np.argmin(np.abs(time_axis - end_time))
                
                # Create the custom binary mask
                custom_mask = np.zeros(960, dtype=int)
                custom_mask[start_idx:end_idx] = 1
                
                Y_working[seg_idx] = custom_mask.reshape(960, 1)
                print(f"  ✅ Custom event saved from {start_time:.1f}s to {end_time:.1f}s.")
                corrections_made += 1
            else:
                print("  ⚠️ Not enough clicks detected. Keeping original doctor label.")
        except Exception as e:
            print(f"  ⚠️ Error capturing clicks: {e}. Keeping original doctor label.")
        plt.close(fig)

    elif user_input == 'y':
        print("  ✅ Accepted! AI boundaries saved to Silver Standard.")
        Y_working[seg_idx] = ai_preds.reshape(960, 1)
        corrections_made += 1
        plt.close(fig)
        
    elif user_input == 'r':
        print(f"  🤔 Flagged Segment {seg_idx} for RLHF.")
        rlhf_queue.append(seg_idx)
        plt.close(fig)
        
    elif user_input == 'q':
        plt.close(fig)
        break
        
    else:
        print("  ❌ Rejected. Keeping original doctor label.")
        plt.close(fig)

# --- FINAL SAVE AND CLEANUP ---
np.save(silver_y_path, Y_working)
print(f"\n=========================================")
print("Session Complete!")
print(f"Total AI Corrections Accepted: {corrections_made}")
print(f"Updated Silver File Saved: {silver_y_path}")

if len(rlhf_queue) > 0:
    queue_path = f'rlhf_ambiguous_queue_{TARGET_TYPE}_n{NIGHT_TO_TEST}.npy'
    
    if os.path.exists(queue_path):
        existing_queue = list(np.load(queue_path))
        existing_queue.extend(rlhf_queue)
        final_queue = list(set(existing_queue))
    else:
        final_queue = rlhf_queue
        
    np.save(queue_path, np.array(final_queue))
    print(f"💾 Saved {len(rlhf_queue)} new ambiguous segments to {queue_path}!")
print(f"=========================================")