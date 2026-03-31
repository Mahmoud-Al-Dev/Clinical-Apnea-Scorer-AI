import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from train_lstm import ConvLSTM 

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TYPE = 'OSA'  
NIGHT_TO_TEST = 3
TOP_N_TO_REVIEW = 100 
# ==========================================

print(f"Loading Data for Night {NIGHT_TO_TEST} ({TARGET_TYPE})...")

X = np.load(f'X_{NIGHT_TO_TEST}.npy')
Y_true = np.load(f'Y_{TARGET_TYPE}_{NIGHT_TO_TEST}.npy')
segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')
flags = np.load(f'cleanlab_flags_{TARGET_TYPE}_n{NIGHT_TO_TEST}.npy')

total_flags = len(flags)
review_limit = min(TOP_N_TO_REVIEW, total_flags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Model to show exact AI predictions...")

model = ConvLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
model.load_state_dict(torch.load(f'penta_lstm_{TARGET_TYPE}_weights.pth', map_location=device, weights_only=True))
model.eval()

human_indices = [0, 1, 2, 5] 
ai_indices = [0, 3, 4, 5, 6, 7] 

rlhf_queue = []

for i in range(review_limit):
    seg_idx = int(flags[i])
    raw_segment = X[seg_idx]
    
    time_axis = segment_times[seg_idx]
    vis_signals = raw_segment[:, human_indices]
    ratio_signal = raw_segment[:, 6]
    doc_label = Y_true[seg_idx].flatten()
    
    # Get frame-by-frame AI predictions
    ai_obs = torch.tensor(raw_segment[:, ai_indices], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(ai_obs)
        # Softmax on dim=1 for ConvLSTM shape (Batch, Classes, Timesteps)
        probs = torch.softmax(logits, dim=1)[0, 1, :].cpu().numpy()
        ai_preds = (probs > 0.5).astype(int)
    
    doc_has_event = np.any(doc_label == 1)
    mistake_type = "Doctor False Alarm" if doc_has_event else "Doctor Missed Event"
    
    print(f"\n--- Reviewing Flag {i+1}/{review_limit} (Segment {seg_idx}) ---")
    print(f"Cleanlab Severity: {mistake_type}")
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Cleanlab Flag {i+1} | Segment {seg_idx} | {mistake_type}", fontsize=14)
    
    # 1. PFlow with Comparison
    axes[0].plot(time_axis, vis_signals[:, 0], color='blue', alpha=0.7)
    axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                         where=(doc_label == 1), color='cyan', alpha=0.4, label='Doctor Label')
    axes[0].fill_between(time_axis, np.min(vis_signals[:, 0]), np.max(vis_signals[:, 0]), 
                         where=(ai_preds == 1), color='red', alpha=0.4, label='AI Prediction')
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
    
    print("Options: [Enter] Next | [r] Add to RLHF Queue | [q] Quit")
    user_input = input("Choice: ").strip().lower()
    plt.close(fig)
    
    if user_input == 'r':
        print(f"Flagged Segment {seg_idx} for RLHF.")
        rlhf_queue.append(seg_idx)
    elif user_input == 'q':
        break

if len(rlhf_queue) > 0:
    queue_path = f'rlhf_ambiguous_queue_{TARGET_TYPE}_n{NIGHT_TO_TEST}.npy'
    np.save(queue_path, np.array(rlhf_queue))
    print(f"\n💾 Saved {len(rlhf_queue)} ambiguous segments to {queue_path}")