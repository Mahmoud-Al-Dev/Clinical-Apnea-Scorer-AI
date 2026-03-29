import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from actor_critic_lstm import ActorCriticLSTM

# ==========================================
# --- USER CONTROLS ---
# ==========================================
NIGHT_TO_TEST = 1       

MODEL_CHANNELS = 6      # What the AI was trained on
TOTAL_X_CHANNELS = 8    # What is actually stored in X_2.npy

# 1. THE AI SLICE (Must match self.ai_indices from apnea_env.py)
AI_INDICES = [0, 3, 4, 5, 6, 7]

# Set your exact time window here. Set to None to plot the entire night.
WINDOW_START_SEC = None
WINDOW_END_SEC = None

# 2. THE VISUALIZATION SLICE (The 4 you actually want to see)
VISUALIZE_INDICES = [0, 1, 2, 5]  

VISUALIZE_NAMES = [
    'PFlow_Clean',       # Relative Index 0 
    'Thorax_Clean',      # Relative Index 1 
    'Abdomen_Clean',     # Relative Index 2 
    'SaO2_Smooth',       # Relative Index 5 
]
# ==========================================

print(f"1. Loading Data for Night {NIGHT_TO_TEST}...")
X = np.load(f'X_{NIGHT_TO_TEST}.npy')
segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')
Y_true_CA = np.load(f'Y_CA_{NIGHT_TO_TEST}.npy')
Y_true_OSA = np.load(f'Y_OSA_{NIGHT_TO_TEST}.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("2. Loading Dual Binary RLHF Agents...")
# Model expects 6 channels!
model_ca = ActorCriticLSTM(input_size=MODEL_CHANNELS, hidden_size=128, num_layers=2).to(device)
model_ca.load_state_dict(torch.load('rlhf_penta_lstm_CA_weights.pth', map_location=device, weights_only=True))
model_ca.eval()

model_osa = ActorCriticLSTM(input_size=MODEL_CHANNELS, hidden_size=128, num_layers=2).to(device)
model_osa.load_state_dict(torch.load('rlhf_penta_lstm_OSA_weights.pth', map_location=device, weights_only=True))
model_osa.eval()

print("3. Running RLHF Prediction on ALL segments (Batch Processing)...")
batch_size = 64  
num_segments = len(X)
probs_ca = np.zeros((num_segments, 960))
probs_osa = np.zeros((num_segments, 960))

with torch.no_grad():
    for i in range(0, num_segments, batch_size):
        end_idx = min(i + batch_size, num_segments)
        
        # FIX: Slice X down to the 6 channels the AI expects before making the tensor!
        x_sliced_for_ai = X[i:end_idx][:, :, AI_INDICES] 
        batch_x = torch.tensor(x_sliced_for_ai, dtype=torch.float32).to(device)
        
        logits_ca, _ = model_ca(batch_x)
        probs_ca[i:end_idx] = torch.softmax(logits_ca, dim=-1)[:, :, 1].cpu().numpy()
        
        logits_osa, _ = model_osa(batch_x)
        probs_osa[i:end_idx] = torch.softmax(logits_osa, dim=-1)[:, :, 1].cpu().numpy()
        
        if i % 512 == 0:
            print(f"   Processed {i}/{num_segments} segments...")

print("4. Stitching the overlapping segments back together (All Channels)...")
win_samples = 960
step_samples = 640 
total_samples = step_samples * (num_segments - 1) + win_samples

# Holds all 8 channels for stitching so the visualizer has access to everything
full_features = np.zeros((total_samples, TOTAL_X_CHANNELS))
full_time = np.zeros(total_samples)

full_y_ca = np.zeros(total_samples)
full_y_osa = np.zeros(total_samples)

full_probs_ca = np.zeros(total_samples)
full_probs_osa = np.zeros(total_samples)
overlap_counts = np.zeros(total_samples)

for i in range(num_segments):
    start_idx = i * step_samples
    end_idx = start_idx + win_samples
    
    # Stitch all 8 channels for the plot
    full_features[start_idx:end_idx, :] += X[i, :, :]
    full_time[start_idx:end_idx] = segment_times[i] 
    
    full_y_ca[start_idx:end_idx] = np.maximum(full_y_ca[start_idx:end_idx], Y_true_CA[i].flatten())
    full_y_osa[start_idx:end_idx] = np.maximum(full_y_osa[start_idx:end_idx], Y_true_OSA[i].flatten())
    
    full_probs_ca[start_idx:end_idx] += probs_ca[i]
    full_probs_osa[start_idx:end_idx] += probs_osa[i]
    overlap_counts[start_idx:end_idx] += 1

# Average the overlapping areas
full_features /= overlap_counts[:, None] 
full_probs_ca /= overlap_counts
full_probs_osa /= overlap_counts

full_classes_ca = (full_probs_ca > 0.5).astype(int)
full_classes_osa = (full_probs_osa > 0.5).astype(int)

print("5. Applying the 10-second Clinical Cleanup Filter...")
labeled_ca, num_ca = label(full_classes_ca == 1)
for i in range(1, num_ca + 1):
    if (labeled_ca == i).sum() < 320: 
        full_classes_ca[labeled_ca == i] = 0

labeled_osa, num_osa = label(full_classes_osa == 1)
for i in range(1, num_osa + 1):
    if (labeled_osa == i).sum() < 320:
        full_classes_osa[labeled_osa == i] = 0

# =========================================================
# 6. Apply Time Window Masking
# =========================================================
print("6. Filtering Time Window...")
mask = np.ones(len(full_time), dtype=bool)

if WINDOW_START_SEC is not None:
    mask &= (full_time >= WINDOW_START_SEC)
if WINDOW_END_SEC is not None:
    mask &= (full_time <= WINDOW_END_SEC)

# Slice all arrays using the mask
plot_time = full_time[mask]
plot_features = full_features[mask, :] 
plot_y_ca = full_y_ca[mask]
plot_y_osa = full_y_osa[mask]
plot_classes_ca = full_classes_ca[mask]
plot_classes_osa = full_classes_osa[mask]

# Extract and Print Exact AI Event Timestamps
print("\n" + "="*50)
print(f"--- AI Predicted Events in Window {WINDOW_START_SEC}s - {WINDOW_END_SEC}s ---")

def print_event_times(classes_array, time_array, label_name):
    labeled_events, num_events = label(classes_array == 1)
    print(f"\n{label_name} Predictions ({num_events}):")
    if num_events == 0:
        print("  None detected in this window.")
    for i in range(1, num_events + 1):
        event_times = time_array[labeled_events == i]
        start_t, end_t = event_times[0], event_times[-1]
        print(f"  - Event {i}: Start: {start_t:.2f}s | End: {end_t:.2f}s | Duration: {(end_t - start_t):.2f}s")

print_event_times(plot_classes_ca, plot_time, "CA (Red)")
print_event_times(plot_classes_osa, plot_time, "OSA (Orange)")
print("="*50 + "\n")

# =========================================================
# 7. Plotting ONLY the Chosen Channels
# =========================================================
num_plots = len(VISUALIZE_INDICES)
print(f"7. Plotting the {num_plots} chosen channels...")

fig, axes = plt.subplots(num_plots, 1, figsize=(20, 3 * num_plots), sharex=True)

window_title = f"({WINDOW_START_SEC}s - {WINDOW_END_SEC}s)" if WINDOW_START_SEC else "(Full Night)"
fig.suptitle(f"Full Night Dual-Model RLHF Apnea Detection - Night {NIGHT_TO_TEST} {window_title}", fontsize=16)

for i, ax in enumerate(axes):
    actual_channel_idx = VISUALIZE_INDICES[i]
    sig = plot_features[:, actual_channel_idx]
    
    ax.plot(plot_time, sig, color='blue', alpha=0.6, linewidth=1.0)
    
    ax.fill_between(plot_time, np.min(sig), np.max(sig), where=(plot_y_ca == 1), color='gray', alpha=0.3, label='Doctor: CA' if i==0 else "")
    ax.fill_between(plot_time, np.min(sig), np.max(sig), where=(plot_y_osa == 1), color='cyan', alpha=0.3, label='Doctor: OSA' if i==0 else "")
    
    sig_max = np.max(sig)
    ax.plot(plot_time, plot_classes_ca * sig_max * 0.8, color='red', linewidth=2.0, label='RLHF: CA' if i==0 else "")
    ax.plot(plot_time, plot_classes_osa * sig_max * 0.9, color='orange', linewidth=2.0, label='RLHF: OSA' if i==0 else "")
    
    ax.set_ylabel(VISUALIZE_NAMES[i], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend(loc='upper right')

plt.xlabel("Real Elapsed Time (Seconds)", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
plt.show()