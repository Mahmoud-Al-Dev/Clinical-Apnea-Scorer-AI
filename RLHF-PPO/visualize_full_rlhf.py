import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from actor_critic_lstm import ActorCriticLSTM

# ==========================================
# --- USER CONTROLS ---
# ==========================================
NIGHT_TO_TEST = 2       # Change to 1 or 2 to switch patients!
INPUT_CHANNELS = 6      # Ensure this matches your model architecture (6 or 7)

# NEW: Define your exact time window here. 
# Set both to None to plot the entire night!
WINDOW_START_SEC = 6500  
WINDOW_END_SEC = 7000
# ==========================================

print(f"1. Loading Data for Night {NIGHT_TO_TEST}...")
X = np.load(f'X_{NIGHT_TO_TEST}.npy')
segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')
Y_true_CA = np.load(f'Y_CA_{NIGHT_TO_TEST}.npy')
Y_true_OSA = np.load(f'Y_OSA_{NIGHT_TO_TEST}.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("2. Loading Dual Binary RLHF Agents...")
model_ca = ActorCriticLSTM(input_size=INPUT_CHANNELS, hidden_size=128, num_layers=2).to(device)
model_ca.load_state_dict(torch.load('rlhf_penta_lstm_CA_weights.pth', map_location=device, weights_only=True))
model_ca.eval()

model_osa = ActorCriticLSTM(input_size=INPUT_CHANNELS, hidden_size=128, num_layers=2).to(device)
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
        batch_x = torch.tensor(X[i:end_idx], dtype=torch.float32).to(device)
        
        logits_ca, _ = model_ca(batch_x)
        probs_ca[i:end_idx] = torch.softmax(logits_ca, dim=-1)[:, :, 1].cpu().numpy()
        
        logits_osa, _ = model_osa(batch_x)
        probs_osa[i:end_idx] = torch.softmax(logits_osa, dim=-1)[:, :, 1].cpu().numpy()
        
        if i % 512 == 0:
            print(f"   Processed {i}/{num_segments} segments...")

print("4. Stitching the overlapping segments back together (All Channels)...")
win_samples = 960
step_samples = 640 
num_segments = len(X)
total_samples = step_samples * (num_segments - 1) + win_samples

# CHANGED: Create a 2D array to hold all 6 channels instead of just PFlow
full_features = np.zeros((total_samples, INPUT_CHANNELS))
full_time = np.zeros(total_samples)

full_y_ca = np.zeros(total_samples)
full_y_osa = np.zeros(total_samples)

full_probs_ca = np.zeros(total_samples)
full_probs_osa = np.zeros(total_samples)
overlap_counts = np.zeros(total_samples)

for i in range(num_segments):
    start_idx = i * step_samples
    end_idx = start_idx + win_samples
    
    # CHANGED: Stitch all channels at once
    full_features[start_idx:end_idx, :] += X[i, :, :]
    full_time[start_idx:end_idx] = segment_times[i] 
    
    full_y_ca[start_idx:end_idx] = np.maximum(full_y_ca[start_idx:end_idx], Y_true_CA[i].flatten())
    full_y_osa[start_idx:end_idx] = np.maximum(full_y_osa[start_idx:end_idx], Y_true_OSA[i].flatten())
    
    full_probs_ca[start_idx:end_idx] += probs_ca[i]
    full_probs_osa[start_idx:end_idx] += probs_osa[i]
    overlap_counts[start_idx:end_idx] += 1

# Average the overlapping areas
full_features /= overlap_counts[:, None] # Broadcast divide across all 6 channels
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
plot_features = full_features[mask, :] # Mask all 6 channels
plot_y_ca = full_y_ca[mask]
plot_y_osa = full_y_osa[mask]
plot_classes_ca = full_classes_ca[mask]
plot_classes_osa = full_classes_osa[mask]

# =========================================================

print("7. Plotting the multi-channel timeline...")
channel_names = ['PFlow_Clean', 'Abdomen_Clean', 'Ratio', 'SaO2_Deriv', 'PFlow_Var', 'Vitalog2']

# Create a stacked plot with 6 rows
fig, axes = plt.subplots(INPUT_CHANNELS, 1, figsize=(20, 3 * INPUT_CHANNELS), sharex=True)

window_title = f"({WINDOW_START_SEC}s - {WINDOW_END_SEC}s)" if WINDOW_START_SEC else "(Full Night)"
fig.suptitle(f"Full Night Dual-Model RLHF Apnea Detection - Night {NIGHT_TO_TEST} {window_title}", fontsize=16)

for i, ax in enumerate(axes):
    sig = plot_features[:, i]
    
    # Plot the raw feature signal
    ax.plot(plot_time, sig, color='blue', alpha=0.6, linewidth=1.0)
    
    # Plot the Doctor's Targets (Shaded gray/cyan blocks extending from top to bottom)
    ax.fill_between(plot_time, np.min(sig), np.max(sig), where=(plot_y_ca == 1), color='gray', alpha=0.3, label='Doctor: CA' if i==0 else "")
    ax.fill_between(plot_time, np.min(sig), np.max(sig), where=(plot_y_osa == 1), color='cyan', alpha=0.3, label='Doctor: OSA' if i==0 else "")
    
    # Plot AI Predictions (Scaled to fit the specific channel's height cleanly)
    sig_max = np.max(sig)
    ax.plot(plot_time, plot_classes_ca * sig_max * 0.8, color='red', linewidth=2.0, label='RLHF: CA' if i==0 else "")
    ax.plot(plot_time, plot_classes_osa * sig_max * 0.9, color='orange', linewidth=2.0, label='RLHF: OSA' if i==0 else "")
    
    ax.set_ylabel(channel_names[i], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend(loc='upper right')

plt.xlabel("Real Elapsed Time (Seconds)", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for the suptitle
plt.show()