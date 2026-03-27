import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train_lstm import PentaLSTM 

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TIME_SEC =  13324
NIGHT_TO_TEST = 1  # Change this to 1 or 2 to switch patients!
# ==========================================

print(f"1. Loading Data for Night {NIGHT_TO_TEST}...")
X = np.load(f'X_{NIGHT_TO_TEST}.npy')
segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')

Y_true_CA = np.load(f'Y_CA_{NIGHT_TO_TEST}.npy')
Y_true_OSA = np.load(f'Y_OSA_{NIGHT_TO_TEST}.npy')

# The AI only sees the 6 mathematical features
ai_indices = [0, 3, 4, 5, 6, 7] 

# 2. Dynamically find the correct Segment Index
start_times = segment_times[:, 0]
seg_idx = np.argmin(np.abs(start_times - TARGET_TIME_SEC))
real_start = segment_times[seg_idx, 0]
real_end = segment_times[seg_idx, -1]
print(f"✅ Target time {TARGET_TIME_SEC}s found in Segment {seg_idx} ({real_start:.1f}s to {real_end:.1f}s)")

print("3. Loading Binary Models...")
model_ca = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model_ca.load_state_dict(torch.load('penta_lstm_CA_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model_ca.eval()

model_osa = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model_osa.load_state_dict(torch.load('penta_lstm_OSA_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model_osa.eval()

print("4. Making Parallel Predictions...")
ai_input_data = X[seg_idx, :, ai_indices] 
input_tensor = torch.tensor(ai_input_data, dtype=torch.float32).unsqueeze(0)

# CRITICAL FIX: Ensure the shape is (Batch, Sequence_Length, Features)
if input_tensor.shape[2] == 960:
    input_tensor = input_tensor.transpose(1, 2)
    
with torch.no_grad():
    raw_ca = model_ca(input_tensor) 
    pred_ca = torch.argmax(raw_ca, dim=1).squeeze().numpy()

    raw_osa = model_osa(input_tensor) 
    pred_osa = torch.argmax(raw_osa, dim=1).squeeze().numpy()

print("5. Plotting PSG Multi-View...")
time_axis = segment_times[seg_idx]
y_ca_flat = Y_true_CA[seg_idx].flatten()
y_osa_flat = Y_true_OSA[seg_idx].flatten()

# Extract the specific visual signals from the 8-channel array
pflow = X[seg_idx, :, 0]
thorax = X[seg_idx, :, 1]
abdomen = X[seg_idx, :, 2]
sao2 = X[seg_idx, :, 5]
ratio = X[seg_idx, :, 6]

# --- CREATE 5 SUBPLOTS ---
fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"Dual-Model AI Apnea Detection (PSG View) - Segment {seg_idx}", fontsize=14)

# 1. PFlow + Predictions + Labels
axs[0].plot(time_axis, pflow, label='PFlow_Clean', color='blue', alpha=0.7)
axs[0].fill_between(time_axis, np.min(pflow), y_ca_flat * np.max(pflow), color='gray', alpha=0.3, label='Doctor: CA')
axs[0].fill_between(time_axis, np.min(pflow), y_osa_flat * np.max(pflow), color='cyan', alpha=0.3, label='Doctor: OSA')
axs[0].plot(time_axis, pred_ca * np.max(pflow), color='red', linewidth=2.5, label='AI Pred: CA')
axs[0].plot(time_axis, pred_osa * (np.max(pflow) * 1.05), color='orange', linewidth=2.5, label='AI Pred: OSA')
axs[0].set_ylabel("PFlow / Preds")
axs[0].legend(loc='upper right', fontsize='small')

# 2. Thorax Effort
axs[1].plot(time_axis, thorax, color='green')
axs[1].set_ylabel("Thorax_Clean")

# 3. Abdomen Effort
axs[2].plot(time_axis, abdomen, color='purple')
axs[2].set_ylabel("Abdomen_Clean")

# 4. Effort Flow Ratio (The OSA Signature)
axs[3].plot(time_axis, ratio, color='darkred')
axs[3].set_ylabel("Effort_Flow_Ratio")

# 5. SaO2 Smooth
axs[4].plot(time_axis, sao2, color='magenta')
axs[4].set_ylabel("SaO2_Smooth")
axs[4].set_xlabel("Real Elapsed Time (Seconds)")

# Clean up the grid for all subplots
for ax in axs:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.95) # Make room for the main title
plt.show()