import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train_lstm import PentaLSTM 

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TIME_SEC =  10220
NIGHT_TO_TEST = 2  # Change this to 1 or 2 to switch patients!
# ==========================================

print(f"1. Loading Data for Night {NIGHT_TO_TEST}...")
# Dynamically load the specific night's files based on your folder structure
X = np.load(f'X_{NIGHT_TO_TEST}.npy')
segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')

Y_true_CA = np.load(f'Y_CA_{NIGHT_TO_TEST}.npy')
Y_true_OSA = np.load(f'Y_OSA_{NIGHT_TO_TEST}.npy')

# 2. Dynamically find the correct Segment Index
start_times = segment_times[:, 0]
seg_idx = np.argmin(np.abs(start_times - TARGET_TIME_SEC))
real_start = segment_times[seg_idx, 0]
real_end = segment_times[seg_idx, -1]
print(f"✅ Target time {TARGET_TIME_SEC}s found in Segment {seg_idx} ({real_start:.1f}s to {real_end:.1f}s)")

print("3. Loading Binary Models...")
# Initialize TWO models (they default to 2-class output now based on your updated train_lstm.py)
model_ca = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model_ca.load_state_dict(torch.load('penta_lstm_CA_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model_ca.eval()

model_osa = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model_osa.load_state_dict(torch.load('penta_lstm_OSA_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model_osa.eval()

print("4. Making Parallel Predictions...")
input_tensor = torch.tensor(X[seg_idx], dtype=torch.float32).unsqueeze(0)
 
with torch.no_grad():
    # Get CA Prediction
    raw_ca = model_ca(input_tensor) 
    pred_ca = torch.argmax(raw_ca, dim=1).squeeze().numpy()

    # Get OSA Prediction
    raw_osa = model_osa(input_tensor) 
    pred_osa = torch.argmax(raw_osa, dim=1).squeeze().numpy()

print("5. Plotting Results...")
time_axis = segment_times[seg_idx]
y_ca_flat = Y_true_CA[seg_idx].flatten()
y_osa_flat = Y_true_OSA[seg_idx].flatten()

plt.figure(figsize=(12, 6))
plt.plot(time_axis, X[seg_idx, :, 0], label='PFlow (Input)', color='blue', alpha=0.5)

# --- PLOT THE DOCTOR'S TARGETS ---
plt.fill_between(time_axis, 0, y_ca_flat, color='gray', alpha=0.3, label='Doctor: CA')
plt.fill_between(time_axis, 0, y_osa_flat, color='cyan', alpha=0.3, label='Doctor: OSA')

# --- PLOT THE AI PREDICTIONS ---
plt.plot(time_axis, pred_ca, color='red', linewidth=2.5, label='AI Pred: CA')
# We multiply the OSA line by 1.05 so if they overlap, we can still see both
plt.plot(time_axis, pred_osa * 1.05, color='orange', linewidth=2.5, label='AI Pred: OSA')

plt.title(f"Dual-Model AI Apnea Detection - Segment {seg_idx}")
plt.xlabel("Real Elapsed Time (Seconds)")
plt.ylabel("Normalized Amplitude / Class Output")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()