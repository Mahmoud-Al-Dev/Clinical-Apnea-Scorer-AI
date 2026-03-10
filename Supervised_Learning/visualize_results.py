import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train_lstm import PentaLSTM 

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TIME_SEC = 10510  # Type the exact second you want to check!
# ==========================================

print("1. Loading Data...")
X = np.load('X_train_PentaLSTM.npy')
Y_true = np.load('Y_train_Labels.npy')
segment_times = np.load('segment_times.npy')

# 2. Dynamically find the correct Segment Index using actual timestamps
start_times = segment_times[:, 0]
seg_idx = np.argmin(np.abs(start_times - TARGET_TIME_SEC))

real_start = segment_times[seg_idx, 0]
real_end = segment_times[seg_idx, -1]
print(f"✅ Target time {TARGET_TIME_SEC}s found in Segment {seg_idx} ({real_start:.1f}s to {real_end:.1f}s)")

print("3. Loading Multi-Class Model...")
model = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load('penta_lstm_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

print("4. Making Prediction...")
input_tensor = torch.tensor(X[seg_idx], dtype=torch.float32).unsqueeze(0)
 
with torch.no_grad():
    raw_output = model(input_tensor) # Shape: (1, 3, 960)
    # argmax(dim=1) picks the class (0, 1, or 2) with the highest score
    predictions = torch.argmax(raw_output, dim=1).squeeze().numpy()

print("5. Plotting Results...")
time_axis = segment_times[seg_idx]
y_flat = Y_true[seg_idx].flatten()

plt.figure(figsize=(12, 6))
plt.plot(time_axis, X[seg_idx, :, 0], label='PFlow (Input)', color='blue', alpha=0.5)

# --- PLOT THE DOCTOR'S TARGETS ---
plt.fill_between(time_axis, 0, (y_flat == 1).astype(int), color='gray', alpha=0.3, label='Doctor: CA')
plt.fill_between(time_axis, 0, (y_flat == 2).astype(int), color='cyan', alpha=0.3, label='Doctor: OH (OSA)')

# --- PLOT THE AI PREDICTIONS ---
plt.plot(time_axis, (predictions == 1).astype(int), color='red', linewidth=2.5, label='AI Pred: CA')
# We multiply the OH line by 1.05 so if the AI guesses both at the same time, the lines don't completely hide each other
plt.plot(time_axis, (predictions == 2).astype(int) * 1.05, color='orange', linewidth=2.5, label='AI Pred: OH (OSA)')

plt.title(f"Multi-Class AI Apnea Detection - Segment {seg_idx}")
plt.xlabel("Real Elapsed Time (Seconds)")
plt.ylabel("Normalized Amplitude / Class Output")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()