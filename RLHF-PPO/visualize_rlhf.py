import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from actor_critic_lstm import ActorCriticLSTM

# ==========================================
# --- USER CONTROLS ---
# ==========================================
TARGET_TIME_SEC = 10923
NIGHT_TO_TEST = 2       # Change to 1 or 2 to switch patients!
INPUT_CHANNELS = 6     # Ensure this matches your model architecture (6 or 7)
# ==========================================

print(f"1. Loading Data for Night {NIGHT_TO_TEST}...")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("3. Loading Dual Binary RLHF Agents...")
# Load CA Agent
model_ca = ActorCriticLSTM(input_size=INPUT_CHANNELS, hidden_size=128, num_layers=2).to(device)
model_ca.load_state_dict(torch.load('rlhf_penta_lstm_CA_weights.pth', map_location=device, weights_only=True))
model_ca.eval()

# Load OSA Agent
model_osa = ActorCriticLSTM(input_size=INPUT_CHANNELS, hidden_size=128, num_layers=2).to(device)
model_osa.load_state_dict(torch.load('rlhf_penta_lstm_OSA_weights.pth', map_location=device, weights_only=True))
model_osa.eval()

print("4. Making Parallel Predictions...")
input_tensor = torch.tensor(X[seg_idx], dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    # CA Prediction
    logits_ca, _ = model_ca(input_tensor)
    pred_ca = torch.argmax(torch.softmax(logits_ca, dim=-1), dim=-1).squeeze().cpu().numpy()
    
    # OSA Prediction
    logits_osa, _ = model_osa(input_tensor)
    pred_osa = torch.argmax(torch.softmax(logits_osa, dim=-1), dim=-1).squeeze().cpu().numpy()

print("5. Plotting Results...")
time_axis = segment_times[seg_idx]
y_ca_flat = Y_true_CA[seg_idx].flatten()
y_osa_flat = Y_true_OSA[seg_idx].flatten()

plt.figure(figsize=(12, 6))
plt.plot(time_axis, X[seg_idx, :, 0], label='PFlow (Input Signal)', color='blue', alpha=0.5)

# --- PLOT THE DOCTOR'S TARGETS ---
plt.fill_between(time_axis, 0, y_ca_flat, color='gray', alpha=0.3, label='Doctor: CA')
plt.fill_between(time_axis, 0, y_osa_flat, color='cyan', alpha=0.3, label='Doctor: OSA')

# --- PLOT THE AI PREDICTIONS ---
plt.plot(time_axis, pred_ca, color='red', linewidth=2.5, label='RLHF Pred: CA')
plt.plot(time_axis, pred_osa * 1.05, color='orange', linewidth=2.5, label='RLHF Pred: OSA')

plt.title(f"Dual-Model RLHF Agent Apnea Detection - Night {NIGHT_TO_TEST}, Segment {seg_idx}")
plt.xlabel("Real Elapsed Time (Seconds)")
plt.ylabel("Normalized Amplitude / Class Output")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()