import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Import the Actor-Critic brain
from actor_critic_lstm import ActorCriticLSTM

print("1. Loading Data and Multi-Class RLHF Agent...")
# We use a single night for visualization, NOT the Master dataset!
X = np.load('X_train_PentaLSTM.npy')
Y_true = np.load('Y_train_Labels.npy')
segment_times = np.load('segment_times.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCriticLSTM(input_size=6, hidden_size=128, num_layers=2).to(device)
model.load_state_dict(torch.load('rlhf_penta_lstm_weights.pth', map_location=device, weights_only=True))
model.eval()

print("2. Running RLHF Prediction on ALL segments...")
input_tensor = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    action_logits, _ = model(input_tensor)
    # Convert Actor logits to probabilities for all 3 classes
    probabilities = torch.softmax(action_logits, dim=-1).cpu().numpy() 

print("3. Stitching the overlapping segments back together...")
win_samples = 960
step_samples = 640  # 20 seconds at 32 Hz
num_segments = len(X)

total_samples = step_samples * (num_segments - 1) + win_samples

full_pflow = np.zeros(total_samples)
full_y = np.zeros(total_samples)
full_time = np.zeros(total_samples)
full_preds_probs = np.zeros((3, total_samples)) # Matrix for 3 classes
overlap_counts = np.zeros(total_samples)

for i in range(num_segments):
    start_idx = i * step_samples
    end_idx = start_idx + win_samples
    
    full_pflow[start_idx:end_idx] += X[i, :, 0]
    full_time[start_idx:end_idx] = segment_times[i] # Use real timestamps!
    full_y[start_idx:end_idx] = np.maximum(full_y[start_idx:end_idx], Y_true[i].flatten())
    
    # Add probabilities for all 3 classes
    full_preds_probs[:, start_idx:end_idx] += probabilities[i].T
    overlap_counts[start_idx:end_idx] += 1

full_pflow /= overlap_counts
full_preds_probs /= overlap_counts

# Decide the final winner (0, 1, or 2) for every single timestep
full_classes = np.argmax(full_preds_probs, axis=0)

# --- THE 10-SECOND CLEANUP FILTER ---
print("4. Applying the 10-second Clinical Cleanup Filter...")

# Clean up CA (Class 1)
labeled_ca, num_ca = label(full_classes == 1)
for i in range(1, num_ca + 1):
    if (labeled_ca == i).sum() < 10:
        full_classes[labeled_ca == i] = 0

# Clean up OSA (Class 2)
labeled_oh, num_oh = label(full_classes == 2)
for i in range(1, num_oh + 1):
    if (labeled_oh == i).sum() < 10:
        full_classes[labeled_oh == i] = 0

# Recalculate final counts
_, final_ca_count = label(full_classes == 1)
_, final_oh_count = label(full_classes == 2)
print(f"✅ Total valid CA detected by RLHF Agent: {final_ca_count}")
print(f"✅ Total valid OSA detected by RLHF Agent: {final_oh_count}")

print("5. Plotting the full night timeline...")
plt.figure(figsize=(20, 6)) 

plt.plot(full_time, full_pflow, label='PFlow', color='blue', alpha=0.5, linewidth=0.5)

# Fill Doctor Labels
plt.fill_between(full_time, 0, (full_y == 1).astype(int), color='gray', alpha=0.4, label='Doctor: CA')
plt.fill_between(full_time, 0, (full_y == 2).astype(int), color='cyan', alpha=0.4, label='Doctor: OH (OSA)')

# Plot AI Predictions
plt.plot(full_time, (full_classes == 1).astype(int), color='red', linewidth=1.5, label='RLHF: CA')
plt.plot(full_time, (full_classes == 2).astype(int) * 1.05, color='orange', linewidth=1.5, label='RLHF: OH (OSA)')

plt.title("Full Night Multi-Class RLHF Apnea Detection")
plt.xlabel("Real Elapsed Time (Seconds)")
plt.ylabel("Apnea Events")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()