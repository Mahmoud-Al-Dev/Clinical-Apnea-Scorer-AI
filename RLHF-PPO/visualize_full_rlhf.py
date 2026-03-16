import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from actor_critic_lstm import ActorCriticLSTM

# ==========================================
# --- USER CONTROLS ---
# ==========================================
NIGHT_TO_TEST = 1       # Change to 1 or 2 to switch patients!
INPUT_CHANNELS = 6      # Ensure this matches your model architecture (6 or 7)
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
batch_size = 64  # Process in small chunks to save GPU memory
num_segments = len(X)
probs_ca = np.zeros((num_segments, 960))
probs_osa = np.zeros((num_segments, 960))

with torch.no_grad():
    for i in range(0, num_segments, batch_size):
        end_idx = min(i + batch_size, num_segments)
        # Only move the current small batch to the GPU
        batch_x = torch.tensor(X[i:end_idx], dtype=torch.float32).to(device)
        
        # Get CA probabilities
        logits_ca, _ = model_ca(batch_x)
        probs_ca[i:end_idx] = torch.softmax(logits_ca, dim=-1)[:, :, 1].cpu().numpy()
        
        # Get OSA probabilities
        logits_osa, _ = model_osa(batch_x)
        probs_osa[i:end_idx] = torch.softmax(logits_osa, dim=-1)[:, :, 1].cpu().numpy()
        
        # Optional: Print progress so you know it's working
        if i % 512 == 0:
            print(f"   Processed {i}/{num_segments} segments...")

print("4. Stitching the overlapping segments back together...")
win_samples = 960
step_samples = 640 
num_segments = len(X)
total_samples = step_samples * (num_segments - 1) + win_samples

full_pflow = np.zeros(total_samples)
full_time = np.zeros(total_samples)

full_y_ca = np.zeros(total_samples)
full_y_osa = np.zeros(total_samples)

full_probs_ca = np.zeros(total_samples)
full_probs_osa = np.zeros(total_samples)
overlap_counts = np.zeros(total_samples)

for i in range(num_segments):
    start_idx = i * step_samples
    end_idx = start_idx + win_samples
    
    full_pflow[start_idx:end_idx] += X[i, :, 0]
    full_time[start_idx:end_idx] = segment_times[i] 
    
    full_y_ca[start_idx:end_idx] = np.maximum(full_y_ca[start_idx:end_idx], Y_true_CA[i].flatten())
    full_y_osa[start_idx:end_idx] = np.maximum(full_y_osa[start_idx:end_idx], Y_true_OSA[i].flatten())
    
    full_probs_ca[start_idx:end_idx] += probs_ca[i]
    full_probs_osa[start_idx:end_idx] += probs_osa[i]
    overlap_counts[start_idx:end_idx] += 1

full_pflow /= overlap_counts
full_probs_ca /= overlap_counts
full_probs_osa /= overlap_counts

# Convert back to binary predictions based on 0.5 threshold
full_classes_ca = (full_probs_ca > 0.5).astype(int)
full_classes_osa = (full_probs_osa > 0.5).astype(int)

print("5. Applying the 10-second Clinical Cleanup Filter...")
labeled_ca, num_ca = label(full_classes_ca == 1)
for i in range(1, num_ca + 1):
    if (labeled_ca == i).sum() < 250: # Adjust if your sampling rate differs! (250 is ~8s at 32Hz)
        full_classes_ca[labeled_ca == i] = 0

labeled_osa, num_osa = label(full_classes_osa == 1)
for i in range(1, num_osa + 1):
    if (labeled_osa == i).sum() < 250:
        full_classes_osa[labeled_osa == i] = 0

print("6. Plotting the full night timeline...")
plt.figure(figsize=(20, 6)) 

plt.plot(full_time, full_pflow, label='PFlow', color='blue', alpha=0.5, linewidth=0.5)

plt.fill_between(full_time, 0, full_y_ca, color='gray', alpha=0.4, label='Doctor: CA')
plt.fill_between(full_time, 0, full_y_osa, color='cyan', alpha=0.4, label='Doctor: OSA')

plt.plot(full_time, full_classes_ca, color='red', linewidth=1.5, label='RLHF: CA')
plt.plot(full_time, full_classes_osa * 1.05, color='orange', linewidth=1.5, label='RLHF: OSA')

plt.title(f"Full Night Dual-Model RLHF Apnea Detection - Night {NIGHT_TO_TEST}")
plt.xlabel("Real Elapsed Time (Seconds)")
plt.ylabel("Apnea Events")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()