import torch
import numpy as np
import matplotlib.pyplot as plt
from train_lstm import PentaLSTM
from scipy.ndimage import label

print("1. Loading Data and Models...")
NIGHT_TO_TEST = 2  # Change this to 1 or 2 to switch patients!

print(f"1. Loading Data for Night {NIGHT_TO_TEST}...")
# Dynamically load the specific night's files based on your folder structure
X = np.load(f'X_{NIGHT_TO_TEST}.npy')
segment_times = np.load(f'segment_times_n{NIGHT_TO_TEST}.npy')

Y_true_CA = np.load(f'Y_CA_{NIGHT_TO_TEST}.npy')
Y_true_OSA = np.load(f'Y_OSA_{NIGHT_TO_TEST}.npy')
model_ca = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model_ca.load_state_dict(torch.load('penta_lstm_CA_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model_ca.eval()

model_osa = PentaLSTM(input_size=6, hidden_size=128, num_layers=2)
model_osa.load_state_dict(torch.load('penta_lstm_OSA_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model_osa.eval()

print("2. Running AI Prediction on ALL segments...")
input_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    # Get probabilities for Class 1 (Apnea) for both models
    probs_ca = torch.softmax(model_ca(input_tensor), dim=1)[:, 1, :].numpy() 
    probs_osa = torch.softmax(model_osa(input_tensor), dim=1)[:, 1, :].numpy() 

print("3. Stitching overlapping segments back together...")
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

# Convert averaged probabilities back to binary decisions (Threshold > 0.5)
full_classes_ca = (full_probs_ca > 0.5).astype(int)
full_classes_osa = (full_probs_osa > 0.5).astype(int)

print("4. Applying 10-second clinical cleanup filter...")
labeled_ca, num_ca = label(full_classes_ca == 1)
for i in range(1, num_ca + 1):
    if (labeled_ca == i).sum() < 250:
        full_classes_ca[labeled_ca == i] = 0

labeled_osa, num_osa = label(full_classes_osa == 1)
for i in range(1, num_osa + 1):
    if (labeled_osa == i).sum() < 250:
        full_classes_osa[labeled_osa == i] = 0

print("5. Plotting the full night timeline...")
plt.figure(figsize=(20, 6)) 

plt.plot(full_time, full_pflow, label='PFlow', color='blue', alpha=0.5, linewidth=0.5)

plt.fill_between(full_time, 0, full_y_ca, color='gray', alpha=0.4, label='Doctor: CA')
plt.fill_between(full_time, 0, full_y_osa, color='cyan', alpha=0.4, label='Doctor: OSA')

plt.plot(full_time, full_classes_ca, color='red', linewidth=1.5, label='AI: CA')
plt.plot(full_time, full_classes_osa * 1.05, color='orange', linewidth=1.5, label='AI: OSA')

plt.title("Full Night Dual-Model AI Apnea Detection")
plt.xlabel("Real Elapsed Time (Seconds)")
plt.ylabel("Apnea Events")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()