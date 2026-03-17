import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, resample, detrend
from scipy.ndimage import uniform_filter1d, median_filter
from sklearn.preprocessing import StandardScaler


# =============================================================
# 1. LOAD & FILTER DATA (Using Pandas)
# =============================================================
print("1. Loading and filtering data...")
df = pd.read_csv('Data\ON030217-06(10000-11000s).csv', 
                 names=['PFlow', 'Thorax', 'Abdomen', 'SaO2', 'Vitalog1', 'Vitalog2', 'time_sec'])

window_start, window_end = 10000, 11000
data = df[(df['time_sec'] >= window_start) & (df['time_sec'] <= window_end)].copy()

# =============================================================
# 2. FEATURE EXTRACTION & ENGINEERING
# =============================================================
print("2. Extracting and Engineering features at 256 Hz...")
fs_original = 256
# A. Detrending
data['PFlow_Detrend'] = detrend(data['PFlow'])
data['Thorax_Detrend'] = detrend(data['Thorax'])
data['Abdomen_Detrend'] = detrend(data['Abdomen'])
data['Vitalog1_Med'] = median_filter(data['Vitalog1'], size=fs_original)
data['Vitalog2_Med'] = median_filter(data['Vitalog2'], size=fs_original)

# 2. Gentle Low-Pass Filter to remove the high-frequency fuzz (Cutoff at 2.0 Hz)
def apply_lowpass(signal, cutoff, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, signal)

data['Vitalog1_Clean'] = apply_lowpass(data['Vitalog1_Med'], 2.0, fs_original)
data['Vitalog2_Clean'] = apply_lowpass(data['Vitalog2_Med'], 2.0, fs_original)

# B. SaO2 Smoothing
data['SaO2_Smooth'] = uniform_filter1d(data['SaO2'], size=512)

# C. Butterworth Band-Pass Filters
def apply_bandpass(signal, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='bandpass')
    return filtfilt(b, a, signal)

data['PFlow_Clean'] = apply_bandpass(data['PFlow_Detrend'], 0.15, 0.7, fs_original)
data['Thorax_Clean'] = apply_bandpass(data['Thorax_Detrend'], 0.1, 0.3, fs_original)
data['Abdomen_Clean'] = apply_bandpass(data['Abdomen_Detrend'], 0.1, 0.3, fs_original)

# D. UPPER/LOWER ENVELOPE & AMPLITUDE WIDTH 
data['PFlow_Upper'] = np.abs(hilbert(data['PFlow_Clean']))
data['PFlow_Lower'] = -data['PFlow_Upper']
data['PFlow_Width'] = data['PFlow_Upper'] - data['PFlow_Lower'] 

data['Thorax_Upper'] = np.abs(hilbert(data['Thorax_Clean']))
data['Thorax_Lower'] = -data['Thorax_Upper']
data['Thorax_Width'] = data['Thorax_Upper'] - data['Thorax_Lower']

data['Abdomen_Upper'] = np.abs(hilbert(data['Abdomen_Clean']))
data['Abdomen_Lower'] = -data['Abdomen_Upper']
data['Abdomen_Width'] = data['Abdomen_Upper'] - data['Abdomen_Lower']

# E. FEATURE ENGINEERING
window_size = 5 * fs_original 
data['Thorax_Abdomen_Corr'] = data['Thorax_Width'].rolling(window=window_size, center=True).corr(data['Abdomen_Width'])
data['Thorax_Abdomen_Corr'] = data['Thorax_Abdomen_Corr'].bfill().ffill()
data['Effort_Flow_Ratio'] = (data['Thorax_Width'] + data['Abdomen_Width']) / (data['PFlow_Width'] + 0.001)

# --- F. ADVANCED PHYSIOLOGICAL FEATURES ---
print("   Calculating advanced physiological features...")

# 1. SaO2 Derivative (Rate of Desaturation)
data['SaO2_Deriv'] = np.gradient(data['SaO2_Smooth'])

# 2. Thorax-Abdomen Phase Angle (The OSA Gold Standard)
phase_thorax = np.unwrap(np.angle(hilbert(data['Thorax_Clean'])))
phase_abdomen = np.unwrap(np.angle(hilbert(data['Abdomen_Clean'])))
data['Phase_Angle'] = np.abs(phase_thorax - phase_abdomen)

# 3. Airflow Rolling Variance (The "Silence" Detector)
data['PFlow_Var'] = data['PFlow_Clean'].rolling(window=3*fs_original, center=True).var()
data['PFlow_Var'] = data['PFlow_Var'].bfill().ffill()

# Compile features
feature_columns = [
    'PFlow', 'Thorax', 'Abdomen',                       # 0, 1, 2
    'PFlow_Clean', 'Thorax_Clean', 'Abdomen_Clean',     # 3, 4, 5
    'PFlow_Upper', 'PFlow_Lower', 'PFlow_Width',        # 6, 7, 8
    'Thorax_Upper', 'Thorax_Lower', 'Thorax_Width',     # 9, 10, 11
    'Abdomen_Upper', 'Abdomen_Lower', 'Abdomen_Width',  # 12, 13, 14
    'SaO2_Smooth', 'Thorax_Abdomen_Corr', 'Effort_Flow_Ratio', # 15, 16, 17
    'SaO2_Deriv', 'Phase_Angle', 'PFlow_Var',            # 18, 19, 20
    'Vitalog1_Clean','Vitalog2_Clean'                   # 21, 22 
]

features_256hz = data[feature_columns].values
time_256hz = data['time_sec'].values

# =============================================================
# 3. DOWNSAMPLING (256 Hz -> 32 Hz)
# =============================================================
print("3. Downsampling from 256Hz to 32Hz...")
fs_target = 32
downsample_ratio = fs_target / fs_original
target_length = int(len(features_256hz) * downsample_ratio)

features_32hz = resample(features_256hz, target_length)
time_32hz = np.linspace(window_start, window_end, target_length)

# =============================================================
# 4. SEGMENTATION (30s Window, 10s Overlap)
# =============================================================
print("4. Segmenting into 30s windows with 10s overlap...")
window_seconds = 30
overlap_seconds = 10
step_seconds = window_seconds - overlap_seconds

win_samples = window_seconds * fs_target  
step_samples = step_seconds * fs_target   

segments = []
segment_times = [] # NEW: Array to hold the real time axis for each segment

for i in range(0, len(features_32hz) - win_samples + 1, step_samples):
    # Slice the data
    segment = features_32hz[i : i + win_samples, :]
    segments.append(segment)
    
    # Slice the time array exactly the same way
    seg_time = time_32hz[i : i + win_samples]
    segment_times.append(seg_time)

segments = np.array(segments)
segment_times = np.array(segment_times) # Convert to numpy array for easy indexing
print(f"   Created {segments.shape[0]} segments of shape {segments.shape[1:]}")

# =============================================================
# 5. SEGMENT-WISE NORMALIZATION (Z-Score)
# =============================================================
print("5. Applying segment-wise Z-score normalization...")
normalized_segments = np.zeros_like(segments)

for i in range(segments.shape[0]):
    scaler = StandardScaler()
    normalized_segments[i] = scaler.fit_transform(segments[i])

# =============================================================
# 6. VISUALIZATION OF THE PIPELINE
# =============================================================
# ---------------- CHEAT SHEET FOR CHANNELS ----------------
# [0, 1, 2]   = RAW: PFlow, Thorax, Abdomen
# [3, 4, 5]   = CLEAN: PFlow, Thorax, Abdomen
# [6, 7, 8]   = PFLOW ENVS: Upper, Lower, Width
# [9, 10, 11] = THORAX ENVS: Upper, Lower, Width
# [12, 13, 14]= ABDOMEN ENVS: Upper, Lower, Width
# [15, 16, 17]= ADVANCED: SaO2, Corr, Ratio
# ----------------------------------------------------------

# --- USER CONTROLS (Set your channels here!) ---
manual_channel_1 = 0      # Plot 1: e.g., 0 for Raw PFlow
manual_channel_2 = 3      # Plot 2: e.g., 3 for Clean PFlow

channel_to_plot = 8       # Plots 3-6: The feature to track (e.g., 8 for PFlow_Width)
segment_index_to_view = 2 # Which 30s window to zoom in on
# -----------------------------------------------

# Create 6 subplots instead of 4, make the figure taller
fig, axes = plt.subplots(6, 1, figsize=(12, 18))
fig.tight_layout(pad=5.0)

# --- PLOT 1: Manual Custom View 1 ---
name_1 = feature_columns[manual_channel_1]
axes[0].plot(time_256hz, features_256hz[:, manual_channel_1], color='gray', label=name_1)
axes[0].set_title(f"1. Custom View (256 Hz) - {name_1}")
axes[0].legend(loc="upper right")

# --- PLOT 2: Manual Custom View 2 ---
name_2 = feature_columns[manual_channel_2]
axes[1].plot(time_256hz, features_256hz[:, manual_channel_2], color='purple', label=name_2)
axes[1].set_title(f"2. Custom View (256 Hz) - {name_2}")
axes[1].legend(loc="upper right")

# --- PLOTS 3 to 6: The Processing Pipeline ---
feature_name = feature_columns[channel_to_plot]

# Plot 3: 256 Hz Feature
axes[2].plot(time_256hz, features_256hz[:, channel_to_plot], color='blue', label=feature_name)
axes[2].set_title(f"3. Feature Extracted (256 Hz) - {feature_name}")
axes[2].legend(loc="upper right")

# Plot 4: 32 Hz Downsampled
axes[3].plot(time_32hz, features_32hz[:, channel_to_plot], color='cyan', label=f'{feature_name} (32 Hz)')
axes[3].set_title(f"4. Downsampled (32 Hz) - {feature_name}")
axes[3].legend(loc="upper right")

# Plot 5: Sliced Segment (Raw Scale)
time_segment = segment_times[segment_index_to_view]
axes[4].plot(time_segment, segments[segment_index_to_view, :, channel_to_plot], color='orange', label=f'{feature_name} (Raw)')
axes[4].set_title(f"5. Segment {segment_index_to_view} (Un-normalized) - Real Time Axis")
axes[4].legend(loc="upper right")

# Plot 6: Sliced Segment (Z-Score)
axes[5].plot(time_segment, normalized_segments[segment_index_to_view, :, channel_to_plot], color='green', label=f'{feature_name} (Z-Score)')
axes[5].set_title(f"6. Segment {segment_index_to_view} (Normalized: Mean=0, Std=1) - Real Time Axis")
axes[5].legend(loc="upper right")

plt.show()

# =============================================================
# 7. EXTRACT CORE CHANNELS & SAVE X_TRAIN
# =============================================================
# NOTE: We are skipping Pseudo-Label generation because you have 
# real clinical doctor scores for the Y tensor!
# =============================================================
print("7. Extracting 6 core channels for AI input...")

# Indices for: [PFlow_Clean(3), Abdomen_Clean(5), Ratio(17), SaO2_Deriv(18), PFlow_Var(20), Vitalog2(22)]
core_indices = [3, 5, 17, 18, 20, 22]

# We extract ONLY these from the NORMALIZED segments
X_train = normalized_segments[:, :, core_indices]

# Save BOTH the features and the time alignments to disk!
np.save('X_train_PentaLSTM.npy', X_train)
np.save('segment_times.npy', segment_times) 

print(f"\n✅ SUCCESS! Preprocessing complete.")
print(f"   X_train saved: {X_train.shape} (Segments, Time-Steps, Channels)")
print(f"   segment_times saved: {segment_times.shape}")
print("   Ready for clinical label synchronization script.")