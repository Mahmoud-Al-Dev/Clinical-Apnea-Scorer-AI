import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, resample, detrend
from scipy.ndimage import uniform_filter1d, median_filter
from sklearn.preprocessing import StandardScaler

# =============================================================
# 1. LOAD & CLEAN DATA (Using Pandas)
# =============================================================
print("1. Loading and cleaning full night data...")

NO_OF_CHANNELS=7
NIGHT_TEST=7
if NO_OF_CHANNELS==6:
    df = pd.read_csv('Data\ON030217-06.csv', 
                 names=['PFlow', 'Thorax', 'Abdomen', 'SaO2', 'Vitalog1', 'Vitalog2', 'time_sec'])
else:
    df = pd.read_csv('Data\TR041016-05.csv', 
                 names=['PFlow', 'Thorax', 'Abdomen', 'SaO2', 'Vitalog1', 'Vitalog2','ECG', 'time_sec'])

fs_original = 256

# --- A. Dynamic 20-Minute Trimming ---
real_start = df['time_sec'].iloc[0]
real_end = df['time_sec'].iloc[-1]

window_start = real_start + 1200
window_end = real_end - 1200

# Slice the middle portion of the night
data = df[(df['time_sec'] >= window_start) & (df['time_sec'] <= window_end)].copy()

# --- B. Remove Sensor Dropouts ---
valid_sao2_mask = data['SaO2'] > 10 
data = data[valid_sao2_mask].copy()

# --- C. Stitch the Data Together ---
data.reset_index(drop=True, inplace=True)
new_time_axis = np.arange(len(data)) / fs_original + window_start
data['time_sec'] = new_time_axis

print(f"   Removed 40 mins of awake time & dropped {sum(~valid_sao2_mask)} rows of dead sensor data.")

# =============================================================
# 2. FEATURE EXTRACTION & ENGINEERING
# =============================================================
print("2. Extracting and Engineering features at 256 Hz...")

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

# Correctly map the time axis based on the new, shortened data length
time_32hz = np.linspace(new_time_axis[0], new_time_axis[-1], target_length)

# =============================================================
# 4. SEGMENTATION (30s Window, 10s Overlap)
# =============================================================
print("4. Segmenting into 30s windows with 10s overlap...")
win_samples = 30 * fs_target  
step_samples = 20 * fs_target   

segments = []
segment_times = []

for i in range(0, len(features_32hz) - win_samples + 1, step_samples):
    segments.append(features_32hz[i : i + win_samples, :])
    segment_times.append(time_32hz[i : i + win_samples])

segments = np.array(segments)
segment_times = np.array(segment_times)
print(f"   Created {segments.shape[0]} segments.")

# =============================================================
# 5. SEGMENT-WISE NORMALIZATION (Z-Score)
# =============================================================
print("5. Applying segment-wise Z-score normalization...")
normalized_segments = np.zeros_like(segments)

VAR_IDX = 20 

for i in range(segments.shape[0]):
    seg = segments[i]
    scaler = StandardScaler()
    norm_seg = scaler.fit_transform(seg)
    
    stds = np.std(seg, axis=0)
    
    for col in range(seg.shape[1]):
        if col == VAR_IDX:
            # If it's literally all zeros (dead signal), just keep it zeros
            if stds[col] == 0:
                norm_seg[:, col] = 0.0
            continue 
            
        # Standard safety catch for all other signals (including SaO2_Deriv)
        if stds[col] < 1e-4:
            norm_seg[:, col] = 0.0
            
    normalized_segments[i] = norm_seg
# =============================================================
# 6. EXTRACT CORE CHANNELS & SAVE X_TRAIN
# =============================================================
print("6. Extracting 5 core channels for AI input...")

# The Ultimate 6:
# [3] PFlow_Clean
# [11] Thorax_Width
# [14] Abdomen_Width
# [15] SaO2_Smooth
# [17] Effort_Flow_Ratio  <-- The OSA Savior
# [18] SaO2_Deriv

core_indices = [3, 4, 5, 11, 14, 15, 17, 18] 

X_train = normalized_segments[:, :, core_indices]
np.save(f'X_{NIGHT_TEST}', X_train)

# We save the segment_times so we can map the doctor's labels perfectly later!
np.save(f'segment_times_n{NIGHT_TEST}.npy', segment_times)

print(f"\n✅ SUCCESS! Preprocessing complete.")
print(f"   X_train saved: {X_train.shape} (Segments, Time-Steps, Channels)")