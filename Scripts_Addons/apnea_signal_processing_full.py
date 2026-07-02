import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, hilbert, resample, detrend
from scipy.ndimage import uniform_filter1d, median_filter
from sklearn.preprocessing import StandardScaler

# =============================================================
# --- USER CONTROLS (BATCH PROCESSING) ---
# =============================================================

# Name of the folder where all output arrays will be saved
OUTPUT_FOLDER = 'Nights_Vitalog'

# List of nights to process: (Night_ID, 'File_Path', Number_of_Channels)
NIGHTS_TO_PROCESS = [
    (1, 'Data/1HK190117-06.csv', 8),
    (2, 'Data/2MU090517-06.csv', 8),
    (3, 'Data/3ON020217-06.csv', 7),
    (4, 'Data/4ON030217-06.csv', 7),
    (5, 'Data/5ST300317-06.csv', 8),
    (6, 'Data/6ST310317-06.csv', 8),
    (7, 'Data/7TR041016-05.csv', 8),
    (8, 'Data/8MI220916-06.csv', 8),
    (9, 'Data/9MG080916-07.csv', 8),
    (10, 'Data/ASK051016-08.csv', 6),
    (11, 'Data/BDD010217-06.csv', 8),
    (12, 'Data/CGK180916-05.csv', 8),
    (13, 'Data/DKB041216-06.csv', 8),
    (14, 'Data/EHK200117-06-a.csv', 8),
    (15, 'Data/FMB110517-05.csv', 8),
    (16, 'Data/GMI220916-06.csv', 8),
    (17, 'Data/HNK220916-05.csv', 8),
    (18, 'Data/ISK041016-06.csv', 8),
    (19, 'Data/JSS260117-06.csv', 8),
    (20, 'Data/KAB090417-05.csv', 8),
    (21, 'Data/LAB100417-05.csv', 8),
    (22, 'Data/MAD221116-05.csv', 8),
    (23, 'Data/NAL150117-10.csv', 8),
    (24, 'Data/OAR210317-05.csv', 8),
    (25, 'Data/PAR220317-05.csv', 8),
    (26, 'Data/QBG070517-06.csv', 8),
    (27, 'Data/RBH061016-06.csv', 8),
    (28, 'Data/SBH071016-06.csv', 8),
    (29, 'Data/TCD171116-05.csv', 8),
    (30, 'Data/UCR201116-06(2).csv', 8),
    (31, 'Data/VCR211116-06.csv', 8),
    (32, 'Data/WCS210317-06.csv', 8),
    (33, 'Data/XCS220317-06.csv', 8),
    (34, 'Data/YDD010217-06.csv', 8),
    (35, 'Data/ZED270417-05.csv', 8)
]

fs_original = 256
fs_target = 32
# =============================================================

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"🚀 Starting Batch Processing for {len(NIGHTS_TO_PROCESS)} nights...")

for night_id, file_path, num_channels in NIGHTS_TO_PROCESS:
    print(f"\n{'='*60}")
    print(f"▶ Processing Night {night_id} | Channels: {num_channels} | File: {file_path}")
    print(f"{'='*60}")

    # --- 0. DYNAMIC CHANNEL ROUTING ---
    if num_channels == 8:
        col_names = ['PFlow', 'Thorax', 'Abdomen', 'SaO2', 'Vitalog1', 'Vitalog2', 'ECG', 'time_sec']
    elif num_channels == 7:
        col_names = ['PFlow', 'Thorax', 'Abdomen', 'SaO2', 'Vitalog1', 'Vitalog2', 'time_sec']
    else:
        print(f"⚠️ Skipping Night {night_id}: Unsupported channel count ({num_channels}). Only 7 or 8 are processed.")
        continue

    # =============================================================
    # 1. LOAD & CLEAN DATA
    # =============================================================
    print("1. Loading and cleaning full night data...")
    try:
        df = pd.read_csv(file_path, names=col_names)
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found. Skipping...")
        continue

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
    def apply_lowpass(signal, cutoff, fs, order=2):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff/nyq, btype='low')
        return filtfilt(b, a, signal)

    data['Vitalog1_Clean'] = apply_lowpass(data['Vitalog1_Med'], 2.0, fs_original)
    data['Vitalog2_Clean'] = apply_lowpass(data['Vitalog2_Med'], 2.0, fs_original)

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

    # 2. Thorax-Abdomen Phase Angle
    phase_thorax = np.unwrap(np.angle(hilbert(data['Thorax_Clean'])))
    phase_abdomen = np.unwrap(np.angle(hilbert(data['Abdomen_Clean'])))
    data['Phase_Angle'] = np.abs(phase_thorax - phase_abdomen)

    # 3. Airflow Rolling Variance
    data['PFlow_Var'] = data['PFlow_Clean'].rolling(window=3*fs_original, center=True).var()
    data['PFlow_Var'] = data['PFlow_Var'].bfill().ffill()

    # Compile features (We only care about the core indices later, but we map them consistently)
    feature_columns = [
        'PFlow', 'Thorax', 'Abdomen',                       # 0, 1, 2
        'PFlow_Clean', 'Thorax_Clean', 'Abdomen_Clean',     # 3, 4, 5
        'PFlow_Upper', 'PFlow_Lower', 'PFlow_Width',        # 6, 7, 8
        'Thorax_Upper', 'Thorax_Lower', 'Thorax_Width',     # 9, 10, 11
        'Abdomen_Upper', 'Abdomen_Lower', 'Abdomen_Width',  # 12, 13, 14
        'SaO2_Smooth', 'Thorax_Abdomen_Corr', 'Effort_Flow_Ratio', # 15, 16, 17
        'SaO2_Deriv', 'Phase_Angle', 'PFlow_Var',            # 18, 19, 20
        'Vitalog1_Clean','Vitalog2_Clean'                 #21,22
    ]

    features_256hz = data[feature_columns].values
    time_256hz = data['time_sec'].values

    # =============================================================
    # 3. DOWNSAMPLING (256 Hz -> 32 Hz)
    # =============================================================
    print("3. Downsampling from 256Hz to 32Hz...")
    downsample_ratio = fs_target / fs_original
    target_length = int(len(features_256hz) * downsample_ratio)

    features_32hz = resample(features_256hz, target_length)
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
                if stds[col] == 0:
                    norm_seg[:, col] = 0.0
                continue 
            if stds[col] < 1e-4:
                norm_seg[:, col] = 0.0
                
        normalized_segments[i] = norm_seg

    # =============================================================
    # 6. EXTRACT CORE CHANNELS & SAVE X_TRAIN
    # =============================================================
    print("6. Extracting core channels and saving...")

    core_indices = [3, 4, 5, 11, 14, 15, 17, 18] 
    X_train = normalized_segments[:, :, core_indices]

    x_save_path = os.path.join(OUTPUT_FOLDER, f'X_{night_id}.npy')
    time_save_path = os.path.join(OUTPUT_FOLDER, f'segment_times_n{night_id}.npy')

    np.save(x_save_path, X_train)
    np.save(time_save_path, segment_times)

    print(f"✅ Night {night_id} complete!")
    print(f"   Saved to: {x_save_path}")

print("\n🎉 All nights have been successfully processed!")