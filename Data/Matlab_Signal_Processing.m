close all;
clear; % Good practice to clear workspace when changing windows

% =============================================================
% 1. ADJUST YOUR WINDOW HERE
% =============================================================
window_start = 3500; 
window_end   = 3600;

% =============================================================
% 2. LOAD & FILTER DATA
% =============================================================
data = readtable('ON020217-06.csv'); 
data.Properties.VariableNames = {'PFlow', 'Thorax', 'Abdomen', 'SaO2', 'Vitalog1', 'Vitalog2', 'time_sec'};

% Filter using your new variables
golden_window_idx = (data.time_sec >= window_start) & (data.time_sec <= window_end);
golden_data = data(golden_window_idx, :);

% Extract only the needed variables (Ignoring Vitalog1 & 2)
PFlow = golden_data.PFlow;
Thorax = golden_data.Thorax;
Abdomen = golden_data.Abdomen;
SaO2 = golden_data.SaO2;
time_sec = golden_data.time_sec;
Vitalog1=golden_data.Vitalog1;
Vitalog2=golden_data.Vitalog2;
% =============================================================
% 3. FEATURE EXTRACTION (Filtering & Envelopes)
% =============================================================
fs = 256; % Sampling rate

% --- 3.1 Detrending (Baseline Stabilization) ---
% Removes linear trends/drifts before applying filters
PFlow_Detrend    = detrend(PFlow);
Thorax_Detrend   = detrend(Thorax);
Abdomen_Detrend  = detrend(Abdomen);

% --- 3.2 SaO2 Smoothing ---
SaO2_Continuous = smoothdata(SaO2, 'movmean', 512);

% --- 3.3 Butterworth Band-Pass Filter (0.1 - 0.3 Hz) ---
order = 2;
freq_band = [0.15 0.7];
[b, a] = butter(order, freq_band/(fs/2), 'bandpass');
freq_band2 = [0.1 0.3];
[c, d] = butter(order, freq_band2/(fs/2), 'bandpass');

% Apply filtfilt (zero-phase filtering) to avoid time shifts
PFlow_Clean   = filtfilt(b, a, PFlow_Detrend);
Thorax_Clean  = filtfilt(c, d, Thorax_Detrend);
Abdomen_Clean = filtfilt(c, d, Abdomen_Detrend);

% --- 3.4 Envelope Extraction ---
[upper_env, ~]    = envelope(PFlow_Clean, 256, 'peak');
[Thorax_Env, ~]   = envelope(Thorax_Clean, 256, 'peak');
[Abdomen_Env, ~]  = envelope(Abdomen_Clean, 256, 'peak');

% =============================================================
% 4. MASTER VISUALIZATION (Dynamic Window)
% =============================================================
figure('Name', 'Dynamic Feature Extraction (Band-Pass & Detrended)', 'Position', [100, 100, 1200, 800]);

% PFlow
ax1 = subplot(4,1,1);
plot(time_sec, PFlow_Detrend, 'Color', [0.8 0.8 0.8]); hold on;
plot(time_sec, PFlow_Clean, 'Color', [0,0.7,0.9]); hold on;
plot(time_sec, upper_env, 'b', 'LineWidth', 2);
title(['PFlow (Window: ', num2str(window_start), 's to ', num2str(window_end), 's)']);
ylabel('PFlow'); xlim([window_start window_end]);

% Thorax
ax2 = subplot(4,1,2);
plot(time_sec, Thorax_Detrend, 'Color', [0.8 0.8 0.8]); hold on;
plot(time_sec, Thorax_Clean, 'Color', [0,0.7,0.9]); hold on;
plot(time_sec, Thorax_Env, 'r', 'LineWidth', 2);
ylabel('Thorax'); xlim([window_start window_end]);

% Abdomen
ax3 = subplot(4,1,3);
plot(time_sec, Abdomen_Detrend, 'Color', [0.8 0.8 0.8]); hold on;
plot(time_sec, Abdomen_Clean, 'Color', [0,0.7,0.9]); hold on;
plot(time_sec, Abdomen_Env, 'g', 'LineWidth', 2);
ylabel('Abdomen'); xlim([window_start window_end]);

% SaO2
ax4 = subplot(4,1,4);
plot(time_sec, SaO2, 'Color', [0.8 0.8 0.8]); hold on;
plot(time_sec, SaO2_Continuous, 'k', 'LineWidth', 2);
ylabel('SaO2 (%)'); xlabel('Time (Seconds)');
xlim([window_start window_end]);

% Link axes for zooming
linkaxes([ax1, ax2, ax3, ax4], 'x');

% =========================================================================
% FULL DATA OVERVIEW (1000s Window)
% =========================================================================
% Use this to scan for your next sleep apnea event

% 1. Filter the data for the full range provided by Mohammad
full_range_idx = (data.time_sec >= 3000) & (data.time_sec <= 4000);

% Extract only the 4 signals we care about right now + time
full_data = data(full_range_idx, {'PFlow', 'Thorax', 'Abdomen', 'SaO2', 'time_sec'});

% 2. Create the overview figure
figure('Name', 'Full Overview: 3,000s - 4,000s', 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

% 3. Plot all channels in a stacked layout for easy comparison
s_overview = stackedplot(full_data, 'XVariable', 'time_sec');
grid on;
title('Full Dataset Scan: Identify Next Golden Window');