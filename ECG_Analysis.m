% ELEC 408 Assignment: ECG Analysis
% Ekin Yelken 20166126

% Clear Workspace and Load Data
clear; clc; close all;

%% Raw ECG Signals and Peaks

normal = load("A00001.mat");
atrialFibrillation = load("A00004.mat");
noisyData = load("A00585.mat");

% Extract Signals
normal_signal = normal.val;
atrialFibrillation_signal = atrialFibrillation.val;
noisy_signal = noisyData.val;

% Set Sampling Frequency Assumption
sampling_frequency = 300; % Hz

% Convert To Time Axis
normalTime = (0:length(normal_signal)-1)/sampling_frequency;
atrialTime = (0:length(atrialFibrillation_signal)-1)/sampling_frequency;
noisyTime  = (0:length(noisy_signal)-1)/sampling_frequency;

% Aproximate Peaks

% Select minimum voltage for P Wave peak in PQRST complex 
voltageCutoff = 400;

% findpeaks returns value at peak and index corresponding to that value
[normalPeak_val, normalPeak_idx] = findpeaks(normal_signal, 'MinPeakDistance', 150);
[atrialPeak_val, atrialPeak_idx] = findpeaks(atrialFibrillation_signal, 'MinPeakDistance', 150);

normalPeak_idx = normalPeak_idx(normalPeak_val >= voltageCutoff);
normalPeak_val = normalPeak_val(normalPeak_val >= voltageCutoff);

atrialPeak_idx = atrialPeak_idx(atrialPeak_val >= voltageCutoff);
atrialPeak_val = atrialPeak_val(atrialPeak_val >= voltageCutoff);

% Plot ECGs
figure('Name', 'Raw ECG Signal - Time Domain', 'NumberTitle', 'off');


subplot(3,1,1);
plot(normalTime,normal_signal);
hold on;
plot(normalTime(normalPeak_idx), normalPeak_val, 'rx');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Normal HR');

subplot(3,1,2);
plot(atrialTime, atrialFibrillation_signal);
hold on;
plot(atrialTime(atrialPeak_idx), atrialPeak_val, 'rx');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Atrial Fibrillation HR');

subplot(3,1,3);
plot(noisyTime, noisy_signal);
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Noisy HR');



%% Frequency Analysis


% Isolate PQRST Segments
segment_length = mean(diff(normalPeak_idx));

% Preallocate data structure for storing PQRST segments
num_segments   = length(normalPeak_idx)-1;
pqrst_segments = zeros(num_segments, round(segment_length));

% Separate segments
for i = 1:num_segments
    start_idx = normalPeak_idx(i) + round(segment_length/2);
    end_idx   = start_idx + round(segment_length)-1;
    
    if(end_idx > length(normal_signal))
        break;
    end
    % Extract Segments
    pqrst_segments(i, :) = normal_signal(start_idx:end_idx);
end

% Apply Windowing Function
window = hamming(round(segment_length))'; % Arbitrarily chose Hamming Window
windowed_segments = pqrst_segments .* window;

% PQRST Segments in Time Domain with window
firstPQRST_segment = pqrst_segments(1, :) .* window;
secondPQRST_segment = pqrst_segments(2, :) .* window;

pqrstTime_1 = (0:length(firstPQRST_segment)-1) / sampling_frequency;
pqrstTime_2 = (0:length(secondPQRST_segment)-1) / sampling_frequency;

% Segment Length
N1 = length(windowed_segments(1, :));
N2 = length(windowed_segments(2, :));

% Compute One-Sided FFT
fft_segment1 = abs(fft(firstPQRST_segment));
fft_segment2 = abs(fft(secondPQRST_segment));

fft_segment1 = fft_segment1(1:floor(N1/2)+1);
fft_segment2 = fft_segment2(1:floor(N2/2)+1);


% Define the frequency axis for the segments
freq_axis_1 = (0:floor(N1/2)) * (sampling_frequency / N1);
freq_axis_2 = (0:floor(N2/2)) * (sampling_frequency / N2);

% Convert to Frequency Axis
frequencies = (0:length(fft_segment1)-1 * (sampling_frequency) / length(fft_segment1));

% Plot FFT of the segments
figure('Name', 'Fourier Transformed ECG Signal', 'NumberTitle', 'off');
grid on;

subplot(2,2,1);
plot(pqrstTime_1, firstPQRST_segment);
title('First PQRST Segment');
xlabel('Time (s)');
ylabel('Voltage (mV)');


subplot(2,2,2);
plot(freq_axis_1, fft_segment1);
title('One-Sided FFT of First PQRST');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(2,2,3);
plot(pqrstTime_2, secondPQRST_segment);
title('Second PQRST Segment');
xlabel('Time (s)');
ylabel('Voltage (mV)');

subplot(2,2,4);
plot(freq_axis_2, fft_segment2);
title('One-Sided FFT of Second PQRST');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

%% Bandpass Butterworth Filter
n_order = 3;
cuttoff_lowFreq = 5;  % Hz
cuttoff_highFreq = 15; % Hz

[b, a] = butter(n_order, [cuttoff_lowFreq cuttoff_highFreq] / (sampling_frequency / 2), 'bandpass');

filtered_normal = filtfilt(b, a, normal_signal);
filtered_atrialFibrillation = filtfilt(b, a, atrialFibrillation_signal);
filtered_noisy  = filtfilt(b, a, noisy_signal);


% Plot Regular and Filtered ECG Signals
figure('Name', 'Filtered ECG Signal - Time Domain', 'NumberTitle', 'off');
subplot(3,3,1);
plot(normalTime, normal_signal, 'b');
%hold on;
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Raw Normal Signal');

subplot(3,3,2);
plot(normalTime,filtered_normal, 'r');
xlabel('Time (s)');
title('Filtered Normal Signal');

subplot(3,3,3);
plot(normalTime, normal_signal, 'b');
hold on;
plot(normalTime,filtered_normal, 'r');
xlabel('Time (s)');
title('Raw vs Filtered: Normal Signal');


subplot(3,3,4);
plot(atrialTime, atrialFibrillation_signal, 'b');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Raw Atrial Fibrillation Signal');

subplot(3,3,5);
plot(atrialTime, filtered_atrialFibrillation, 'r');
xlabel('Time (s)');
title('Filtered Atrial Fibrillation Signal');

subplot(3,3,6);
plot(atrialTime, atrialFibrillation_signal, 'b');
hold on;
plot(atrialTime, filtered_atrialFibrillation, 'r');
xlabel('Time (s)');
title('Raw vs Filtered: Atrial Fibrillation Signal');


subplot(3,3,7);
plot(noisyTime, noisy_signal, 'b');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Raw Noisy Signal');

subplot(3,3,8);
plot(noisyTime, filtered_noisy, 'r');
xlabel('Time (s)');
title('Raw Filtered Noisy Signal');

subplot(3,3,9);
plot(noisyTime, noisy_signal, 'b');
hold on;
plot(noisyTime, filtered_noisy, 'r');
xlabel('Time (s)');
title('Raw vs Filtered: Noisy Signal');

%% Detect R-Peaks
% Derivative Filter
kernel = [1 2 0 -2 -1];
derivative_normal = filter(kernel, 1, filtered_normal);
derivative_atrialFibrillation  = filter(kernel, 1, filtered_atrialFibrillation);
derivative_noisy  = filter(kernel, 1, filtered_noisy);

% Squaring the Signal
squared_normal = derivative_normal .^ 2;
squared_atrialFibrillation  = derivative_atrialFibrillation .^ 2;
squared_noisy  = derivative_noisy .^ 2;

% Moving Average Filter
N_samples = 30;
movingAVG_normal = movmean(squared_normal, N_samples);
movingAVG_atrialFibrillation  = movmean(squared_atrialFibrillation, N_samples);
movingAVG_noisy  = movmean(squared_noisy, N_samples);

% Detect R-peaks after processing
R_peaks_height = 100000; % Select minimum height to remove P and T wave peaks
[~, R_peaks_normal]              = findpeaks(movingAVG_normal, 'MinPeakDistance', 0.25 * sampling_frequency, 'MinPeakHeight', R_peaks_height);
[~, R_peaks_atrialFibrillation]  = findpeaks(movingAVG_atrialFibrillation, 'MinPeakDistance', 0.25 * sampling_frequency, 'MinPeakHeight', R_peaks_height);
[~, R_peaks_noisy]               = findpeaks(movingAVG_noisy, 'MinPeakDistance', 0.25 * sampling_frequency, 'MinPeakHeight', R_peaks_height);

% Plot Detected Peaks
figure('Name', 'R-Peaks ECG Signal (findpeaks) - Time Domain', 'NumberTitle', 'off');
subplot(3,1,1);
plot(normalTime, movingAVG_normal);
hold on;
plot(R_peaks_normal/sampling_frequency, movingAVG_normal(R_peaks_normal), 'ro');
title('R-peaks Detected: Normal Signal');

subplot(3,1,2);
plot(atrialTime, movingAVG_atrialFibrillation);
hold on;
plot(R_peaks_atrialFibrillation/sampling_frequency, movingAVG_atrialFibrillation(R_peaks_atrialFibrillation), 'ro');
title('R-peaks Detected: Atrial Fibrillation Signal');

subplot(3,1,3);
plot(noisyTime, movingAVG_noisy);
hold on;
plot(R_peaks_noisy/sampling_frequency, movingAVG_noisy(R_peaks_noisy), 'ro');
title('R-peaks Detected: Noisy Signal');

%% Adaptive Thresholding

% Only completing for normal heart rate

% Parameters for Adaptive Thresholding
SPKI = max(movingAVG_normal(1:2*sampling_frequency));  % Initial signal peak estimate
NPKI = mean(movingAVG_normal(1:2*sampling_frequency)); % Initial noise peak estimate
Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
Threshold_I2 = 0.5 * Threshold_I1;

% Estimate the number of expected R-peaks based on previous detection
estimated_peaks = length(R_peaks_normal);  % Use previous peak detection to preallocate

% Preallocate R-peaks array
R_peaks_normal_Adaptive = zeros(1, estimated_peaks);
peak_count = 0; % Counter to track number of detected peaks

% Search window based on RR interval
searchback_window = round(1.66 * sampling_frequency);

for i = 2:length(movingAVG_normal)
    if movingAVG_normal(i) > Threshold_I1  % Primary threshold check
        peak_count = peak_count + 1;
        R_peaks_normal_Adaptive(peak_count) = i;  % Store R-peak index
        SPKI = 0.125 * movingAVG_normal(i) + 0.875 * SPKI;  % Update signal peak
        Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
        Threshold_I2 = 0.5 * Threshold_I1;
    elseif movingAVG_normal(i) > Threshold_I2  % Secondary threshold (searchback)
        if peak_count > 0 && (i - R_peaks_normal_Adaptive(peak_count)) > searchback_window
            peak_count = peak_count + 1;
            R_peaks_normal_Adaptive(peak_count) = i;
            SPKI = 0.25 * movingAVG_normal(i) + 0.75 * SPKI;
            Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
            Threshold_I2 = 0.5 * Threshold_I1;
        end
    else
        NPKI = 0.125 * movingAVG_normal(i) + 0.875 * NPKI;  % Update noise peak
        Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
        Threshold_I2 = 0.5 * Threshold_I1;
    end
end

% Remove any uninitialized zeros in the preallocated array
R_peaks_normal_Adaptive = R_peaks_normal_Adaptive(1:peak_count);

% Plot Results
% Find common R-peaks between adaptive thresholding and original detection
[valid_peaks, idx_adaptive, idx_original] = intersect(R_peaks_normal_Adaptive, R_peaks_normal);

% Use only valid R-peaks for plotting
R_peaks_normal_Adaptive_Valid = R_peaks_normal_Adaptive(idx_adaptive);

% Plot Results using movingAVG_normal instead of normal_signal
figure('Name', 'R-Peaks ECG Signal (Adaptive Thresholding) - Time Domain', 'NumberTitle', 'off');
subplot(2,1,1);
plot(normalTime, movingAVG_normal, 'b');  % Use the processed signal for visualization
hold on;
plot(normalTime(R_peaks_normal_Adaptive_Valid), movingAVG_normal(R_peaks_normal_Adaptive_Valid), 'ro');
title('Processed Signal with Valid Detected R-Peaks');
xlabel('Time (s)');
ylabel('Magnitude');
grid on;

% Create pulse train for detected R-peaks
pulse_train = zeros(size(movingAVG_normal));
pulse_train(R_peaks_normal_Adaptive_Valid) = max(movingAVG_normal);

% Remove zero values to only keep detected peaks for better visualization
non_zero_indices = find(pulse_train > 0);
pulse_train_nonzero = pulse_train(non_zero_indices);
pulse_time_nonzero = normalTime(non_zero_indices);

% Define maximum height for the pulse train
max_height = 175000;  % Set desired limit for stem height

% Limit the pulse train values to the specified height
pulse_train_nonzero_limited = min(pulse_train_nonzero, max_height);

% Plot the pulse train with limited step height
subplot(2,1,2);
plot(normalTime, movingAVG_normal, 'b');
hold on;
stem_handle = stem(pulse_time_nonzero, pulse_train_nonzero_limited, 'r', 'Marker', 'none', 'LineWidth', 2);

title('Pulse Train of Valid Detected QRS Complexes');
xlabel('Time (s)');
ylabel('Magnitude');
grid on;
%% Analyze Signals

% Heart Rate Calculation
RR_intervals_normal = diff(R_peaks_normal) / sampling_frequency;
RR_intervals_atrialFibrillation  = diff(R_peaks_atrialFibrillation) / sampling_frequency;
RR_intervals_noisy  = diff(R_peaks_noisy) / sampling_frequency;

HR_normal = 60 ./ RR_intervals_normal;
HR_atrialFibrillation  = 60 ./ RR_intervals_atrialFibrillation;
HR_noisy  = 60 ./ RR_intervals_noisy;

fprintf('Normal HR: %.2f bpm\n', mean(HR_normal));
fprintf('Atrial Fibrillation HR: %.2f bpm\n', mean(HR_atrialFibrillation));
fprintf('Noisy HR: %.2f bpm\n', mean(HR_noisy));

% Heart Rate Variability (HRV)
HRV_normal = sqrt(mean(diff(RR_intervals_normal).^2));
HRV_atrialFibrillation  = sqrt(mean(diff(RR_intervals_atrialFibrillation).^2));
HRV_noisy  = sqrt(mean(diff(RR_intervals_noisy).^2));

fprintf('\nNormal HRV: %.2f\n', HRV_normal);
fprintf('Atrial Fibrillation HRV: %.2f\n', HRV_atrialFibrillation);
fprintf('Noisy HRV: %.2f\n', HRV_noisy);