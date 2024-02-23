#!/usr/bin/env python
# coding: utf-8

# # ELEC 408 Assignment 1: ECG and Heart Rate
# ## Ekin Yelken 20166126 ekin.yelken@queensu.ca
# ### Date of Submission: Friday, February 16, 2024
# #### Description: 
# The objective of this code is to analyze the data associated three different ECG Signal waveforms: Normal, Atrial Fibrillation, and Noisy. 
# 
# #### Part A: ECG Signal Analysis
# The waveform for each ECG Signal is split into PQRST Segments, used to create a Fast Fourier Transform (FFT) and compared to the fourier transform of the entire wave form. The Three signals' are also compared in the frequency domain.
# 
# #### Part B: Pan-Tompkins Algorithm
# 1. Bandpass filter (5-15 Hz).
# 2. Derivative filter to highlight the QRS complex.
# 3. Squaring of the signal.
# 4. Signal averaging (MWI) to eliminate high-frequency noise.
# 5. Adaptive threshold for decision.
# 
# #### Part C: Plot R-Peaks and Associated Pulse Train for each ECG Signal (using Adaptive Thresholding)
# 
# #### Part D: Critical Physiological indicators are calculated including 
# 
# Heart Rate (HR): 
# $$
# HR (bpm) = \frac{60}{RR(s)}
# $$
# 
# Where, RR is the distance between two R-peaks.
# 
# Heart Rate Variability (HRV) using the root mean square of successive differences (RMSSD).
# 
# 
# 
# 
# #### Disclaimer:
# This code has been created and authored for the purpose of the ELEC 408 course at Queen's University. Replication or Reproduction including screenshots for any purpose requires the explicit persmission of the author listed above. Please contact with inquiries if the reader wishes to use this code.

# ### Import Libraries

# In[1]:


import scipy.io as sio
import scipy.signal as ss
from scipy.signal import find_peaks, find_peaks, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from mpl_axes_aligner import align


# ## Task 1: Plot the ECG Signals

# ### i) Get ECG Data From .mat Files

# In[2]:


# Load files
normal_file = "A00001.mat"
normal = sio.loadmat(normal_file)['val']
af_file     = "A00004.mat"
atFib  = sio.loadmat(af_file    )['val']
noisy_file  = "A00585.mat"
noisy  = sio.loadmat(noisy_file )['val']
sample_freq = 300 # Sampling Frequnecy


# #### Get the Peaks of the ECG Signals

# In[3]:


norm_time  = np.arange(normal.size)/sample_freq # convert to time x axis with sample frequency
normal_array = np.zeros(normal.size)
for i in range(normal.size):
    normal_array[i] = normal[0][i] # put values from matlab file into an array
# find peaks, distance was included to remove peaks detected from noise
normal_peaks, _ = ss.find_peaks(normal_array, distance=150) 


# In[4]:


atFib_time    = np.arange(atFib.size )/sample_freq
atFib_array = np.zeros(atFib.size)
for j in range(atFib.size):
    atFib_array[j] = atFib[0][j]
atFib_peaks, _ = ss.find_peaks(atFib_array, distance = 200)


# In[5]:


noisy_time = np.arange(noisy.size )/sample_freq
noisy_array = np.zeros(noisy.size)   
for k in range(noisy.size):
    noisy_array[k] = noisy[0][k]
    noisy_peaks, _ = ss.find_peaks(noisy_array)


# #### Plot the ECG waveforms and some peaks

# In[6]:


fig, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True)

fig.set_figheight(5)
fig.set_figwidth(15)

ax1.plot(norm_time, normal_array)
ax1.plot(normal_peaks/sample_freq, normal_array[normal_peaks], "x")
ax1.set_xlabel("Time (s)"   )
ax1.set_xticks(np.arange(0, 31, step=1))
ax1.set_ylabel("Voltage (mV)")
ax1.set_title("Normal HR"   )

ax2.plot(atFib_time, atFib.T)
ax2.plot(atFib_peaks/sample_freq, atFib_array[atFib_peaks], "x")
ax2.set_xticks(np.arange(0, 31, step=1))
ax2.set_xlabel("Time (s)"             )
ax2.set_ylabel("Voltage (mV)"          )
ax2.set_title("Atrial Fibrillation HR")

# The last bit of the noisy signal swings extremely from high to low voltage, making it had 
# to see any other details. Thus the last 1000 datapoints were not graphed to perserve clarity
ax3.plot(noisy_time[:-1000], noisy.T[:-1000]) 
ax3.set_xticks(np.arange(0, 16, step=1))
ax3.set_xlabel("Time (s)"   )
ax3.set_ylabel("Voltage (mV)")
ax3.set_title("Noisy HR"    )

plt.show()


# ## Student Commentary
# - The normal heart rate appears to have a periodic distance between the peaks of the PQRST clusters.
# - In atrial fibrilation, the PQRST peaks are no longer appearing periodically. Also the T waves are really pronouced.
# - The noisy signal is not very easy to discern any waves in the PQRST clusters.

# ## Task 2: FFT Comparison of Different ECGs

# #### i) Spliit the ECG waveform into PQRST Segments

# In[7]:


def seg_length(peaks, peakType):
    seg_len = peaks[2] - peaks[1]
    print(f'The {peakType} Heartbeat has a PQRST segment length of {seg_len/300} seconds.')
    return int(seg_len)

def get_segments(array, peaks, peakType):
    pqrst_segments = []
    segment_length = int(seg_length(peaks, peakType))
    for i in range(len(peaks) - 1):
        # i = 0, a=0, b = segment_length
        # difference = b - a = segment_length
        # --
        # i = 1, a = segment_length, b = a + segment_length = segment_length + segment_length = 2*segment_length
        # difference = b - a = 2*segment_length - segment_length = segment_length
        # --
        # i = 2, a = 2*segment_length, b = a + segment_length = 2*segment_length + segment_length = 3*segment_length
        # difference = b - a = 3*segment_length - 2*segment_length = segment_length
        # --
        a = i*segment_length 
        b = a + segment_length
        s = normal_array[a:b]
        pqrst_segments.append(s)
    return pqrst_segments, peakType


# #### ii) Define Function to get ECG Signal, Windowed ECG Signal, FFT, and FFT Frequnecy.

# In[8]:


def padded_fft_comp(segments, padding_factor):
    ecg_segment = segments
    
    N = len(ecg_segment)
    
    # Get unpadded windowed segment
    unpadded_window = ecg_segment*np.bartlett(N)
    
    # Zero padding to reduce error
    N_padded = N * padding_factor
    
    # Get window
    padded_window = np.bartlett(N_padded)
    
    # Implementing zero padding
    zero_padded = np.pad(ecg_segment, (0, N_padded - N), mode='constant')

    # Multiply the padded ECG segments by the window
    padded_windowed_ecg_segment = zero_padded * padded_window
    
    fft_result  = np.fft.fft(padded_windowed_ecg_segment)
    freq = np.fft.fftfreq(len(fft_result), 1/sample_freq)
    return freq, fft_result, ecg_segment, unpadded_window


# ### Create Plots

# #### Get FFT for Two Segments and Define a Time Axis

# In[9]:


pqrst_segments, t = get_segments(normal_array, normal_peaks, "Normal")

# Alpha Segment
freq_alpha, fft_result_alpha, original_alpha, windowed_alpha = padded_fft_comp(pqrst_segments[1], 2)
norm_time_alpha = np.arange(original_alpha.size)/sample_freq

# Beta Segment
freq_beta, fft_result_beta, original_beta, windowed_beta = padded_fft_comp(pqrst_segments[2], 2)
norm_time_beta  = (np.arange(original_beta.size )/sample_freq)+norm_time_alpha[-1]


# #### Plot Original and Windowed ECG Segments

# In[10]:


# Set Subplots
fig2, (ax_original_alpha, ax_original_beta) =  plt.subplots(2, constrained_layout=True)

# Alpha Segment: original and windowed ECG
ax_original_alpha.plot(norm_time_alpha, original_alpha, '.', label='Original ECG Segment')
ax_original_alpha.plot(norm_time_alpha, windowed_alpha, label='Windowed ECG Segment')
ax_original_alpha.set_title('Normal ECG Segment Alpha')
ax_original_alpha.set_xlabel('Time (s)')
ax_original_alpha.set_ylabel('Voltage (mV)')
ax_original_alpha.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, fancybox=True, shadow=True)

# Beta Segment: original and windowed ECG
ax_original_beta.plot(norm_time_beta, original_beta, '.', label='Original ECG Segment: Beta')
ax_original_beta.plot(norm_time_beta, windowed_beta, label='Windowed ECG Segment: Beta')
ax_original_beta.set_title('Normal ECG Segment Beta')
ax_original_beta.set_xlabel('Time (s)')
ax_original_beta.set_ylabel('Voltage (mV)')

fig2.set_figheight(10)
fig2.set_figwidth(12)
plt.show()


# ## Student Commentary
# We obtain a segment length of approximately 0.76 seconds. by applying this throughout the Normal heartbeat signal, we isolate the PQRST segments. looking at the first two segments, we see the blue line shows the original ECG signal for the normal ECG, the orange line shows the Widowed ECG. I also used np.pad() to do padding. 
# 
# The window seems appropriate as it is removing the increases in amplitude away from the PQRST complex we are interested in. For example, in segment beta, we see that around 0.8 seconds the original blue signal dips with the same magnitude as the r-peak at 1.1 seconds. 
# 
# If we tried to take the FFt, we might see some differences which is unexpected for a periodic signal like the normal heartbeat

# #### iii) Plot the FFT for Each Segment

# In[11]:


fig3, ((ax_fft_alpha, ax_fft_beta), (ax_raw_fft_alpha, ax_raw_fft_beta)) =  plt.subplots(2,2, constrained_layout=True)
aL = len(fft_result_alpha)
aP2 = abs(fft_result_alpha/aL)
aP1 = aP2[:aL//2]*2
afr = sample_freq/aL*np.arange(0, aL/2)

# Alpha Segment FFT
ax_raw_fft_alpha.plot(freq_alpha, abs(fft_result_alpha))
ax_raw_fft_alpha.set_title('Raw FFT of Windowed Segment: Alpha')
ax_raw_fft_alpha.set_xlabel('Frequency (Hz)')
ax_raw_fft_alpha.set_ylabel('Magnitude')
ax_raw_fft_alpha.set_xlim([-50, 50])

ax_fft_alpha.plot(afr, aP1)
ax_fft_alpha.set_title('Single Sided Amplitude Spectrum: Alpha')
ax_fft_alpha.set_xlabel('Frequency (Hz)')
ax_fft_alpha.set_ylabel('Amplitude')
ax_fft_alpha.set_xlim([0, 50])

bL = len(fft_result_beta)
bP2 = abs(fft_result_beta/bL)
bP1 = aP2[:bL//2]*2
bfr = sample_freq/bL*np.arange(0, bL/2)

# Beta Segment FFT
ax_raw_fft_beta.plot(freq_beta, np.abs(fft_result_beta))
ax_raw_fft_beta.set_title('Raw FFT of Windowed Segment: Beta')
ax_raw_fft_beta.set_xlabel('Frequency (Hz)')
ax_raw_fft_beta.set_ylabel('Magnitude')
ax_raw_fft_beta.set_xlim([-50, 50])

ax_fft_beta.plot(bfr, bP1)
ax_fft_beta.set_title('Single Sided Amplitude Spectrum: Beta')
ax_fft_beta.set_xlabel('Frequency (Hz)')
ax_fft_beta.set_ylabel('Amplitude')
ax_fft_beta.set_xlim([0, 50])

fig3.set_figheight(6)
fig3.set_figwidth(12)
plt.show()


# ## Student Commentary
# The FFT looks good, The shapes are similar and the peaks ar around the centre. When we obtain a sigle sided amplitude semptrum of the normal heartbeat signal, we get the same result for both PQRST segments. The beta window has slightly less defined tall peaks which is expected as the original (blue) signal for it is not as smooth as the one for alpha.

# #### iv) Comapre the Normal, Atrial Fibrillation, and Noisy Signals in the Frequency Domain

# #### Get the signals into arrays

# In[12]:


# Create a numPy array with the normal heartbeat ecg data
atFib_array = np.zeros(atFib.size)
noisy_array = np.zeros(noisy.size)

for j in range(atFib.size):
    atFib_array[j] = atFib[0][j]
    
for k in range(noisy.size):
    noisy_array[k] = noisy[0][k]


# #### Compare the FFT of all the signals in the frequency domain

# In[13]:


signals = [normal_array, atFib_array, noisy_array]
t = [norm_time, atFib_time, noisy_time]

# Perform FFT on each signal
fft_results = [np.fft.fft(sig) for sig in signals]

# Frequency axis
f = [np.fft.fftfreq(len(fft_results[0]), (1/sample_freq)), np.fft.fftfreq(len(fft_results[1]), (1/sample_freq)), np.fft.fftfreq(len(fft_results[2]), (1/sample_freq))]

L=30



# Plot the signals and their frequency representations
plt.figure(figsize=(12, 8))

names = ["Normal", "Atrial Fibrillation", "Noisy"]

# Plot signals
for i in range(3):
    plt.subplot(3, 2, 2*i+1)
    plt.plot(t[i], signals[i])
    plt.title(f'{names[i]} Signal')
    plt.ylabel('Magnitude')

# Plot frequency domain representations
for i in range(3):
    aL = len(fft_results[i])
    aP2 = abs(fft_results[i]/aL)
    aP1 = aP2[:aL//2]*2
    afr = sample_freq/aL*np.arange(0, aL/2)
    plt.subplot(3, 2, 2*i+2)
    plt.plot(afr, aP1)
    #plt.plot(f[i], np.abs(fft_results[i]))
    plt.title(f'Frequency Domain - {names[i]} Signal')
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0, 50])

plt.tight_layout()
plt.show()


# ## Student Commentary
# As the signals becomes less periodic and noisier, peaks across the frequency domain become less prominent and higher frequency which makes sense. 
# - The normal heartbeat has very prominent peaks for many frequencies. 
# - The atrial fibrillation signal is characterized as a signal without a regular P-wave and irregular RR intervals. As a result, the frequency content of the peaks in the FFT are less prominent and more spread out due to the irregularity and lack of clear periodicity.
# - The noisy signal has even fewer and even more spread-out peaks. It is very random and not easy to see much, as expected.
# 

# #### iv) Compare the FFT of the PQRST Segment with the FFT of the Other Spectra

# In[14]:


plot_first = True

for i in range(4):
    plt.subplot(1, 4, 1*i+1)
    if plot_first == True: 
        plt.plot(freq_alpha, np.abs(fft_result_alpha))
        plt.title('PQRST', fontsize = 11)
        plt.xlabel('Frequency (Hz)', fontsize = 8)
        plt.ylabel('Magnitude ', fontsize = 8)
        plt.xlim([-50, 50])
        plt.yticks(np.arange(0, 700000, step=(700000/7)), fontsize = 8)
        plt.ylim([0, 700000])
        plot_first = False
        continue
    i=i-1
    plt.plot(f[i], np.abs(fft_results[i]))
    plt.title(f'{names[i]}', fontsize = 11)
    plt.xlabel('Frequency (Hz)', fontsize = 8)
    plt.xlim([-50, 50])
    plt.yticks([])
    plt.grid(axis = "y")
    plt.ylim([0, 700000])

plt.figure(figsize=(12, 6))
plt.tight_layout(pad=0.5)
plt.show()


# ## Student Commentary
# The single PQRST segment is on the left. I perserved the y axis to illustrate how the peaks become much larger in amplitude and more peaks across a wider variety of frequencies are apparent for the noisier and more less periodic signals. This makes sense.

# ## Task 3: Filtering

# ### a) Butterworth Filter

# In[15]:


def bwBand(low, high, fs, n):
    return butter(n, [low, high], btype='band', fs=fs)

def butterFilter(x, low, high, n, sample_freq):
    b, a = bwBand(low, high, sample_freq, n)
    y = filtfilt(b, a, x)
    return y


# In[16]:


butter_results = [butterFilter(sig, low=5, high=15, n=3, sample_freq=sample_freq) for sig in signals]

for i in range(3):
    plt.plot(t[i], signals[i], label='Unfiltered', color='r')
    plt.plot(t[i], butter_results[i], label='Butterworth', color='b')
    plt.title(f'{names[i]} Signal')

    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.show()


# ## Student Commentary
# The Butterworth filter changes the signal so it seems to have less noise, the peaks have lower amplitudes, and a bit less ambiguity on where the baseline is.

# ### b) Derivative Filter

# In[17]:


def derivativeFilter(x, b, a, T):
    b = b*(1/8*T)
    y = filtfilt(b, a, x)
    return y


# In[18]:


derivative_results = [derivativeFilter(x=sig, b=np.array([1, 2, 0, -2, -1]), a=1, T=sample_freq) for sig in butter_results]

for i in range(3):
    figA, axisA = plt.subplots()
    axisA.plot(t[i], derivative_results[i], label='Derivative', color='r')
    axisB = axisA.twinx()
    axisB.plot(t[i], signals[i], label='Unfiltered', color='b')
    plt.title(f'{names[i]}')
    align.yaxes(axisA, 0, axisB, 0)

    figA.legend()
    axisA.set_xlabel("Time [s]")
    axisB.set_ylabel("Magnitude (Unfiltered)")
    axisA.set_ylabel("Magnitude (Derivative)")
    plt.show()


# ## Student Commentary
# The derivative filter significantly enhances the features, namely the R-Peaks, and the Q, S, and Twaves but to a lesser extent. The peaks in the negative direction are significantly enhanced. This makes it so it is easy to detect peaks. Also, low frequency noise causing baseline drift is reduced, improving overall clarity.
# 
# Note: I aligned the signals so the baseline is at 0 so it is easier to compare. The orange signal pertains to the axis on the left which has a factor of 1e7 to it.

# ### c) Squared Results

# In[19]:


square_results = [deri**2 for deri in derivative_results]

for i in range(3):
    figB, axisC = plt.subplots()
    axisC.plot(t[i], derivative_results[i], label = 'Derivative', color='b')
    axisD = axisC.twinx()
    axisD.plot(t[i], square_results[i], label = 'Square Results', color='r')
    plt.title(f'{names[i]}')
    align.yaxes(axisC, 0, axisD, 0)
    figB.legend()
    axisC.set_xlabel("Time [s]")
    axisD.set_ylabel("Magnitude (Derivative)")
    axisC.set_ylabel("Magnitude (Square Results)")
    plt.show()


# ## Student Commentary
# The R-peaks very well defined. There are no longer any negative values in the filtered results which could be problematic for identifying Q and S waves vs P and T waves.
# 
# Note: I aligned the signals so the baseline is at 0 so it is easier to compare. The orange signal pertains to the axis on the left which has a factor of 1e7 to it.

# ### d) i) Signal Averaging

# In[20]:


def moveAVG(x, N):
    #output_signal = np.convolve(x, np.ones(N)/N, mode='valid')
    output_signal = np.zeros(len(x))
    for n in range(len(x)):
        for k in range(N):
            if n - k >= 0:
                output_signal[n] += x[n - k]
        output_signal[n] /= N
    return output_signal


# In[27]:


avg_results = [moveAVG(x=sig, N=30) for sig in square_results]

for i in range(3):
    figC, axisE = plt.subplots()
    axisE.plot(t[i], signals[i], label = 'Unfiltered', color='r')
    axisF = axisE.twinx()
    axisF.plot(t[i], avg_results[i], label = 'Moving Average', color='b')
    plt.title(f'{names[i]}')
    align.yaxes(axisE, 0, axisF, 0)
    figC.legend()
    axisE.set_xlabel("Time [s]")
    axisF.set_ylabel("Magnitude (Moving Average)")
    axisE.set_ylabel("Magnitude (Unfiltered)")
    plt.show()


# ## Student Commentary
# We see that there is significantly less noise at the lower frequencies and it is easier to see the ECG signal

# ### d) ii) R-Peaks considering a Minimum Distance

# In[38]:


min_peak_distance = 0.25 * sample_freq

norm_peaks = np.zeros(39)
atFib_peaks = np.zeros(32)
noisy_peaks = np.zeros(44)

peaks_now = [norm_peaks, atFib_peaks, noisy_peaks]

pps =[]

for i in range(3):
    plt.plot(t[i], avg_results[i], label='Signal', color='b')
    p, _ = ss.find_peaks(avg_results[i], distance = min_peak_distance, height=0.2e12)
    print(f'{len(p)} peaks were found for the {names[i]} signal')
    plt.scatter(p/sample_freq, avg_results[i][p], color='r')
    plt.title(f'{names[i]}')
    pps.append(p)
    align.yaxes(axisE, 0, axisF, 0)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.show()


# ## Student Commentary
# With the minimum R-peak distance set, the number of R-peaks detected in the signal as it gets les periodic decreases. This makes sense since there might be PQRST complexes where the R-R peak distance is smaller than the minimum.

# ### e) i) Adaptive Thresholding to mark R peaks

# In[44]:


class heart_rate():

    def __init__(self,signal,samp_freq, index):
        '''
        Initialize Variables
        :param signal: input signal
        :param samp_freq: sample frequency of input signal
        '''

        # Initialize variables
        self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (0 for i in range(8))

        self.T_wave = False          
        self.m_win = avg_results[index]
        self.b_pass = butter_results[index]
        self.samp_freq = 300
        self.signal = signals[index]
        self.win_150ms = round(0.15*self.samp_freq)

        self.RR_Low_Limit = 0
        self.RR_High_Limit = 0
        self.RR_Missed_Limit = 0
        self.RR_Average1 = 0
        self.TI1 = []
        self.TF1 = []
        self.TI2 = []
        self.TF2 = []


    def approx_peak(self):
        '''
        Approximate peak locations
        '''   

        # FFT convolution
        slopes = ss.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

        # Finding approximate peak locations
        for i in range(round(0.5*self.samp_freq) + 1,len(slopes)-1):
            if (slopes[i] > slopes[i-1]) and (slopes[i+1] < slopes[i]):
                self.peaks.append(i)  


    def adjust_rr_interval(self,ind):
        '''
        Adjust RR Interval and Limits
        :param ind: current index in peaks array
        '''

        # Finding the eight most recent RR intervals
        self.RR1 = np.diff(self.peaks[max(0,ind - 8) : ind + 1])/self.samp_freq   

        # Calculating RR Averages
        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        # Finding the eight most recent RR intervals lying between RR Low Limit and RR High Limit  
        if (ind >= 8):
            for i in range(0, 8):
                if (self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit): 
                    self.RR2.append(self.RR1[i])

                    if (len(self.RR2) > 8):
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)    

        # Adjusting the RR Low Limit and RR High Limit
        if (len(self.RR2) > 7 or ind < 8):
            self.RR_Low_Limit = 0.92 * RR_Average2        
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2


    def searchback(self,peak_val,RRn,sb_win):
        '''
        Searchback
        :param peak_val: peak location in consideration
        :param RRn: the most recent RR interval
        :param sb_win: searchback window
        '''

        # Check if the most recent RR interval is greater than the RR Missed Limit
        if (RRn > self.RR_Missed_Limit):
            # Initialize a window to searchback  
            win_rr = self.m_win[peak_val - sb_win + 1 : peak_val + 1] 

            # Find the x locations inside the window having y values greater than Threshold I1             
            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

              # Find the x location of the max peak value in the search window
            if (len(coord) > 0):
                for pos in coord:
                    if (win_rr[pos] == max(win_rr[coord])):
                        x_max = pos
                        break
            else:
                x_max = None

          # If the max peak value is found
            if (x_max is not None):   
                # Update the thresholds corresponding to moving window integration
                self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI                         
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1         

                # Initialize a window to searchback 
                win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) -1, x_max)]  

                # Find the x locations inside the window having y values greater than Threshold F1                   
                coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

                # Find the x location of the max peak value in the search window
                if (len(coord) > 0):
                    for pos in coord:
                        if (win_rr[pos] == max(win_rr[coord])):
                            r_max = pos
                            break
                else:
                    r_max = None

                # If the max peak value is found
                if (r_max is not None):
                    # Update the thresholds corresponding to bandpass filter
                    if self.b_pass[r_max] > self.Threshold_F2:                                                        
                        self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF                            
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1      

                        # Append the probable R peak location                      
                        self.r_locs.append(r_max)                                                


    def find_t_wave(self,peak_val,RRn,ind,prev_ind):
        if (self.m_win[peak_val] >= self.Threshold_I1): 
            if (ind > 0 and 0.20 < RRn < 0.36):
                # Find the slope of current and last waveform detected        
                curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms/2) : peak_val + 1]))
                last_slope = max(np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms/2) : self.peaks[prev_ind] + 1]))

                # If current waveform slope is less than half of last waveform slope
                if (curr_slope < 0.5*last_slope):  
                    # T Wave is found and update noise threshold                      
                    self.T_wave = True                             
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI 

            if (not self.T_wave):
                # T Wave is not found and update signal thresholds
                if (self.probable_peaks[ind] > self.Threshold_F1):   
                    self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI                                         
                    self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF 

                    # Append the probable R peak location
                    self.r_locs.append(self.probable_peaks[ind])  

                else:
                    self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF                   

        # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I1) or (self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
            self.NPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.NPKI  
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF


    def adjust_thresholds(self,peak_val,ind):
        if (self.m_win[peak_val] >= self.Threshold_I1): 
            # Update signal threshold
            self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI

            if (self.probable_peaks[ind] > self.Threshold_F1):                                            
                self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF 

                # Append the probable R peak location
                self.r_locs.append(self.probable_peaks[ind])  

            else:
                # Update noise threshold
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF                                    

        # Update noise thresholds    
        elif (self.m_win[peak_val] < self.Threshold_I2) or (self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.NPKI  
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF


    def update_thresholds(self):
        self.TI1.append(self.Threshold_I1)
        self.TF1.append(self.Threshold_F1)
        self.TI2.append(self.Threshold_I2)
        self.TF2.append(self.Threshold_F2)

        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1 
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False 

    def ecg_searchback(self):
        # Filter the unique R peak locations
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))

        # Initialize a window to searchback
        win_200ms = round(0.2*self.samp_freq)

        for r_val in self.r_locs:
            coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)

            # Find the x location of the max peak value
            if (len(coord) > 0):
                for pos in coord:
                    if (self.signal[pos] == max(self.signal[coord])):
                        x_max = pos
                        break
            else:
                x_max = None

            # Append the peak location
            if (x_max is not None):   
                self.result.append(x_max)


    def find_r_peaks(self):
        '''
        R Peak Detection
        '''

        # Find approximate peak locations
        self.approx_peak()

        # Iterate over possible peak locations
        for ind in range(len(self.peaks)):

            # Initialize the search window for peak detection
            peak_val = self.peaks[ind]
            win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms), min(self.peaks[ind] + self.win_150ms, len(self.b_pass)-1), 1)
            max_val = max(self.b_pass[win_300ms], default = 0)

            # Find the x location of the max peak value
            if (max_val != 0):        
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0])

            if (ind < len(self.probable_peaks) and ind != 0):
                # Adjust RR interval and limits
                self.adjust_rr_interval(ind)

                # Adjust thresholds in case of irregular beats
                if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit): 
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2

                RRn = self.RR1[-1]

                # Searchback
                self.searchback(peak_val,RRn,round(RRn*self.samp_freq))

                # T Wave Identification
                self.find_t_wave(peak_val,RRn,ind,ind-1)

            else:
              # Adjust threholds
              self.adjust_thresholds(peak_val,ind)

            # Update threholds for next iteration
            self.update_thresholds()

        # Searchback in ECG signal 
        self.ecg_searchback()

        return self.result, self.TI1, self.TF1, self.TI2, self.TF2


# In[45]:


ps = []
for i in range(3):
    signal = avg_results[i]

    # Find the R peak locations
    hr = heart_rate(signal,sample_freq, index=i)
    result_r_peaks, TI1, TF1, TI2, TF2 = hr.find_r_peaks()
    ps.append((i, result_r_peaks))
    result = np.array(result_r_peaks)

    # Clip the x locations less than 0 (Learning Phase)
    result = result[result > 0]

    # Plotting the R peak locations in ECG signal
    plt.figure(figsize = (20,4), dpi = 100)
    #plt.xticks(np.arange(0, len(signal/sample_freq)+1, 600))
    plt.plot(signal, color = 'r')        
    plt.scatter(result, signal[result]*1.6, color = 'b', s = 50, marker= '*')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("R Peak Locations")
    plt.show()


# ### e) ii) Pulse Train of the detected QR

# In[48]:


for k in range(3):
    plt.plot(signals[k], color='b', label=names[k])
    for peak in pps[k]:
        plt.axvline(x=peak, color='r', linestyle='-', ymax=0.9, ymin=0.45)

    plt.xlabel('Fre')
    plt.ylabel('Amplitude')
    plt.title('ECG Signals with Detected QRS Complexes')
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[42]:


def calcHR(peaks):
    values = [x / sample_freq for x in peaks]
    return sum(np.diff(values)) / len(np.diff(values))

def HRV(peaks):
    values = [x / sample_freq for x in peaks]
    return np.mean(np.diff(values)**2)

def getGood(peaks, d):
    #peaks, _  = ss.find_peaks(array, distance=d)
    print(f'HeartRate = {60/calcHR(peaks)} BPM')
    print(f'HearRate Variability = {HRV(peaks)}')

print('Normal')
getGood(pps[0], 150)
print('Atrial Fibrillation:')
getGood(pps[1], 200)
print('Atrial Fibrillation:')
getGood(pps[2], 100)


# ## Student Commentary
# The heart rate is calculated to be around 80 bpm. This makes sense since it is right in the middle of the range for average resting heart rate (60-100 bpm).
# 
# As expected the heartrate variability is very high for atrial fibrilation and noisy signals. The heart rate should be around 80 bpm but with the some sort of pathology
