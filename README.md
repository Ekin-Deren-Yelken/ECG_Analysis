# ECG_Analysis
Analyzing ECG Signals Using MATLAB

The objective of this code is to analyze the data associated three different ECG signal waveforms following a normal heartbeat, a heart undergoing atrial fibrillation, and a noisy ECG signal. First, fast Fourier transforms (FFTs) are used to compare segments and spectra in the frequency domain. Next, the Pan-Tompkins Algorithm is used on each ECG signal to identify R peaks in QRS clusters by applying a series of filters (ButterWorth, Derivative, Squaring, and Moving Average) to clean the data. R-peaks are then found using findpeaks() and an adaptive thresholding method. 

The peaks can be used to calculate the heart rate (HR) and Hear rate variability (HRV) via root mean square of successive differences (RMSSD)

### Heart Rate (HR) Calculation

The heart rate (HR) is calculated using the formula:


$HR (bpm) = \frac{60}{RR_{\text{interval}} \ (s)}$


Where:

- $HR$ > Heart Rate in beats per minute (bpm)
- $RR_{\text{interval}}$ > Time between consecutive R-peaks in seconds

---

#### Heart Rate Variability (HRV) Calculation

The root mean square of successive differences (RMSSD) is calculated as:

$RMSSD = \sqrt{\frac{ \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}{N}}$

Where:

- $RR_i$ = $RR_{\text{interval}}$ > The i-th RR interval
- $N$ > Total number of RR intervals
- $RMSSD$ > provides an estimate of short-term HRV

---

#### Raw ECG Signal - Time Domain
![Raw ECG Signal - Time Domain](https://github.com/user-attachments/assets/7f86f6d0-10b1-43f8-a4a6-563a8aacf1d9)

### Fourier Transformed ECG Signal
![Fourier Transformed ECG Signal](https://github.com/user-attachments/assets/d0b10304-e27e-4d66-8d6e-e4c11dd9a50c)

### Filtered ECG Signal - Time Domain
![Filtered ECG Signal - Time Domain](https://github.com/user-attachments/assets/86fa545f-291c-4ad0-96db-da85bd85151b)

### R-Peaks ECG Signal (findpeaks) - Time Domain
![R-Peaks ECG Signal (findpeaks) - Time Domain](https://github.com/user-attachments/assets/433bc363-7373-40a0-9a0b-f7d542fd268b)

### R-Peaks ECG Signal (Adaptive Thresholding) - Time Domain
![R-Peaks ECG Signal (Adaptive Thresholding) - Time Domain](https://github.com/user-attachments/assets/ffa286b1-aa79-4f41-89e3-597e930a28ea)

---

### Analysis

Normal HR: 78.88 bpm

Normal HRV: 0.03

The normal heart rate seems to be around 80 bpm, which is an average resting heart rate. It is expected that the HRV is close to 0 because some level of variability is normal and healthy.

--

Atrial Fibrillation HR: 65.60 bpm

Atrial Fibrillation HRV: 0.23

The atrial fibrilla􀆟on is, lower around 65 bpm. While this is still in the range of average resting heart rates it is close to the botom end of the range and the extreme HRV of 0.23 suggests pathology.

--

Noisy HR: 103.22 bpm

Noisy HRV: 0.38

The noisy filter has much hgher heart rate and variability. The fact that there is only 18 seconds of data could contribute to the fact of the doubling. If there were more data and more filtering, it could be possible to recover more reasonable results.
