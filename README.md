# ECG_Analysis
The objective of this code is to analyze the data associated three different ECG signal waveforms: a normal heartbeat, a heart undergoing atrial fibrillation, and a noisy ECG signal. First, fast Fourier transforms (FFTs) are used to compare segments and spectra in the frequency domain. Next, transfer functions from the Pan-Tompkins Algorithm [1] is used on each ECG signal to identify R peaks in QRS clusters by applying a series of filters and an adaptive thresholding method. Finally, the peaks can be used to calculate the heart rate and heart rate variability for each ECG signal using Equation 1.

𝐻R [𝑏pm] =60/𝑅R [𝑠ec] (Eq. 1)

Where RR is the distance between consecutive R-peaks in QRS complexes of an ECG.


Results

![image](https://github.com/Ekin-Deren-Yelken/ECG_Analysis/assets/128660105/954db25e-0de9-4907-ac9a-d50d677481b0)

-> Unfiltered signals with R-peaks marked


![image](https://github.com/Ekin-Deren-Yelken/ECG_Analysis/assets/128660105/9ab298ba-0a8d-4ec4-8a25-7af5a308d097)

-> Normal Heartbeat segment isolation and windowing with zero padding.


![image](https://github.com/Ekin-Deren-Yelken/ECG_Analysis/assets/128660105/7ac706b8-747c-4342-8ad8-b8315d0b0276)

-> FFT of Normal Heartbeat segment


![image](https://github.com/Ekin-Deren-Yelken/ECG_Analysis/assets/128660105/ac46aed2-2a70-4fef-9b08-5fdefd8f36d2)

-> Unfiltered signals compared to the result of a 5-step filtering process form Pan-Tompkins [1]


![image](https://github.com/Ekin-Deren-Yelken/ECG_Analysis/assets/128660105/e3881af4-5ad0-4824-878e-37d22a3aa0c0)

-> R-Peak detection using adaptive thresholding on filtered signals


![image](https://github.com/Ekin-Deren-Yelken/ECG_Analysis/assets/128660105/76a4c093-6e26-423b-a9b4-26f95b9d911c)

-> Heart rate and heart rate variability calculations for each signal



References and Credits

Generative AI was used to supplement code. 
The adaptive thresholding code to find R-peaks was adapted from an external source [2].

[1] Pan, J. and Tompkins, W., 1985. A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering, BME-32(3), pp.230-236.
[2] K. Sharma, “Antimatercorrade/pan_tompkins_qrs_detection: Pan Tompkins QRS wave detection algorithm python implementation,” GitHub, htps://github.com/antimatercorrade/Pan_Tompkins_QRS_Detection/tree/main
