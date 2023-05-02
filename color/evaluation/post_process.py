"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
import os
import h5py

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT', path=None):
    """Calculate video-level HR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz equals [45, 150] beats per min
        #[b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        # bandpass filter between [0.08, 0.5] Hz equals [5, 30] breaths per min
        # SCAMPS dataset has breath rate drawn normally from [8, 24] bpm range.
        # COHFACE dataset breath rate distribution - same as SCAMPS.
        # ACL dataset has breath rate for normal infants in the range [30, 50] breaths per min.
        # bandpass filter between [0.5, 0.8] = [30, 48]
        #[b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')    # SCAMPS and COHFACE dataset bandpass filter bw.
        [b, a] = butter(1, [0.1 / fs * 2, 1.0 / fs * 2], btype='bandpass')     # ACL dataset bandpass filter bw.
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    if hr_method == 'FFT':
        # Original low, high frequency cutoffs for Adult Heart/Respiration rate estimation.
        #hr_pred = _calculate_fft_hr(predictions, fs=fs, low_pass=0.08, high_pass=0.5)
        #hr_label = _calculate_fft_hr(labels, fs=fs, low_pass=0.08, high_pass=0.5)
        
        # Changing the frequency for infant respiration rate estimation
        hr_pred = _calculate_fft_hr(predictions, fs=fs, low_pass=0.1, high_pass=1.0)
        hr_label = _calculate_fft_hr(labels, fs=fs, low_pass=0.1, high_pass=1.0)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')

    if path and hr_method=='FFT':
        # Dump the predicted and label waveforms used for metric calculation.
        hf = h5py.File(os.path.join('/scratch/manne.sa/data/ACL_23/ACL/post_output', path+'_output.hdf5'), 'w')
        hf.create_dataset('respiration', data=predictions)
        hf.close()
    
        hf = h5py.File(os.path.join('/scratch/manne.sa/data/ACL_23/ACL/post_output', path+'_label.hdf5'), 'w')
        hf.create_dataset('respiration', data=labels)
        hf.close()

    if hr_method=='FFT':
        print("Respiration rate for video:{0} - output:{1}; label:{2}".format(path, hr_pred, hr_label))
    return hr_label, hr_pred

