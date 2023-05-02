# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2
from tqdm import tqdm
import h5py

# post proc lib imports
import scipy
from scipy.signal import butter
from scipy.sparse import spdiags

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

def vis_flow(im1, flow):
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    #cv2.imwrite('im2Warped_new.jpg', im2W[:, :, ::-1] * 255)
    #cv2.imwrite('outFlow.png', rgb)
    return rgb


# Respiration rate estimation from the flow estimated:
def resample_ppg(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    return np.interp(np.linspace(
        1, input_signal.shape[0], target_length), np.linspace(
        1, input_signal.shape[0], input_signal.shape[0]), input_signal)

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

# FFT for rate estimation
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

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument('-input', type=str, default='', help='input video filename')
parser.add_argument('-output', type=str, default='', help='output filename')
parser.add_argument('-sample_rate', type=int, default=1, help='sample rate')
args = parser.parse_args()

cap = cv2.VideoCapture(args.input)
out = h5py.File(args.output, 'w')
prev, frame_step, avg_flow_x, avg_flow_y = None, 0, [], [] 

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
print(num_frames)

for frame_step in tqdm(range(num_frames)):
    ret, img = cap.read()

    if not ret:
        raise ValueError("No video")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float) / 255.

    if  frame_step == 0:
        img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
        prev = img
        
    if frame_step > 0 and frame_step%args.sample_rate==0:
        # Reduce the time taken for flow computation.
        img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)

        u, v, im2W = pyflow.coarse2fine_flow(
            prev, img, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
 
        flow_u, flow_v = flow[..., 0], flow[..., 1]
        avg_flow_x.append(2.0*flow_u.mean())    
        avg_flow_y.append(2.0*flow_v.mean())
        prev = img

# post-proc configs
fs=30
[b, a] = butter(1, [0.1 / fs * 2, 1.0 / fs * 2], btype='bandpass')     # infant dataset bandpass filter bw.
flowx=np.array(avg_flow_x)
flowx = resample_ppg(flowx, num_frames)

# post-proc
predictions_x = scipy.signal.filtfilt(b, a, np.double(flowx))
rr_x = _calculate_fft_hr(predictions_x, fs=fs, low_pass=0.1, high_pass=1.0)
print('Estimated respiration rate for the video:', rr_x)

cap.release()
out.create_dataset('flow_x', data=avg_flow_x)
out.create_dataset('flow_y', data=avg_flow_y)
out.close()