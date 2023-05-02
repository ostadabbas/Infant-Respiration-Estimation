import os, sys, glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy
import h5py
import cv2
import json
from math import ceil
import re

# Generate.hdf5 files from annotations in excel files - may not align with number of frames exactly well.
path = './annotations/'
fns = [f for f in os.listdir(path) if '.json' in f]

for fn in fns:
    in_fn = os.path.join(path, fn)
    annotations = json.load(open(in_fn))['metadata']
    timestamps = [10*round(timestamp['z'][0], 1) for timestamp in annotations.values()]*10
    
    # decide the start and end-times for the video - create an impulse signal
    vid_fn = os.path.join('infants/', re.sub('_rr_[0-9a-zA-Z]+.json', '.mp4', fn))
    cap = cv2.VideoCapture(vid_fn)
    #print(cap.get(cv2.CAP_PROP_FRAME_COUNT), round(cap.get(cv2.CAP_PROP_FRAME_COUNT)/30.0, 1), timestamps[-1], 10*ceil(timestamps[-1]))
    
    end_time = ceil(timestamps[-1])
    timeline = [int(t) for t in np.linspace(0, end_time, end_time)]
    impulse = [1.0 if t in timestamps else 0.0 for t in timeline]
    
    # convolve with a guassian kernel to create a smooth respiration signal
    wave_form = scipy.ndimage.gaussian_filter1d(impulse, 2)
    
    out_fn = re.sub('_rr_[0-9a-zA-Z]+.json', '.hdf5', fn)
    hf = h5py.File(os.path.join(path, out_fn), 'w')
    hf.create_dataset('respiration', data=wave_form)
    hf.create_dataset('filename', data=out_fn.replace('hdf5', 'mp4'))
    hf.close()