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
import os

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument('-input', type=str, default='', help='input video filename')
parser.add_argument('-output', type=str, default='', help='output video/npy filename')
parser.add_argument('-sample_rate', type=int, default=5, help='sample rate')
args = parser.parse_args()

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

def flow_rr(input_fn, output_fn, gt_fn):
    cap = cv2.VideoCapture(input_fn)
    out = h5py.File(output_fn, 'w')
    gt = h5py.File(gt_fn, 'r')
    prev, frame_step, avg_flow_x, avg_flow_y = None, 0, [], [] 

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print(num_frames)

    #while cap.isOpened():
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

    cap.release()
    out.create_dataset('flow_x', data=avg_flow_x)
    out.create_dataset('flow_y', data=avg_flow_y)
    out.create_dataset('gt_rr', data=gt['respiration'][:])
    out.close()

subs = [dirn for dirn in os.listdir('cohface') if os.path.isdir(os.path.join('cohface', dirn))]
print(subs)

for sub in subs:
    for fn in os.listdir(os.path.join('cohface', sub)):
        video_fn = os.path.join(os.path.join('cohface', sub), fn) + '/data.avi'
        out_fn = os.path.join(os.path.join('cohface', sub+'_'+fn+'.hdf5'))
        gt_fn = os.path.join(os.path.join('cohface', sub), fn) + '/data.hdf5'

        if os.path.exists(out_fn):
            continue
        
        print(video_fn)
        flow_rr(video_fn, out_fn, gt_fn)
