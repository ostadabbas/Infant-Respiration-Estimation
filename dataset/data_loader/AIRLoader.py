"""The dataloader for AIR dataset.

Dataset contains in-the-wild infant and adult videos with GT respiration waveforms manually annotated.
Same dataloader can be used to load flow videos if similar naming and directory structure is followed.
"""
import glob
import os
import re

import cv2
import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class AIRLoader(BaseLoader):
    """The data loader for the ACL dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an AIR dataloader.
            Args:
                data_path(str): path of a folder which stores raw/flow videos and gt data.
                -----------------
                     AIR/
                     |   |-- D01/
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |      |...
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |...
                     |   |-- Y01/
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |      |...
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (For AIR dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            dir_len = len(glob.glob(data_dir + os.sep + "*"))  # Supports arbitrary number of videos in each subject.
            for i in range(dir_len):
                subject = os.path.split(data_dir)[-1]
                dirs.append({"index": '{0}_{1}'.format(subject, i),
                             "path": os.path.join(data_dir, str(i).zfill(3))})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []
        for i in choose_range:
            data_dirs_new.append(data_dirs[i])
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        video_fn = glob.glob(data_dirs[i]['path']+'/*.mp4')[0]
        label_fn = glob.glob(data_dirs[i]['path']+'/*.hdf5')[0]

        frames = self.read_video(
            video_fn)
        bvps = self.read_wave(
            label_fn)
        
        # Better resample using scipy resampling instead of numpy interpolation
        target_length = frames.shape[0]
        #bvps = BaseLoader.resample_air(bvps, target_length)
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        sample_rate = int(VidObj.get(cv2.CAP_PROP_FPS))//5

        frame_count = 0
        while (success):
            if frame_count%sample_rate == 0:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
                frames.append(frame)
            frame_count += 1
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        f = h5py.File(bvp_file, 'r')
        resp = f['respiration'][:]
        return resp
