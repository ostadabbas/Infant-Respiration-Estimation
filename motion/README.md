# Motion-based respiration signal estimation

## Setup
`python setup.py build_ext -i`

## Tests
Run the following command to get flow signals and the respiration rate for a single clip in the folder:
```
python video_flow.py -input 103_01.mp4 -output 103_01_rr.hdf5 -sample_rate 5
```

Run the following command to get flow signals for COHFACE test data:
```
python cohface_eval.py -input cohface-dir -sample_rate 5
```
