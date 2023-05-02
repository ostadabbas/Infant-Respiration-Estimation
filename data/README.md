# Dataset creation

# Preprocessing
Steps:
1. Source clean (noise and compression artefact free) videos with still infants from YouTube.
2. Trim the arbitrary length videos into shorter clips of 30-60 seconds.
3. Annotate each clip with exhalation start points using [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/app/via_video_annotator.html)
4. Group the annotations in a folder `annotations` and run the following script:
```
python waveform_gen.py
```
5. waveforms will be stored as .hdf5 files.

## Dataset
We release the public dataset in the `dataset.xlsx` with the following information:
1. Video Link
2. Start and end times for each clip. 