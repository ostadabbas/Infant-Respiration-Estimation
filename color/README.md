# Color-based respiration signal estimation

## Setup:
Install the required libraries and packages:
`pip install -r requirements.txt` or through the following command:
```
. ./setup.sh
```

## Training:
Modify the appropriate config file to use the desired train, validation, and test datasets for the training. 

Sample training command to train, validate and test on the SCAMPS dataset using the TS-CAN model:
```
python main.py --config_file configs/train_configs/SCAMPS_SCAMPS_SCAMPS_TSCAN_BASIC.yaml
``` 
## Testing:
Specify the pretrained model path in the config file and run the following command:
```
python main.py --config_file configs/infer_configs/SCAMPS_SCAMPS_SCAMPS_TSCAN_BASIC.yaml
```