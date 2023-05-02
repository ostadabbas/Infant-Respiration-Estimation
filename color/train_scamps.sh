#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=scamps_br_v1
module load cuda/10.2
module load anaconda3/2022.05
source activate rppg-toolbox
python main.py --config_file configs/train_configs/SCAMPS_SCAMPS_SCAMPS_TSCAN_BASIC.yaml
