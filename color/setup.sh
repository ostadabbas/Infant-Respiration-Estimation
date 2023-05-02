#conda remove --name rppg-toolbox --all -y
#conda create -n rppg-toolbox python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch -q -y
module load cuda/10.2
module load anaconda3/2022.05
source activate rppg-toolbox
