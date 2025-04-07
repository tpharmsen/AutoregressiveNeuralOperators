#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_a100
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
## SBATCH --reservation=terv92681

module load 2024
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda env list
conda activate grad312
conda env list
cd AutoregressiveNeuralOperators

python src/train.py --CB wandb.yaml --CD 3set-bicubic.yaml --out /bicubic --name bicubic