#!/bin/bash
#SBATCH --job-name=mtt_a100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3

module load 2024
module spider CUDA/12.6.0
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0

python src/train.py --CB wandb.yaml --CD 3set-fourier.yaml --out MultiGPUtest