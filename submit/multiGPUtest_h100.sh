#!/bin/bash
#SBATCH --job-name=mtt_h100
#SBATCH --output=slurmlogs/%x_%j.out
#SBATCH --error=slurmlogs/%x_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=05:00:00
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

python src/train.py --CB wandb_highfreq.yaml --CD 5set-fourier.yaml --CT 3_h100.yaml --CM bigger.yaml --out 3h100run