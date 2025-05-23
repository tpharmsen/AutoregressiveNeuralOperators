#!/bin/bash
#SBATCH --job-name=mtt_h100
#SBATCH --output=slurmlogs/%x_%j.out
#SBATCH --error=slurmlogs/%x_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2

module load 2024
module load CUDA/12.6.0
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0

srun python src/train.py --CB wandb_highfreq --CD quick3 --CT std --CM swiglu --out utiltest