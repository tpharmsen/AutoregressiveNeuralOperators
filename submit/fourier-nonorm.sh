#!/bin/bash
#SBATCH --job-name=fftnorm
#SBATCH --partition=gpu_mig
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --reservation=terv92681

module load 2024
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda env list
conda activate grad312
conda env list
cd AutoregressiveNeuralOperators

srun python src/train.py --CB wandb.yaml --CD 3set-fourier.yaml --out /fourier-nonorm --name fourier-nonorm --CT nonorm.yaml