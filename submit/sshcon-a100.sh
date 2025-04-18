#!/bin/bash
#SBATCH --job-name=sshconnect
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#BATCH --reservation=terv92681

module load 2024
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

