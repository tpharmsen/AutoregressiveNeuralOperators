#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --reservation=terv92681
#SBATCH --wrap="sleep infinity"

module load 2024
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

