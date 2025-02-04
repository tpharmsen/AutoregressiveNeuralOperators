#!/bin/bash
#SBATCH --job-name=sbu2hr1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# #SBATCH -N 1
module load 2024
module load Anaconda3/2024.06-1
. "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

python src/train.py --conf conf/hpc_01DS1_1.yaml --trainer PFTB