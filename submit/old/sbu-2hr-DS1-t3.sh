#!/bin/bash
#SBATCH --job-name=t3a100
#SBATCH --partition=gpu_a100
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# #SBATCH -N 1
module load 2023
module load Anaconda3/2023.07-2
. "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh"
conda activate grad311_hpc2023
cd AutoregressiveNeuralOperators

python src/train.py --conf conf/hpc_01DS1_1.yaml --trainer PFTB