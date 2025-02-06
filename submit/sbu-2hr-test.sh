#!/bin/bash
#SBATCH --job-name=sbu2hr-t
#SBATCH --partition=gpu_mig
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

module load 2024
module load Anaconda3/2024.06-1
. "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

python src/train.py --conf conf/hpc_01DS4_10.yaml --trainer PFTB