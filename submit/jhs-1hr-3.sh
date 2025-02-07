#!/bin/bash
#SBATCH --job-name=jhs1hr3
#SBATCH --partition=gpu_mig
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --reservation=jhs_tue2022

module load 2023
module load Anaconda3/2023.07-2
. "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh"
conda activate grad311_hpc2023
cd AutoregressiveNeuralOperators

python src/train.py --conf conf/hpc_01DS4_3.yaml --trainer PFTB