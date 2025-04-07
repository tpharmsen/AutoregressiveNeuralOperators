#!/bin/bash
#SBATCH --job-name=swinUnetTest
#SBATCH --partition=gpu_mig
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --reservation=terv92681

module load 2023
module load Anaconda3/2023.07-2
. "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh"
conda activate grad311_hpc2023
cd AutoregressiveNeuralOperators
pip install einops

python src/train.py --conf conf/hpc_01_128_1.yaml --trainer PFTB