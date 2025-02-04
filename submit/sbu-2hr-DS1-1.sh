#!/bin/bash
#SBATCH --job-name=fullres
#SBATCH --partition=gpu_h100
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5G

module load 2022
module load Anaconda3/2022.05
. "/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

python src/train.py --conf conf/hpc_01DS1_1.yaml --trainer PFTB