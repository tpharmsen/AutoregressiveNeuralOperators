#!/bin/bash
#SBATCH --job-name=jhs-2hr-5
#SBATCH --partition=gpu_mig
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --reservation=jhs_tue2022

module load 2022
module load Anaconda3/2022.05
. "/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

python src/train.py --conf conf/hpc_01DS4_5.yaml --trainer PFTB