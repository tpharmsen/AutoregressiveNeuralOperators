#!/bin/bash
#SBATCH --job-name=train_BubbleML-2
#SBATCH --partition=gpu_mig
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1


###SBATCH --reservation=jhs_tue2022

module load 2022
module load Anaconda3/2022.05
. "/sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

python -c "import torch; print('cuda available: ', torch.cuda.is_available())"
python src/train.py --conf conf/example.yaml --trainer simple