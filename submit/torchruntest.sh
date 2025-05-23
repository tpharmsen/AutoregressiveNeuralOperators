#!/bin/bash
#SBATCH --job-name=mtt_h100
#SBATCH --output=slurmlogs/%x_%j.out
#SBATCH --error=slurmlogs/%x_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

module load 2024
module load CUDA/12.6.0
#module spider CUDA/12.6.0
module load Anaconda3/2024.06-1
source "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
conda activate grad312
cd AutoregressiveNeuralOperators

export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0

echo "Running on host: $(hostname)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi

srun python src/train.py \
    --CB wandb_highfreq \
    --CD 5set-fourier \
    --CT deepspeed \
    --CM swiglu \
    --out torchruntest