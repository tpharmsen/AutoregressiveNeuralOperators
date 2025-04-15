#!/bin/bash
#SBATCH --job-name=sshconnect
#SBATCH --partition=gpu_mig
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --reservation=terv92681

