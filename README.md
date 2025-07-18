# FluidGPT
Foundational modelling for Navier-Stokes, Auto-regressive (deterministic) approach VS (probabilistic) One-shot Flowmatching approach

conda activate grad312
python src/train.py --CB [] -- CD [] --CM [] --CT [] --out folder --name wandb_name
lalala

# runcommands WS:
CUDA_VISIBLE_DEVICES=1 python src/train.py 

# runcommands HPC:
salloc --partition=gpu_mig --time=04:00:00 --gpus-per-node=1 --reservation=terv92681
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0

single run:
sbatch submit/simple-reservation-short.sh

multiple runs:
for i in {1..10}; do sbatch submit/jhs-2hr-$i.sh; done

# runcommands docker:
docker build -t _username_/_imagename_:_version_ .

docker push _username_/_imagename_:_version_

docker run --rm --gpus all --shm-size=6g _username_/_imagename_:_version_

# debug current bug
CUDA_LAUNCH_BLOCKING=1 python src/train.py
