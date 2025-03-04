# AutoregressiveNeuralOperators
A small repo for autoregressive neural operator training

# runcommands PC:
python src/train.py --conf conf/example.yaml --trainer PFTB

# runcommands WS:
CUDA_VISIBLE_DEVICES=1 python src/train.py --conf conf/local_01DS8_1.yaml >> testWSV.out

# runcommands HPC:
single run:
sbatch submit/simple-reservation-short.sh

multiple runs:
for i in {1..10}; do sbatch submit/jhs-2hr-$i.sh; done