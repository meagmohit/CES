#!/bin/csh
#SBATCH -A asccasc
#SBATCH -N 1
#SBATCH --partition=surface
#SBATCH -t 22:00:00
#SBATCH -p pbatch
#SBATCH --export=ALL
#
module load cudatoolkit/8.0
use tensorflow-gpu-0.12.1
srun -N1 -n1 python train.py --mode=train
