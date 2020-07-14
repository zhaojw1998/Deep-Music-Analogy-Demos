#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-01:10:00
#SBATCH --mem=200G
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err

module purge
module load python/intel/3.5.i
module load pytorch/gpu/1.3.1-py3.5

python < train.py
