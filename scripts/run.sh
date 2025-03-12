#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l walltime=00:10:00
#PBS -l storage=scratch/um09
#PBS -l jobfs=100GB
#PBS -v PYTHONPATH=/scratch/um09/hl4138/llm-venv/lib/python3.10/site-packages



cd /scratch/um09/hl4138

module load python3/3.9.2 cuda/12.3.2 cudnn/8.9.7-cuda12
module list

source polympnn-venv/bin/activate

cd polymer-mpnn

python3 scripts/2D_mpnn.py 