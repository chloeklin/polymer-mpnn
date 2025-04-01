#!/bin/bash

#PBS -q normal
#PBS -P um09
#PBS -l ncpus=96
#PBS -l mem=150GB
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/um09
#PBS -l jobfs=20GB
#PBS -v PYTHONPATH=/scratch/um09/hl4138/polympnn-venv/lib/python3.9/site-packages



cd /scratch/um09/hl4138

module load python3/3.9.2 cuda/12.3.2 cudnn/8.9.7-cuda12
module list

source polympnn-venv/bin/activate

cd polymer-mpnn/
tar -xzf data.tar.gz -C $JOBFS

cd scripts/

python3 2D_mpnn.py \
    --train_path $JOBFS/data/smiles_train.csv \
    --val_path $JOBFS/data/smiles_val.csv \
    --test_path $JOBFS/data/smiles_test.csv


