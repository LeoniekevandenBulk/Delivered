#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH -n 1
#SBATCH -p gpu

module load python/2.7.9

module load cuda

module load cudnn

module load gcc

export PYTHONPATH=$HOME/pythonpackages/lib64/python:$HOME/pythonpackages/lib/python:$PYTHONPATH

srun THEANO_FLAGS=device=cuda python2 Main.py
