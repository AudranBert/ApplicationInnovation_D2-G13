#!/bin/bash
#SBATCH --job-name=predictionNote
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-app.out


python3 camembert/camembert.py > log-app.txt
