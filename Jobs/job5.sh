#!/bin/bash
#SBATCH --job-name=download_dota_train
#SBATCH --output=download_dota_train.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

cd /media02/hvtham/DATA

echo "Downloading DOTA_TRAIN_608.zip..."

wget "https://huggingface.co/datasets/lehoangan02/DOTA_608_TRAIN/resolve/main/DOTA_TRAIN_608.zip?download=true" -O DOTA_TRAIN_608.zip

echo "Download finished"

echo "Unzipping..."

unzip DOTA_TRAIN_608.zip

echo "Unzip finished"

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
