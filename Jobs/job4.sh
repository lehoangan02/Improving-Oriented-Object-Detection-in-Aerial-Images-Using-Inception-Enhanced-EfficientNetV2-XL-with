#!/bin/bash
#SBATCH --job-name=eval50
#SBATCH --output=eval50.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /media02/hvtham/conda_envs/myenv

PROJECT_DIR=/media02/hvtham/BBAV/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit
WEIGHTS=$PROJECT_DIR/weights_dota/attempt1/model_50.pth

cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

echo "Python: $(which python)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "Running evaluation inference with model_50.pth"

python main.py \
  --data_dir /media02/hvtham/DATA/DOTA_VAL_608 \
  --batch_size 16 \
  --dataset dota \
  --phase eval \
  --conf_thresh 0.1 \
  --resume $WEIGHTS

echo "========================================"
echo "Finished inference"
echo "End time: $(date)"
echo "========================================"
