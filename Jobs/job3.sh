#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --output=eval_all_attempt2_1.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=48:00:00

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /media02/hvtham/conda_envs/myenv

PROJECT_DIR=/media02/hvtham/BBAV/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit
WEIGHTS_DIR=$PROJECT_DIR/weights_dota/attempt2
OUTPUT_DIR=$PROJECT_DIR/eval_results

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

for i in $(seq 1 50); do
  echo "========================================"
  echo "Evaluating model_${i}.pth"
  echo "========================================"

  WEIGHTS="$WEIGHTS_DIR/model_${i}.pth"
  if [ ! -f "$WEIGHTS" ]; then
    echo "Missing weights: $WEIGHTS (skipping)"
    continue
  fi

  python main.py \
    --data_dir /media02/hvtham/DATA/DOTA_VAL_608 \
    --batch_size 16 \
    --dataset dota \
    --phase eval \
    --conf_thresh 0.1 \
    --resume "$WEIGHTS"

  echo "Running DOTA evaluation..."

  cd "$DEVKIT_DIR"
  python dota_evaluation_task1.py > "$OUTPUT_DIR/eval_model_${i}.txt"
  cd "$PROJECT_DIR"
done

echo "========================================"
echo "Finished all evaluations"
echo "End: $(date)"
echo "Results saved in $OUTPUT_DIR"
echo "========================================"
