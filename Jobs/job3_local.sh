#!/bin/bash

set -euo pipefail

echo "========================================"
echo "Running local evaluation"
echo "Machine: $(hostname)"
echo "Start: $(date)"
echo "========================================"

PROJECT_DIR=/home/lhan/Documents/paper/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit
WEIGHTS_DIR=$PROJECT_DIR/weights_dota/attempt1
OUTPUT_DIR=$PROJECT_DIR/eval_results

mkdir -p $OUTPUT_DIR

cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

for i in $(seq 2 50)
do
echo "========================================"
echo "Evaluating model_$i.pth"
echo "========================================"

WEIGHTS=$WEIGHTS_DIR/model_${i}.pth

python main.py \
  --data_dir /home/lhan/Documents/paper/DATA/DOTA_VAL_608 \
  --batch_size 16 \
  --dataset dota \
  --phase eval \
  --conf_thresh 0.1 \
  --resume $WEIGHTS

echo "Running DOTA evaluation..."

cd $DEVKIT_DIR
python dota_evaluation_task1.py > $OUTPUT_DIR/eval_model_${i}.txt
cd $PROJECT_DIR

done

echo "========================================"
echo "Finished all evaluations"
echo "End: $(date)"
echo "Results saved in $OUTPUT_DIR"
echo "========================================"