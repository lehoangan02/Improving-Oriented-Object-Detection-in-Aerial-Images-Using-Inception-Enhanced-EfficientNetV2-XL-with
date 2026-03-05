#!/bin/bash
#SBATCH --job-name=baseline_train_test
#SBATCH --output=effntv2_train_test.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $(hostname)"
echo "Submit time: $(scontrol show job $SLURM_JOB_ID | grep SubmitTime | awk -F= '{print $2}')"
echo "Start time: $(date)"
echo "========================================"

start_time=$(date +%s)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /media02/hvtham/conda_envs/myenv

PROJECT_DIR=/media02/hvtham/BBAV/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit

cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA available check:"
python -c "import torch; print(torch.cuda.is_available())"

if [ ! -f "$DEVKIT_DIR/polyiou.cpython-*.so" ]; then
    echo "Building polyiou..."
    cd $DEVKIT_DIR
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
    cd $PROJECT_DIR
fi

python main.py \
  --data_dir /media02/hvtham/DATA/BridgeTrain \
  --num_epoch 50 \
  --batch_size 8 \
  --dataset dota \
  --phase train \
  --conf_thresh 0.1

end_time=$(date +%s)

echo "========================================"
echo "End time: $(date)"
echo "Total runtime: $((end_time - start_time)) seconds"
echo "========================================"
