#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=output2.log
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

source /home/hvtham/miniconda3/bin/activate
conda activate myenv
cd /media02/hvtham/group3/BBAVectors-Oriented-Object-Detection
git pull
git switch conventional
cd Custom/Scripts

echo "Running evaluation for all models in the model directory..."
# Define variables
validate_data_dir="./datasets/Validate_DOTA_1_0.5"
training_data_dir="./datasets/trainsplit"
# Directory structure:
# -|datasets
# ---|your_validation_data
# -----|images
# -----|labelTxt
conf_thresh=0.1
batch_size=16
dataset="dota"
phase="eval"
model_dir="combinedV1_chpc"
# Directory structure:
# -|weights_dota
# ---| your_model_weights
# -----| model_5.pth
# -----| model_10.pth
# -----| ...
eval_script="dota_evaluation_task1.py"
eval_dir="datasets/DOTA_devkit"
result_dir="Result/combinedV1_chpc"
# A folder named "Result" will be created in the current directory to store the evaluation results

# Array of model epochs to evaluate
epochs=(1 2 3)
cd ../..

# Create the result directory if it doesn't exist
mkdir -p "Result"
mkdir -p "$result_dir"

# Loop through the epochs and run the evaluation
for epoch in "${epochs[@]}"; do
    model_path="${model_dir}/model_${epoch}.pth"
    echo "Running validate evaluation for model at epoch ${epoch}..."
    python3 main.py --data_dir "$validate_data_dir" --conf_thresh "$conf_thresh" --batch_size "$batch_size" --dataset "$dataset" --phase "$phase" --resume "$model_path"
    
    # Change directory to evaluation script location and run evaluation
    echo "Running DOTA evaluation for model at epoch ${epoch}..."
    (cd "$eval_dir" && python "$eval_script") | tee "$result_dir/validate_evaluation_result_for_epoch_${epoch}.txt"

    echo "Running training evaluation for model at epoch ${epoch}..."
    python3 main.py --data_dir "$training_data_dir" --conf_thresh "$conf_thresh" --batch_size "$batch_size" --dataset "$dataset" --phase "$phase" --resume "$model_path"

    # Change directory to evaluation script location and run evaluation
    echo "Running DOTA evaluation for model at epoch ${epoch}..."
    (cd "$eval_dir" && python "$eval_script") | tee "$result_dir/training_evaluation_result_for_epoch_${epoch}.txt"

done
