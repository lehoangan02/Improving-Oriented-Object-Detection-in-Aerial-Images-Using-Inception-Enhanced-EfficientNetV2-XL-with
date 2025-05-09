#!/bin/bash

# Things you need to change:
# 1. data_dir: Path to the directory containing the validation data
# 2. model_dir: Path to the directory containing the model weights

# How to run:
# Make sure you are using Git bash if you are on Windows
# cd Custom/Scripts
# chmod +x evaluate_models.sh
# ./evaluate_models.sh
echo "Running evaluation for all models in the model directory..."
# Define variables
data_dir="./datasets/Validate_DOTA_1_0.5"
# data_dir="./datasets/MiniTrainV1.1" # testing
# Directory structure:
# -|datasets
# ---|your_validation_data
# -----|images
# -----|labelTxt
conf_thresh=(0.1)
batch_size=30
dataset="dota"
phase="eval"
model_dir="ImplicitCorners"
# Directory structure:
# -|weights_dota
# ---| your_model_weights
# -----| model_5.pth
# -----| model_10.pth
# -----| ...
eval_script="dota_evaluation_task1.py"
eval_dir="datasets/DOTA_devkit"
result_dir="Result/ImplicitCorners"
# A folder named "Result" will be created in the current directory to store the evaluation results

# Array of model epochs to evaluate
epochs=(1 5 11)
cd ../..

# Create the result directory if it doesn't exist
mkdir -p "Result"
mkdir -p "$result_dir"

# Loop through the epochs and conf_thresh values
for epoch in "${epochs[@]}"; do
    for thresh in "${conf_thresh[@]}"; do
        model_path="${model_dir}/model_${epoch}.pth"
        echo "Running evaluation for model at epoch ${epoch} with conf_thresh=${thresh}..."
        
        python3 main.py --data_dir "$data_dir" --conf_thresh "$thresh" --batch_size "$batch_size" --dataset "$dataset" --phase "$phase" --resume "$model_path"
        
        # Run DOTA evaluation and save output
        echo "Running DOTA evaluation for model at epoch ${epoch}, conf_thresh=${thresh}..."
        (cd "$eval_dir" && python3 "$eval_script") | tee "$result_dir/evaluation_result_epoch_${epoch}_thresh_${thresh}.txt"
    done
done

