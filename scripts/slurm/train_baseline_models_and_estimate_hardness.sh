#!/bin/bash
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load the modules required by our program
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

dataset_name=$1
split_name=$2

# Remove the first two arguments, the rest are passed to the Python script
shift 2

# Call the Python script with all remaining arguments (e.g., --synthetic --masking_percentage 0.50)
python3 -m src.experiments.train_baseline_models_and_estimate_hardness \
    --dataset_name "$dataset_name" --split "$split_name" "$@"