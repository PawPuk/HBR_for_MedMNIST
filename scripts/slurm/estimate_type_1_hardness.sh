#!/bin/bash
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load required modules
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

dataset_name=$1
split_name=$2

# Shift away the first two arguments; the rest are passed to Python
shift 2

# Run the estimation script
python3 -m src.experiments.estimate_hardness_via_learning_dynamics \
    --dataset_name "$dataset_name" \
    --split "$split_name" \
    --data_type "real" \
    --save_models
    "$@"