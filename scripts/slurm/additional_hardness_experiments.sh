#!/bin/bash
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Load required modules
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

dataset_name=$1

# Run the estimation script
python3 -m src.experiments.estimate_hardness_via_margins --dataset_name "$dataset_name"