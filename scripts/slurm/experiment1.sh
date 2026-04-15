#!/bin/bash

set -e

echo "========================================"
echo "Training baseline models"
echo "========================================"

echo "Training models on balanced dataset (the baseline)..."

dataset_names=('bloodmnist' 'pneumoniamnist')

for dataset_name in "${dataset_names[@]}"
do
    if [ "$dataset_name" == "bloodmnist" ]; then
        dataset_code="bl"
    else
        dataset_code="pn"
    fi

    job_name="${dataset_code}base"
    log_file="Output/output_train_baseline_models_${dataset_name}.out"

    sbatch --job-name="$job_name" --output="$log_file" \
      scripts/slurm/train_baselin_models_and_estimate_hardness.sh "$dataset_name"
done

echo "========================================"
echo "Case Study 1 experiments completed."
echo "========================================"

