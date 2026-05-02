#!/bin/bash

set -e

echo "========================================"
echo "Training baseline models (real + synthetic)"
echo "========================================"

# All datasets
dataset_names=('bloodmnist' 'pneumoniamnist' 'dermamnist' 'pathmnist' 'chestmnist' 'octmnist' 'tissuemnist' 'organamnist' 'organcmnist' 'organsmnist' 'breastmnist' 'retinamnist')

# For both real and synthetic we only use the test split
split_name="test"

# Mapping dataset name -> short code (same as original)
declare -A code_map
code_map["bloodmnist"]="bl"
code_map["pneumoniamnist"]="pn"
code_map["dermamnist"]="de"
code_map["pathmnist"]="pa"
code_map["chestmnist"]="ch"
code_map["octmnist"]="oc"
code_map["tissuemnist"]="ti"
code_map["organamnist"]="oa"
code_map["organcmnist"]="ocm"
code_map["organsmnist"]="osm"
code_map["breastmnist"]="br"
code_map["retinamnist"]="re"

# Split short code
split_code="tst"

# ------------------------------------------------------------
# 1. Real data (original MedMNIST)
# ------------------------------------------------------------
echo "Submitting real data jobs (test split only)..."

for dataset_name in "${dataset_names[@]}"; do
    dataset_code="${code_map[$dataset_name]:-uk}"
    job_name="${split_code}${dataset_code}base_real"
    log_file="Output/output_train_baseline_models_${split_name}_${dataset_name}_real.out"

    sbatch --job-name="$job_name" --output="$log_file" \
        scripts/slurm/train_baseline_models_and_estimate_hardness.sh \
        "$dataset_name" "$split_name"
done

# ------------------------------------------------------------
# 2. Synthetic data (four masking percentages)
# ------------------------------------------------------------
echo "Submitting synthetic data jobs (test split only, masking percentages 0.25, 0.50, 0.75, 1.00)..."

masking_percentages=(0.25 0.50 0.75 1.00)

for dataset_name in "${dataset_names[@]}"; do
    dataset_code="${code_map[$dataset_name]:-uk}"
    for masking in "${masking_percentages[@]}"; do
        # Format mask value for filename (e.g., mask0.25)
        mask_str=$(echo "$masking" | sed 's/\.//')  # removes dot -> 025, 050, 075, 100
        job_name="${split_code}${dataset_code}base_mask${mask_str}"
        log_file="Output/output_train_baseline_models_${split_name}_${dataset_name}_mask${mask_str}.out"

        sbatch --job-name="$job_name" --output="$log_file" \
            scripts/slurm/train_baseline_models_and_estimate_hardness.sh \
            "$dataset_name" "$split_name" "--synthetic" "--masking_percentage" "$masking"
    done
done

echo "========================================"
echo "All jobs submitted."
echo "========================================"