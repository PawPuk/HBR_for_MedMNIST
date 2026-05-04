#!/bin/bash
set -e

dataset_names=(
    'bloodmnist'
    'pneumoniamnist'
    'dermamnist'
    'pathmnist'
    'chestmnist'
    'octmnist'
    'tissuemnist'
    'organamnist'
    'organcmnist'
    'organsmnist'
    'breastmnist'
    'retinamnist'
)

# Mapping dataset name -> short code
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

# Split name (only test split is used)
split_name="test"
split_code="tst"

# Synthetic masking percentages
masking_percentages=(0.25 0.50 0.75 1.00)


echo "========================================"
echo "Submitting synthetic data jobs (test split only)"
echo "Masking percentages: ${masking_percentages[*]}"
echo "========================================"

for dataset_name in "${dataset_names[@]}"; do
    dataset_code="${code_map[$dataset_name]:-uk}"
    for masking in "${masking_percentages[@]}"; do
        # Format mask value for filename (remove dot: 0.25 -> 025)
        mask_str=$(echo "$masking" | sed 's/\.//')
        job_name="${split_code}${dataset_code}${mask_str}"
        log_file="Output/output_train_baseline_models_${split_name}_${dataset_name}_mask${mask_str}.out"

        sbatch --job-name="$job_name" --output="$log_file" \
            scripts/slurm/estimate_hardness.sh \
            "$dataset_name" "$split_name" --synthetic --masking_percentage "$masking"
    done
done

echo "All synthetic data jobs submitted."