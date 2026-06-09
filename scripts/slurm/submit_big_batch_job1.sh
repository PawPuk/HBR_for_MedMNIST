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

# Split name (only val split is used)
split_name="val"
split_code="val"

echo "========================================"
echo "Submitting real data jobs (val split only)"
echo "========================================"

for dataset_name in "${dataset_names[@]}"; do
    dataset_code="${code_map[$dataset_name]:-uk}"
    job_name="${split_code}${dataset_code}"
    log_file="Output/output_train_baseline_models_${split_name}_${dataset_name}_real.out"

    sbatch --job-name="$job_name" --output="$log_file" \
        scripts/slurm/estimate_hardness.sh \
        "$dataset_name" "$split_name"
done

echo "All real data jobs submitted."