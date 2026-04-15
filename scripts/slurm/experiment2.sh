#!/bin/bash

set -e

dataset_names=('bloodmnist' 'pneumoniamnist')

for dataset_name in "${dataset_names[@]}"
do
    if [ "$dataset_name" == "bloodmnist" ]; then
        dataset_code="bl"
    else
        dataset_code="pn"
    fi

    job_name="${dataset_code}base"
    log_file="Output/output_class_level_evaluation_${dataset_name}.out"

    sbatch --job-name="$job_name" --output="$log_file" \
      scripts/slurm/class_level_evaluation.sh "$dataset_name"
done
