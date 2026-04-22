#!/bin/bash

set -e

dataset_names=('bloodmnist' 'pneumoniamnist' 'dermamnist' 'pathmnist' 'chestmnist' 'octmnist' 'tissuemnist' 'organamnist' 'organcmnist' 'organsmnist' 'breastmnist' 'retinamnist')

for dataset_name in "${dataset_names[@]}"
do
    case "$dataset_name" in
        "bloodmnist")
            dataset_code="bl"
            ;;
        "pneumoniamnist")
            dataset_code="pn"
            ;;
        "dermamnist")
            dataset_code="de"
            ;;
        "pathmnist")
            dataset_code="pa"
            ;;
        "chestmnist")
            dataset_code="ch"
            ;;
        "octmnist")
            dataset_code="oc"
            ;;
        "tissuemnist")
            dataset_code="ti"
            ;;
        "organamnist")
            dataset_code="oa"
            ;;
        "organcmnist")
            dataset_code="ocm"
            ;;
        "organsmnist")
            dataset_code="osm"
            ;;
        "breastmnist")
            dataset_code="br"
            ;;
        "retinamnist")
            dataset_code="re"
            ;;
        *)
            dataset_code="uk"
            ;;
    esac

    job_name="${dataset_code}base"
    log_file="Output/output_class_level_evaluation_${dataset_name}.out"

    sbatch --job-name="$job_name" --output="$log_file" \
      scripts/slurm/class_level_evaluation.sh "$dataset_name"
done
