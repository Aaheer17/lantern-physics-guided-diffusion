#!/bin/bash
# Submit pcgrad experiments: low, mid, high, 100k × voxel, lap
# Usage: bash submit_pcgrad.sh

CONFIG_DIR="../configs/full_configs"

CONFIGS=(
    "weighted_sum_low_voxel.yaml"
    #"pcgrad_low_lap.yaml"
    "weighted_sum_mid_voxel.yaml"
    #"pcgrad_mid_lap.yaml"
    "weighted_sum_high_voxel.yaml"
    #"pcgrad_high_lap.yaml"
    #"pcgrad_100k_voxel.yaml"
    #"pcgrad_100k_lap.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG}"
    JOB_NAME="${CONFIG%.yaml}"

    echo "Submitting: ${JOB_NAME}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        run_job.slurm "${CONFIG_PATH}"
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
