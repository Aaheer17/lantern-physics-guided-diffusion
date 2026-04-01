#!/bin/bash
# Submit config experiments: low, mid, high, 100k × voxel, lap
# Usage: bash submit_config.sh

CONFIG_DIR="../configs/full_configs"

CONFIGS=(
    #"config_low_voxel.yaml"
    #"config_low_lap.yaml"
    #"config_mid_voxel.yaml"
    #"config_mid_lap.yaml"
    "config_high_voxel.yaml"
    #"config_high_lap.yaml"
    #"config_100k_voxel.yaml"
   # "config_100k_lap.yaml"
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
