#!/bin/bash
# Submit imtlg experiments: low, mid, high, 100k × voxel, lap
# Usage: bash submit_imtlg.sh

CONFIG_DIR="../configs/full_configs"

CONFIGS=(
    #"imtlg_low_voxel.yaml"
    #"imtlg_low_lap.yaml"
    #"imtlg_mid_voxel.yaml"
    #"imtlg_mid_lap.yaml"
    "imtlg_high_voxel.yaml"
    #"imtlg_high_lap.yaml"
    #"imtlg_100k_voxel.yaml"
    #"imtlg_100k_lap.yaml"
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
