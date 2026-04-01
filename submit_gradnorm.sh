#!/bin/bash
# Submit gradnorm experiments: low, mid, high × voxel, lap
# Usage: bash submit_gradnorm.sh
CONFIG_DIR="../configs/full_configs"
CONFIGS=(
    "gradnorm_low_voxel.yaml"
    "gradnorm_low_laplacian.yaml"
    "gradnorm_mid_voxel.yaml"
    "gradnorm_mid_laplacian.yaml"
    "gradnorm_high_voxel.yaml"
    "gradnorm_high_laplacian.yaml"
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
