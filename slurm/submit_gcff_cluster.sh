#!/bin/bash
#SBATCH --job-name=gcff-main
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/ff_conflab.sif"
CONFIG_PATH="${REPO_ROOT}/python/configs/config_GCFF_cluster.yaml"
DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/conflab"
GCFF_ROOT="${DATA_ROOT}/GCFF"

mkdir -p "${REPO_ROOT}/slurm/logs"
mkdir -p "${GCFF_ROOT}/panel_plots" "${GCFF_ROOT}/logs" "${GCFF_ROOT}/results"

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer
fi

echo "Starting GCFF Slurm job"
echo "Repository: ${REPO_ROOT}"
echo "Container: ${SIF_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Input data: ${DATA_ROOT}/data.pkl"
echo "Panel output: ${GCFF_ROOT}/panel_plots"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

apptainer exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "/tudelft.net:/tudelft.net" \
    "${SIF_PATH}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO_ROOT}/python'
        export PYTHONPATH='${REPO_ROOT}/python:${REPO_ROOT}/python/GCFF'
        export MPLBACKEND=Agg
        export PYTHONUNBUFFERED=1
        python3 GCFF/main_GCFF_new.py --config '${CONFIG_PATH}'
    "
