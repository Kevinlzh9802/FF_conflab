#!/bin/bash
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=3:59:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --mem=48G
#SBATCH --chdir=/home/zonghuan/tudelft/projects/FF_conflab
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gcff/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gcff/slurm_%j.err # Set name of error log. %j is the Slurm jobId

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
