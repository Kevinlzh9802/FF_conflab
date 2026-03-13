#!/bin/bash
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=3:59:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --mem=48G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gcff/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gcff/slurm_%j.err # Set name of error log. %j is the Slurm jobId

set -euo pipefail

neon_path=/tudelft.net/staff-umbrella/neon
project_folder="$(pwd)"

sif_path=$neon_path/apptainer/ff_conflab.sif
config_path=$project_folder/python/configs/config_GCFF_cluster.yaml
data_root=$neon_path/zonghuan/data/conflab
gcff_root=$data_root/GCFF

mkdir -p "$gcff_root/panel_plots" "$gcff_root/logs" "$gcff_root/results"

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer
fi

echo "Starting GCFF Slurm job"
echo "Project: $project_folder"
echo "Container: $sif_path"
echo "Config: $config_path"
echo "Input data: $data_root/data.pkl"
echo "Panel output: $gcff_root/panel_plots"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

apptainer exec \
    --bind "$project_folder:$project_folder" \
    --bind "/tudelft.net:/tudelft.net" \
    "$sif_path" \
    bash -lc "
        set -euo pipefail
        cd '$project_folder/python'
        export PYTHONPATH='$project_folder/python:$project_folder/python/GCFF'
        export MPLBACKEND=Agg
        export PYTHONUNBUFFERED=1
        python3 GCFF/main_GCFF_new.py --config '$config_path'
    "
