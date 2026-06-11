#!/bin/bash
#SBATCH --job-name=gcff-vitpose
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --time=3:59:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gcff/vitpose_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gcff/vitpose_%j.err

set -euo pipefail

neon_path=/tudelft.net/staff-umbrella/neon
project_folder="$(pwd)"

sif_path=$neon_path/apptainer/ff_conflab.sif
data_root=$neon_path/zonghuan/data/conflab
gcff_root=$data_root/GCFF

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="gcff"
K="10"
PLOT="false"
CONFIG="${CONFIG:-$project_folder/python/configs/config_GCFF_cluster_vitpose.yaml}"
FINISHED="${FINISHED:-$gcff_root/data_vitpose_finished.pkl}"
SMOOTHED="${SMOOTHED:-$gcff_root/data_vitpose_finished_smoothed.pkl}"
RESULTS="${RESULTS:-$gcff_root/results}"
SP="${SP:-$data_root/sp_merged.pkl}"
PANEL_PLOTS="${PANEL_PLOTS:-$gcff_root/panel_plots_vitpose}"
PLOT_STEP="${PLOT_STEP:-120}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --mode=*)      MODE="${arg#--mode=}" ;;
        --k=*)         K="${arg#--k=}" ;;
        --plot=*)      PLOT="${arg#--plot=}" ;;
        --plot)        PLOT="true" ;;
        --config=*)    CONFIG="${arg#--config=}" ;;
        --finished=*)  FINISHED="${arg#--finished=}" ;;
        --smoothed=*)  SMOOTHED="${arg#--smoothed=}" ;;
        --results=*)   RESULTS="${arg#--results=}" ;;
        --sp=*)        SP="${arg#--sp=}" ;;
        --plot_dir=*)  PANEL_PLOTS="${arg#--plot_dir=}" ;;
        --plot_step=*) PLOT_STEP="${arg#--plot_step=}" ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 [--mode=gcff|smooth|analysis] [--k=5,10,20] [--plot]" >&2
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "gcff" && "$MODE" != "smooth" && "$MODE" != "analysis" ]]; then
    echo "Error: --mode must be gcff, smooth, or analysis (got: $MODE)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$gcff_root/logs" "$RESULTS" "$PANEL_PLOTS"

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer
fi

echo "Mode       : $MODE"
echo "Project    : $project_folder"
echo "Container  : $sif_path"
echo "Plot BEV   : $PLOT  (step=${PLOT_STEP}, dir=${PANEL_PLOTS})"
case "$MODE" in
    gcff)
        echo "Config     : $CONFIG"
        echo "Output     : $FINISHED"
        ;;
    smooth)
        echo "Input      : $FINISHED"
        echo "Output     : $SMOOTHED"
        echo "k values   : $K"
        ;;
    analysis)
        echo "Input      : $SMOOTHED"
        echo "Results    : $RESULTS"
        echo "k values   : $K"
        echo "SP path    : $SP"
        ;;
esac

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Build optional --plot flag string for Python scripts
PLOT_ARGS=""
if [[ "$PLOT" == "true" ]]; then
    PLOT_ARGS="--plot --finished '$FINISHED' --plot_dir '$PANEL_PLOTS' --plot_step '$PLOT_STEP'"
fi

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    gcff)
        # BEV plots in gcff mode are controlled by config.plots.panels (set to True in vitpose config)
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
                python3 GCFF/main_GCFF_new.py --config '$CONFIG'
            "
        ;;
    smooth)
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
                python3 GCFF/smooth_groupings.py \
                    --input '$FINISHED' \
                    --output '$SMOOTHED' \
                    --k '$K' \
                    $PLOT_ARGS
            "
        ;;
    analysis)
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
                python3 analysis/run_analysis.py \
                    --input '$SMOOTHED' \
                    --results_dir '$RESULTS' \
                    --k '$K' \
                    --sp '$SP' \
                    $PLOT_ARGS
            "
        ;;
esac

echo "Done: mode=$MODE"

# ---------------------------------------------------------------------------
# Notes — typical pipeline:
#
# 1) Convert ViTPose pkls to GCFF format:
#    sbatch ViTPose/slurm/vitpose_to_gcff_daic.sh
#    → data_vitpose.pkl
#
# 2) Run GCFF detection (writes {clue}Res columns, BEV plots via panels:True):
#    sbatch slurm/submit_gcff_vitpose.sh --mode=gcff
#    → data_vitpose_finished.pkl  +  panel_plots_vitpose/bev_*.png
#
# 3) Smooth per-frame groupings with sliding majority-vote window:
#    sbatch slurm/submit_gcff_vitpose.sh --mode=smooth --k=5,10,20
#    sbatch slurm/submit_gcff_vitpose.sh --mode=smooth --k=5,10,20 --plot
#    → data_vitpose_finished_smoothed.pkl  (columns: headRes_k5 etc.)
#    → optional: panel_plots_vitpose/bev_*.png (unsmoothed headRes)
#
# 4) Compute detection-change windows + homogeneity/split heatmaps:
#    sbatch slurm/submit_gcff_vitpose.sh --mode=analysis --k=5,10,20
#    sbatch slurm/submit_gcff_vitpose.sh --mode=analysis --k=5,10,20 --plot
#    → results/homogeneity_k{k}.png + results/split_k{k}.png
#    → optional: panel_plots_vitpose/bev_*.png (unsmoothed headRes)
#
# Override paths via env vars:
#    FINISHED=/custom/path.pkl  sbatch slurm/submit_gcff_vitpose.sh --mode=smooth
#    SP=/custom/sp.pkl          sbatch slurm/submit_gcff_vitpose.sh --mode=analysis --k=10
#    PANEL_PLOTS=/custom/plots/ sbatch slurm/submit_gcff_vitpose.sh --mode=gcff
#
# Speaking status (sp_merged.pkl):
#   Default SP path: .../conflab/sp_merged.pkl
#   Used in analysis mode to filter windows to those with ≥2 simultaneous speakers.
#   If the file does not exist, the filter is skipped and all windows are used.
# ---------------------------------------------------------------------------
