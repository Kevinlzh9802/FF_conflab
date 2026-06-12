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
CONFIG="${CONFIG:-$project_folder/python/configs/config_GCFF_cluster.yaml}"
CLUE=""
DETECTION_DIR="${DETECTION_DIR:-$gcff_root/results/detection}"
DATA="${DATA:-$gcff_root/data.pkl}"
FINISHED="${FINISHED:-$gcff_root/data_finished.pkl}"
SMOOTHED="${SMOOTHED:-$gcff_root/data_finished_smoothed.pkl}"
RESULTS="${RESULTS:-$gcff_root/results}"
SP="${SP:-$data_root/sp_merged.pkl}"
PANEL_PLOTS="${PANEL_PLOTS:-$gcff_root/plots/bev}"
PLOT_STEP="${PLOT_STEP:-120}"
OVERWRITE_FINISHED="true"
OVERWRITE_SMOOTHED="true"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --mode=*)               MODE="${arg#--mode=}" ;;
        --k=*)                  K="${arg#--k=}" ;;
        --plot=*)               PLOT="${arg#--plot=}" ;;
        --plot)                 PLOT="true" ;;
        --config=*)             CONFIG="${arg#--config=}" ;;
        --clue=*)               CLUE="${arg#--clue=}" ;;
        --detection-dir=*)      DETECTION_DIR="${arg#--detection-dir=}" ;;
        --data=*)               DATA="${arg#--data=}" ;;
        --finished=*)           FINISHED="${arg#--finished=}" ;;
        --smoothed=*)           SMOOTHED="${arg#--smoothed=}" ;;
        --results=*)            RESULTS="${arg#--results=}" ;;
        --sp=*)                 SP="${arg#--sp=}" ;;
        --plot_dir=*)           PANEL_PLOTS="${arg#--plot_dir=}" ;;
        --plot_step=*)          PLOT_STEP="${arg#--plot_step=}" ;;
        --overwrite-finished=*) OVERWRITE_FINISHED="${arg#--overwrite-finished=}" ;;
        --overwrite-smoothed=*) OVERWRITE_SMOOTHED="${arg#--overwrite-smoothed=}" ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 [--mode=gcff|smooth|analysis] [--clue=head|shoulder|hip|foot] [--k=5,10,20] [--plot]" >&2
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
mkdir -p "$gcff_root/logs" "$RESULTS" "$PANEL_PLOTS" "$DETECTION_DIR"

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
        echo "Clue       : ${CLUE:-all}"
        echo "DetectDir  : $DETECTION_DIR"
        ;;
    smooth)
        echo "Data       : $DATA"
        echo "DetectDir  : $DETECTION_DIR"
        echo "Finished   : $FINISHED  (overwrite=${OVERWRITE_FINISHED})"
        echo "Smoothed   : $SMOOTHED  (overwrite=${OVERWRITE_SMOOTHED})"
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
    PLOT_ARGS="--plot --plot_dir '$PANEL_PLOTS' --plot_step '$PLOT_STEP'"
fi

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    gcff)
        # Per-clue mode: pass --clue and --detection-dir to enable parallel jobs.
        # All-clues mode (no --clue): runs all 4 clues sequentially, writes data_finished.pkl.
        CLUE_ARG=""
        [[ -n "$CLUE" ]] && CLUE_ARG="--clue '$CLUE' --detection-dir '$DETECTION_DIR'"
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
                python3 GCFF/main_GCFF_new.py --config '$CONFIG' $CLUE_ARG
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
                    --data '$DATA' \
                    --detection-dir '$DETECTION_DIR' \
                    --input '$FINISHED' \
                    --output '$SMOOTHED' \
                    --k '$K' \
                    --overwrite-finished '$OVERWRITE_FINISHED' \
                    --overwrite-smoothed '$OVERWRITE_SMOOTHED' \
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
#    → data.pkl
#
# 2a) Run GCFF detection — 4 parallel jobs, one per body clue:
#    sbatch slurm/submit_gcff_vitpose.sh --mode=gcff --clue=head
#    sbatch slurm/submit_gcff_vitpose.sh --mode=gcff --clue=shoulder
#    sbatch slurm/submit_gcff_vitpose.sh --mode=gcff --clue=hip
#    sbatch slurm/submit_gcff_vitpose.sh --mode=gcff --clue=foot
#    → results/detection/{head,shoulder,hip,foot}.pkl
#
#    (Alternatively, run all clues in one job — slower but no merge needed:
#     sbatch slurm/submit_gcff_vitpose.sh --mode=gcff
#     → data_finished.pkl directly)
#
# 2b) Merge per-clue detections + smooth (after all 4 jobs in 2a complete):
#    sbatch slurm/submit_gcff_vitpose.sh --mode=smooth --k=5,10,20
#    sbatch slurm/submit_gcff_vitpose.sh --mode=smooth --k=5,10,20 --plot
#    → data_finished.pkl           (merged, all 4 clueRes columns)
#    → data_finished_smoothed.pkl  (columns: headRes_k5 etc.)
#    → optional: plots/bev/<batch>/*.png + plots/spectrum/<batch>.png
#
# 3) Compute detection-change windows + homogeneity/split heatmaps:
#    sbatch slurm/submit_gcff_vitpose.sh --mode=analysis --k=5,10,20
#    → results/homogeneity_k{k}.png + results/split_k{k}.png
#
# Overwrite control (smooth mode):
#    --overwrite-finished=false   skip merge if data_finished.pkl already exists
#    --overwrite-smoothed=false   skip smooth if data_finished_smoothed.pkl already exists
#
# Override paths via env vars:
#    DATA=/custom/data.pkl         sbatch slurm/submit_gcff_vitpose.sh --mode=smooth
#    DETECTION_DIR=/custom/det/    sbatch slurm/submit_gcff_vitpose.sh --mode=gcff --clue=head
#    SP=/custom/sp.pkl             sbatch slurm/submit_gcff_vitpose.sh --mode=analysis --k=10
#
# Speaking status (sp_merged.pkl):
#   Default SP path: .../conflab/sp_merged.pkl
#   Used in analysis mode to filter windows to those with ≥2 simultaneous speakers.
#   If the file does not exist, the filter is skipped and all windows are used.
# ---------------------------------------------------------------------------
