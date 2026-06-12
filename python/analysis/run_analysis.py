"""
run_analysis.py

Run detection-change-event windowing and spatial homogeneity/split score analysis
on k-smoothed GCFF grouping results from data_finished_smoothed.pkl.

For each k value the script:
  1. Detects windows bounded by grouping changes in the smoothed {clue}Res_k{k} columns
  2. Optionally filters windows by simultaneous speaker count (requires --sp path)
  3. Computes per-window average homogeneity and split scores across clue pairs
  4. Saves heatmaps: homogeneity_k{k}.png and split_k{k}.png under --results_dir

Usage
-----
# Default k=10, no speaking filter:
python analysis/run_analysis.py

# With speaking status and multiple k values:
python analysis/run_analysis.py --k=5,10,20 \\
    --sp /tudelft.net/staff-umbrella/neon/zonghuan/data/conflab/sp_merged.pkl

# With BEV panel plots (reads unsmoothed finished pkl):
python analysis/run_analysis.py --k=10 --plot \\
    --finished /path/to/data_finished.pkl \\
    --plot_dir /path/to/plots/bev/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

NEON = "/tudelft.net/staff-umbrella/neon"
DEFAULT_INPUT = f"{NEON}/zonghuan/data/conflab/GCFF/data_finished_smoothed.pkl"
DEFAULT_RESULTS = f"{NEON}/zonghuan/data/conflab/GCFF/results"
DEFAULT_FINISHED = f"{NEON}/zonghuan/data/conflab/GCFF/data_finished.pkl"
DEFAULT_PANEL_PLOTS = f"{NEON}/zonghuan/data/conflab/GCFF/plots/bev"

# Add the python directory to path so analysis/spatial imports work
_HERE = Path(__file__).resolve().parent
_PYTHON_ROOT = _HERE.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from analysis.cross_modal import detect_group_num_breakpoints
from analysis.spatial import spatial_scores_df


# ---------------------------------------------------------------------------
# Per-k analysis
# ---------------------------------------------------------------------------

def run_for_k(
    data: pd.DataFrame,
    k: int,
    results_dir: Path,
    sp_path: Optional[str] = None,
) -> None:
    clue_cols = [f"headRes_k{k}", f"shoulderRes_k{k}", f"hipRes_k{k}", f"footRes_k{k}"]

    missing = [c for c in clue_cols if c not in data.columns]
    if missing:
        print(f"  [k={k}] WARNING: columns missing in input pkl: {missing}. Skipping.")
        return

    print(f"\n[k={k}] Detecting grouping-change windows using columns: {clue_cols}")
    windows = detect_group_num_breakpoints(data, clues=clue_cols)

    if windows.empty:
        print(f"  [k={k}] No windows found. Skipping spatial scores.")
        return

    lengths = windows["length"].dropna()
    print(
        f"  [k={k}] Windows (raw): {len(windows)}  "
        f"length mean={lengths.mean():.1f}  median={lengths.median():.1f}  "
        f"min={lengths.min():.0f}  max={lengths.max():.0f}"
    )

    # Speaking-based filtering (requires sp_merged.pkl)
    if sp_path and Path(sp_path).is_file():
        from analysis.cross_modal import count_speaker_groups, filter_windows
        print(f"  [k={k}] Applying speaking filter from: {sp_path}")
        windows = count_speaker_groups(windows, speaking_status_path=sp_path)
        windows = filter_windows(windows)
        print(f"  [k={k}] After speaking filter: {len(windows)} windows")
    else:
        if sp_path:
            print(f"  [k={k}] WARNING: sp file not found at '{sp_path}'. Speaking filter skipped.")
        else:
            print(f"  [k={k}] Note: speaking filter skipped (--sp not provided).")

    if windows.empty:
        print(f"  [k={k}] No windows after filtering. Skipping spatial scores.")
        return

    print(f"  [k={k}] Computing spatial scores ...")
    result = spatial_scores_df(windows, feature_cols=clue_cols)
    if result is None or (isinstance(result, tuple) and len(result) == 2 and result[0] is None):
        print(f"  [k={k}] spatial_scores_df returned no figures (all-NaN). Skipping save.")
        return

    fig_h, fig_s = result

    hom_path = results_dir / f"homogeneity_k{k}.png"
    split_path = results_dir / f"split_k{k}.png"
    fig_h.savefig(hom_path, dpi=150, bbox_inches="tight")
    fig_s.savefig(split_path, dpi=150, bbox_inches="tight")
    print(f"  [k={k}] Saved: {hom_path}")
    print(f"  [k={k}] Saved: {split_path}")

    try:
        import matplotlib.pyplot as plt
        plt.close(fig_h)
        plt.close(fig_s)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run detection-change-event windowing + spatial homogeneity/split analysis "
            "on k-smoothed GCFF grouping results."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Smoothed pkl path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_RESULTS,
        help=f"Directory for output heatmap PNGs. Default: {DEFAULT_RESULTS}",
    )
    parser.add_argument(
        "--k",
        default="10",
        metavar="K_VALUES",
        help="Comma-separated k values matching those used in smooth_groupings.py (default: 10).",
    )
    parser.add_argument(
        "--sp",
        default=None,
        metavar="SP_PATH",
        help=(
            "Path to sp_merged.pkl for speaking-based window filtering. "
            "If omitted, all detected windows are used without filtering."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate BEV panel plots after analysis (reads --finished for unsmoothed results).",
    )
    parser.add_argument(
        "--finished",
        default=DEFAULT_FINISHED,
        help=f"Path to data_finished.pkl used for BEV plotting. Default: {DEFAULT_FINISHED}",
    )
    parser.add_argument(
        "--plot_dir",
        default=DEFAULT_PANEL_PLOTS,
        help=f"Directory for BEV PNG output. Default: {DEFAULT_PANEL_PLOTS}",
    )
    parser.add_argument(
        "--plot_step",
        type=int,
        default=120,
        help="Save one BEV figure every N rows (default: 120).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ks: List[int] = [int(v.strip()) for v in args.k.split(",") if v.strip()]
    if not ks:
        raise SystemExit("--k must be a non-empty comma-separated list of integers.")

    input_path = Path(args.input)
    results_dir = Path(args.results_dir)

    print(f"Loading: {input_path}")
    data = pd.read_pickle(input_path)
    print(f"  {len(data)} rows, columns: {list(data.columns)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir : {results_dir}")
    print(f"k values    : {ks}")
    print(f"sp path     : {args.sp or '(none — speaking filter disabled)'}")

    for k in ks:
        run_for_k(data, k, results_dir, sp_path=args.sp)

    if args.plot:
        finished_path = Path(args.finished)
        print(f"\nBEV plots: loading unsmoothed results from {finished_path}")
        finished_df = pd.read_pickle(finished_path)
        from utils.plot_spacefeat import plot_spacefeat_bev_panels_df
        plot_spacefeat_bev_panels_df(finished_df, output_dir=args.plot_dir, frame_step=args.plot_step)

    print("\nDone.")
