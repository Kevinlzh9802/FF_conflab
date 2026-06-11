"""
smooth_groupings.py

Apply a sliding-window majority-vote (mode) filter to GCFF per-frame grouping results.

For the n-th frame within a (Cam, Vid) sequence, the smoothed result is the most
frequently occurring grouping in the half-open window [n-k, n+k) of 2k frames.
Edge frames are handled by replicating the first/last value (edge padding).

New columns are written for each k value: headRes_k{k}, shoulderRes_k{k},
hipRes_k{k}, footRes_k{k}.  All existing columns are preserved.

Usage
-----
# Default k=10, read data_vitpose_finished.pkl, write data_vitpose_finished_smoothed.pkl:
python GCFF/smooth_groupings.py

# Multiple k values:
python GCFF/smooth_groupings.py --k=5,10,20

# Explicit paths:
python GCFF/smooth_groupings.py \\
    --input  /path/to/data_vitpose_finished.pkl \\
    --output /path/to/data_vitpose_finished_smoothed.pkl \\
    --k 5,10,20
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

NEON = "/tudelft.net/staff-umbrella/neon"
DEFAULT_INPUT = f"{NEON}/zonghuan/data/conflab/GCFF/data_vitpose_finished.pkl"
DEFAULT_OUTPUT = f"{NEON}/zonghuan/data/conflab/GCFF/data_vitpose_finished_smoothed.pkl"
DEFAULT_PANEL_PLOTS = f"{NEON}/zonghuan/data/conflab/GCFF/panel_plots_vitpose"

CLUE_COLS = ["headRes", "shoulderRes", "hipRes", "footRes"]

EMPTY_CANONICAL = "__empty__"


# ---------------------------------------------------------------------------
# Canonical representation for majority vote
# ---------------------------------------------------------------------------

def _canonical(groups) -> str:
    """Stable string key for a grouping (list of lists of ints)."""
    if not groups:
        return EMPTY_CANONICAL
    try:
        return str(sorted([sorted(int(p) for p in g) for g in groups if g]))
    except Exception:
        return EMPTY_CANONICAL


def _mode_grouping(window: list):
    """Return the most common grouping in a window of group-list values."""
    non_empty = [g for g in window if _canonical(g) != EMPTY_CANONICAL]
    if not non_empty:
        return []
    counts = Counter(_canonical(g) for g in non_empty)
    mode_canon = counts.most_common(1)[0][0]
    for g in non_empty:
        if _canonical(g) == mode_canon:
            return g
    return []


def _smooth_sequence(seq: list, k: int) -> list:
    """Apply edge-padded sliding majority vote with window [n-k, n+k) for each n."""
    n = len(seq)
    if n == 0:
        return seq
    # Pad k copies of first/last element; window for frame n is padded[n : n+2k]
    padded = [seq[0]] * k + seq + [seq[-1]] * k
    return [_mode_grouping(padded[i : i + 2 * k]) for i in range(n)]


# ---------------------------------------------------------------------------
# Main smoothing logic
# ---------------------------------------------------------------------------

def smooth(df: pd.DataFrame, ks: List[int]) -> pd.DataFrame:
    """Add smoothed columns {clue}Res_k{k} for each k in ks.

    Smoothing is applied independently per (Cam, Vid) group, sorted by concat_ts.
    Rows not belonging to any (Cam, Vid) group (concat_ts = NaN) are left as [].
    """
    df = df.copy()

    # Initialise new columns with empty lists
    for k in ks:
        for clue in CLUE_COLS:
            col = f"{clue}_k{k}"
            df[col] = [[] for _ in range(len(df))]

    groups = df.groupby(["Cam", "Vid"])
    for (cam, vid), grp_idx in groups.groups.items():
        sub = df.loc[grp_idx].sort_values("concat_ts")
        row_order = sub.index.tolist()

        for clue in CLUE_COLS:
            raw_seq = sub[clue].tolist()
            for k in ks:
                smoothed = _smooth_sequence(raw_seq, k)
                col = f"{clue}_k{k}"
                for idx, val in zip(row_order, smoothed):
                    df.at[idx, col] = val

        print(
            f"  Cam={cam} Vid={vid}: smoothed {len(row_order)} rows "
            f"with k={ks}"
        )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth GCFF per-frame grouping results with a sliding majority-vote filter."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input pkl (data_vitpose_finished.pkl). Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output pkl (data_vitpose_finished_smoothed.pkl). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--k",
        default="10",
        metavar="K_VALUES",
        help="Comma-separated half-window sizes, e.g. 5,10,20 (default: 10).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate BEV panel plots after smoothing (reads --finished for unsmoothed results).",
    )
    parser.add_argument(
        "--finished",
        default=DEFAULT_INPUT,
        help=f"Path to data_vitpose_finished.pkl used for BEV plotting. Default: {DEFAULT_INPUT}",
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

    ks = [int(v.strip()) for v in args.k.split(",") if v.strip()]
    if not ks:
        raise SystemExit("--k must be a non-empty comma-separated list of integers.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading: {input_path}")
    df = pd.read_pickle(input_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    print(f"\nSmoothing with k={ks} ...")
    df_smoothed = smooth(df, ks)

    new_cols = [f"{clue}_k{k}" for k in ks for clue in CLUE_COLS]
    print(f"\nNew columns added: {new_cols}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_smoothed.to_pickle(output_path)
    print(f"\nSaved: {output_path}  ({len(df_smoothed)} rows)")

    if args.plot:
        finished_path = Path(args.finished)
        print(f"\nBEV plots: loading unsmoothed results from {finished_path}")
        finished_df = pd.read_pickle(finished_path)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.plot_spacefeat import plot_spacefeat_bev_panels_df
        plot_spacefeat_bev_panels_df(finished_df, output_dir=args.plot_dir, frame_step=args.plot_step)
