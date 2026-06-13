"""
smooth_groupings.py

Merge per-clue GCFF detection results into data_finished.pkl, then apply a
sliding-window majority-vote (mode) filter to produce data_finished_smoothed.pkl.

Step 1 — Merge
  Loads data.pkl and the 4 per-clue detection pkls from --detection-dir, joins
  them on (Cam, Vid, Seg, Timestamp), and writes data_finished.pkl.

Step 2 — Smooth
  For each k value adds {clue}Res_k{k} columns using a sliding majority-vote
  with half-window k.  The smoothed DataFrame is written to --output.

Usage
-----
# Default paths, k=10:
python GCFF/smooth_groupings.py

# Multiple k values:
python GCFF/smooth_groupings.py --k=5,10,20

# Skip overwriting existing files:
python GCFF/smooth_groupings.py --k=10 \\
    --overwrite-finished=false --overwrite-smoothed=false

# With BEV + spectrum plots:
python GCFF/smooth_groupings.py --k=10 --plot
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEON = "/tudelft.net/staff-umbrella/neon"
DEFAULT_DATA          = f"{NEON}/zonghuan/data/conflab/GCFF/data.pkl"
DEFAULT_INPUT         = f"{NEON}/zonghuan/data/conflab/GCFF/data_finished.pkl"
DEFAULT_OUTPUT        = f"{NEON}/zonghuan/data/conflab/GCFF/data_finished_smoothed.pkl"
DEFAULT_DETECTION_DIR = f"{NEON}/zonghuan/data/conflab/GCFF/results/detection"
DEFAULT_PANEL_PLOTS   = f"{NEON}/zonghuan/data/conflab/GCFF/plots/bev"
DEFAULT_SPECTRUM_PLOTS = f"{NEON}/zonghuan/data/conflab/GCFF/plots/spectrum"

CLUE_COLS = ["headRes", "shoulderRes", "hipRes", "footRes"]
CLUES     = ["head", "shoulder", "hip", "foot"]

EMPTY_CANONICAL = "__empty__"


# ---------------------------------------------------------------------------
# Merge per-clue detection pkls into data_finished.pkl
# ---------------------------------------------------------------------------

def merge_detections(
    data_pkl: Path,
    detection_dir: Path,
    finished_pkl: Path,
    overwrite: bool,
) -> pd.DataFrame:
    """Load data.pkl + 4 per-clue detection pkls and write data_finished.pkl.

    If finished_pkl already exists and overwrite=False, loads and returns it
    without redoing the merge.
    """
    if finished_pkl.exists() and not overwrite:
        print(f"Skipping merge: {finished_pkl} exists (--overwrite-finished=false)")
        return pd.read_pickle(finished_pkl)

    print(f"Merging detections from {detection_dir}")
    print(f"  Base data: {data_pkl}")
    base = pd.read_pickle(data_pkl)
    print(f"  {len(base)} rows loaded.")

    MERGE_KEYS = ["Cam", "Vid", "Seg", "Timestamp"]
    for clue in CLUES:
        det_path = detection_dir / f"{clue}.pkl"
        if not det_path.exists():
            raise FileNotFoundError(
                f"Detection pkl not found: {det_path}\n"
                f"Run: sbatch slurm/submit_gcff_vitpose.sh --mode=gcff --clue={clue}"
            )
        det = pd.read_pickle(det_path)
        col = f"{clue}Res"
        base = base.merge(det[MERGE_KEYS + [col]], on=MERGE_KEYS, how="left")
        base[col] = base[col].apply(lambda x: x if isinstance(x, list) else [])
        non_empty = base[col].apply(bool).sum()
        print(f"  [{clue}] {non_empty}/{len(base)} non-empty frames")

    finished_pkl.parent.mkdir(parents=True, exist_ok=True)
    base.to_pickle(finished_pkl)
    print(f"Saved: {finished_pkl}  ({len(base)} rows)")
    return base


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _canonical(groups) -> str:
    if not groups:
        return EMPTY_CANONICAL
    try:
        return str(sorted([sorted(int(p) for p in g) for g in groups if g]))
    except Exception:
        return EMPTY_CANONICAL


def _mode_grouping(window: list):
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
    """Edge-padded sliding majority vote with half-window k."""
    n = len(seq)
    if n == 0:
        return seq
    padded = [seq[0]] * k + seq + [seq[-1]] * k
    return [_mode_grouping(padded[i : i + 2 * k]) for i in range(n)]


def smooth(df: pd.DataFrame, ks: List[int]) -> pd.DataFrame:
    """Add {clue}Res_k{k} columns via sliding majority-vote per (Cam, Vid) group."""
    df = df.copy()
    for k in ks:
        for clue in CLUE_COLS:
            df[f"{clue}_k{k}"] = [[] for _ in range(len(df))]

    for (cam, vid), grp_idx in df.groupby(["Cam", "Vid"]).groups.items():
        sub = df.loc[grp_idx].sort_values("concat_ts")
        row_order = sub.index.tolist()
        for clue in CLUE_COLS:
            raw_seq = sub[clue].tolist()
            for k in ks:
                smoothed = _smooth_sequence(raw_seq, k)
                col = f"{clue}_k{k}"
                for idx, val in zip(row_order, smoothed):
                    df.at[idx, col] = val
        print(f"  Cam={cam} Vid={vid}: smoothed {len(row_order)} rows with k={ks}")

    return df


# ---------------------------------------------------------------------------
# Spectrum plots (mirrors main_GCFF_new._plot_spectrum_per_batch)
# ---------------------------------------------------------------------------

def _collect_all_person_ids(batch_df: pd.DataFrame) -> List[int]:
    all_ids: set = set()
    for _, row in batch_df.iterrows():
        sf = row.get("spaceFeat") or {}
        if not isinstance(sf, dict):
            continue
        for arr in sf.values():
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 1:
                for pid in a[:, 0]:
                    try:
                        all_ids.add(int(pid))
                    except (ValueError, TypeError):
                        pass
    return sorted(all_ids)


def _plot_spectrum_per_batch(
    data_kp: pd.DataFrame,
    spectrum_dir: Path,
    col_suffix: str = "",
) -> None:
    from tests.group_spectrum import plot_target_grouping_spectrum
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    for (cam, vid, seg), batch_df in data_kp.groupby(["Cam", "Vid", "Seg"]):
        batch_num = f"{int(cam)}{int(vid)}{int(seg)}"
        save_path = spectrum_dir / f"{batch_num}.png"
        target_ids = _collect_all_person_ids(batch_df)
        if not target_ids:
            print(f"  Spectrum [{batch_num}]: no person IDs found, skipping.")
            continue
        try:
            plot_target_grouping_spectrum(
                data_kp=batch_df.reset_index(drop=True),
                target_ids=target_ids,
                save_path=save_path,
                show=False,
                col_suffix=col_suffix,
            )
            plt.close("all")
            print(f"  Spectrum [{batch_num}]: saved {save_path}")
        except Exception as exc:
            print(f"  Spectrum [{batch_num}]: failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    def _bool(v: str) -> bool:
        return v.lower() not in ("false", "0", "no")

    parser = argparse.ArgumentParser(
        description="Merge per-clue GCFF detection pkls and smooth with majority-vote filter."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Path to data.pkl used for merge base. Default: {DEFAULT_DATA}",
    )
    parser.add_argument(
        "--detection-dir",
        default=DEFAULT_DETECTION_DIR,
        help=f"Directory containing per-clue detection pkls. Default: {DEFAULT_DETECTION_DIR}",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to data_finished.pkl (output of merge / input to smooth). Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output pkl (data_finished_smoothed.pkl). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--k",
        default="10",
        metavar="K_VALUES",
        help="Comma-separated half-window sizes, e.g. 5,10,20 (default: 10).",
    )
    parser.add_argument(
        "--overwrite-finished",
        type=_bool,
        default=True,
        metavar="BOOL",
        help="Overwrite data_finished.pkl if it exists (default: true).",
    )
    parser.add_argument(
        "--overwrite-smoothed",
        type=_bool,
        default=True,
        metavar="BOOL",
        help="Overwrite data_finished_smoothed.pkl if it exists (default: true).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate BEV panel plots and spectrum plots from data_finished.pkl.",
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
    parser.add_argument(
        "--spectrum_dir",
        default=DEFAULT_SPECTRUM_PLOTS,
        help=f"Root directory for spectrum PNG output. Subdirs original/, k{{k}}/ are created inside. Default: {DEFAULT_SPECTRUM_PLOTS}",
    )
    parser.add_argument(
        "--frames_root",
        default=f"{NEON}/zonghuan/data/conflab/bbox_kp",
        help=(
            "Root containing {batch}/images/{frame_id:08d}.jpg for BEV pixel overlay. "
            f"Default: {NEON}/zonghuan/data/conflab/bbox_kp"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ks = [int(v.strip()) for v in args.k.split(",") if v.strip()]
    if not ks:
        raise SystemExit("--k must be a non-empty comma-separated list of integers.")

    data_pkl      = Path(args.data)
    detection_dir = Path(args.detection_dir)
    finished_pkl  = Path(args.input)
    output_path   = Path(args.output)

    # ---------------------------------------------------------------------------
    # Step 1: Merge per-clue detection pkls → data_finished.pkl
    # ---------------------------------------------------------------------------
    df = merge_detections(data_pkl, detection_dir, finished_pkl, args.overwrite_finished)
    print(f"\n  {len(df)} rows, columns: {list(df.columns)}")

    # ---------------------------------------------------------------------------
    # Step 2: Smooth
    # ---------------------------------------------------------------------------
    if output_path.exists() and not args.overwrite_smoothed:
        print(f"\nSkipping smoothing: {output_path} exists (--overwrite-smoothed=false)")
    else:
        print(f"\nSmoothing with k={ks} ...")
        df_smoothed = smooth(df, ks)
        new_cols = [f"{clue}_k{k}" for k in ks for clue in CLUE_COLS]
        print(f"\nNew columns added: {new_cols}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_smoothed.to_pickle(output_path)
        print(f"Saved: {output_path}  ({len(df_smoothed)} rows)")

    # ---------------------------------------------------------------------------
    # Step 3: Plots (BEV panels + spectrum) from the merged data_finished.pkl
    # ---------------------------------------------------------------------------
    if args.plot:
        _here = Path(__file__).resolve().parent
        _python_root = _here.parent
        if str(_python_root) not in sys.path:
            sys.path.insert(0, str(_python_root))

        print(f"\nBEV plots: {finished_pkl}  →  {args.plot_dir}")
        from utils.plot_spacefeat import plot_spacefeat_bev_panels_df
        plot_spacefeat_bev_panels_df(df, output_dir=args.plot_dir, frame_step=args.plot_step, frames_root=args.frames_root)

        # Spectrum plots: original + one subdir per k.
        # Use df_smoothed (has both raw and smoothed columns); fall back to loading
        # from disk if smoothing was skipped this session (--overwrite-smoothed=false).
        try:
            df_for_spec = df_smoothed
        except NameError:
            print(f"  Loading smoothed pkl for spectrum: {output_path}")
            df_for_spec = pd.read_pickle(output_path)

        spectrum_root = Path(args.spectrum_dir)
        print(f"\nSpectrum plots  →  {spectrum_root}/{{original,{','.join(f'k{k}' for k in ks)}}}/")
        _plot_spectrum_per_batch(df_for_spec, spectrum_root / "original", col_suffix="")
        for k in ks:
            _plot_spectrum_per_batch(df_for_spec, spectrum_root / f"k{k}", col_suffix=f"_k{k}")

    print("\nDone.")
