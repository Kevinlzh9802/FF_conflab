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
DEFAULT_PANEL_PLOTS = f"{NEON}/zonghuan/data/conflab/GCFF/results/bev"
DEFAULT_SPECTRUM_PLOTS = f"{NEON}/zonghuan/data/conflab/GCFF/results/spectrum"

# Add the python directory to path so analysis/spatial imports work
_HERE = Path(__file__).resolve().parent
_PYTHON_ROOT = _HERE.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.cross_modal import detect_group_num_breakpoints
from analysis.spatial import spatial_scores_df


# ---------------------------------------------------------------------------
# Spectrum plot helpers (mirrors smooth_groupings.py)
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
# Window statistics
# ---------------------------------------------------------------------------

def _compute_window_stats(windows: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Return a DataFrame with one row per group-key containing window count + length stats."""
    stat_cols = ["n_windows", "length_mean", "length_median", "length_min", "length_max", "length_std"]
    rows = []
    for keys, grp in windows.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        lengths = grp["length"].dropna()
        n = len(lengths)
        row.update({
            "n_windows":     n,
            "length_mean":   float(lengths.mean())   if n else float("nan"),
            "length_median": float(lengths.median()) if n else float("nan"),
            "length_min":    float(lengths.min())    if n else float("nan"),
            "length_max":    float(lengths.max())    if n else float("nan"),
            "length_std":    float(lengths.std())    if n else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=group_cols + stat_cols) if rows else pd.DataFrame(columns=group_cols + stat_cols)


# ---------------------------------------------------------------------------
# Per-k analysis
# ---------------------------------------------------------------------------

def _run_for_k_on_data(
    data: pd.DataFrame,
    k: int,
    results_dir: Path,
    sp_path: Optional[str],
    label: str,
) -> Optional[pd.DataFrame]:
    """Shared core: detect windows, apply speaking filter, compute and save scores.

    Returns the final windows DataFrame (after all filtering) so callers can
    compute additional statistics, or None on early exit.
    """
    clue_cols = [f"headRes_k{k}", f"shoulderRes_k{k}", f"hipRes_k{k}", f"footRes_k{k}"]

    missing = [c for c in clue_cols if c not in data.columns]
    if missing:
        print(f"  [{label} k={k}] WARNING: columns missing: {missing}. Skipping.")
        return None

    print(f"\n[{label} k={k}] Detecting grouping-change windows using columns: {clue_cols}")
    windows = detect_group_num_breakpoints(data, clues=clue_cols)

    if windows.empty:
        print(f"  [{label} k={k}] No windows found. Skipping spatial scores.")
        return None

    lengths = windows["length"].dropna()
    print(
        f"  [{label} k={k}] Windows (raw): {len(windows)}  "
        f"length mean={lengths.mean():.1f}  median={lengths.median():.1f}  "
        f"min={lengths.min():.0f}  max={lengths.max():.0f}"
    )

    if sp_path and Path(sp_path).is_file():
        from analysis.cross_modal import count_speaker_groups, filter_windows
        print(f"  [{label} k={k}] Applying speaking filter from: {sp_path}")
        windows = count_speaker_groups(windows, speaking_status_path=sp_path)
        windows = filter_windows(windows)
        print(f"  [{label} k={k}] After speaking filter: {len(windows)} windows")
    else:
        if sp_path:
            print(f"  [{label} k={k}] WARNING: sp file not found at '{sp_path}'. Speaking filter skipped.")
        else:
            print(f"  [{label} k={k}] Note: speaking filter skipped (--sp not provided).")

    if windows.empty:
        print(f"  [{label} k={k}] No windows after filtering. Skipping spatial scores.")
        return None

    print(f"  [{label} k={k}] Computing spatial scores ...")
    result = spatial_scores_df(windows, feature_cols=clue_cols)
    if result is None:
        print(f"  [{label} k={k}] spatial_scores_df returned None. Skipping save.")
        return windows

    fig_h, fig_s = result
    if not hasattr(fig_h, "savefig"):
        # spatial_scores_df returned matrices (empty df or all-NaN) rather than figures
        print(f"  [{label} k={k}] spatial_scores_df returned matrices (empty or all-NaN). Skipping save.")
        return windows

    results_dir.mkdir(parents=True, exist_ok=True)
    hom_path = results_dir / f"homogeneity_k{k}.png"
    split_path = results_dir / f"split_k{k}.png"
    fig_h.savefig(hom_path, dpi=150, bbox_inches="tight")
    fig_s.savefig(split_path, dpi=150, bbox_inches="tight")
    print(f"  [{label} k={k}] Saved: {hom_path}")
    print(f"  [{label} k={k}] Saved: {split_path}")

    try:
        plt.close(fig_h)
        plt.close(fig_s)
    except Exception:
        pass

    return windows


def run_for_k_per_segment(
    data: pd.DataFrame,
    k: int,
    results_dir: Path,
    sp_path: Optional[str] = None,
) -> None:
    """Per-segment variant: treat each (Cam, Vid, Seg) as an independent sequence.

    Uses Timestamp as frame order within each segment instead of concat_ts.
    Metrics PNGs → results_dir/metrics/per-segment/{CamVidSeg}/
    Window stats → results_dir/windows/per-segment/windows_k{k}.csv  (all segments, one file)
    """
    clue_cols = [f"headRes_k{k}", f"shoulderRes_k{k}", f"hipRes_k{k}", f"footRes_k{k}"]

    missing = [c for c in clue_cols if c not in data.columns]
    if missing:
        print(f"  [per-seg k={k}] WARNING: columns missing: {missing}. Skipping.")
        return

    all_stats: List[pd.DataFrame] = []

    for (cam, vid, seg), seg_df in data.groupby(["Cam", "Vid", "Seg"]):
        seg_key = f"{int(cam)}{int(vid)}{int(seg)}"
        print(f"\n[per-seg k={k}] Segment {seg_key}: {len(seg_df)} rows")

        seg_data = seg_df.copy()
        seg_data["concat_ts"] = seg_data["Timestamp"]

        windows = _run_for_k_on_data(
            data=seg_data,
            k=k,
            results_dir=results_dir / "metrics" / "per-segment" / seg_key,
            sp_path=sp_path,
            label=f"per-seg/{seg_key}",
        )

        if windows is not None and not windows.empty:
            w = windows.copy()
            w["Seg"] = int(seg)
            all_stats.append(_compute_window_stats(w, ["Cam", "Vid", "Seg"]))

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        win_dir = results_dir / "windows" / "per-segment"
        win_dir.mkdir(parents=True, exist_ok=True)
        csv_path = win_dir / f"windows_k{k}.csv"
        combined.to_csv(csv_path, index=False)
        print(f"  [per-seg k={k}] Window stats ({len(combined)} segments) → {csv_path}")


def run_for_k(
    data: pd.DataFrame,
    k: int,
    results_dir: Path,
    sp_path: Optional[str] = None,
) -> None:
    """Global analysis for k: metrics PNGs → results_dir/metrics/, window stats → results_dir/windows/."""
    windows = _run_for_k_on_data(data, k, results_dir / "metrics", sp_path, label="global")
    if windows is not None and not windows.empty:
        stats = _compute_window_stats(windows, ["Cam", "Vid"])
        windows_dir = results_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        csv_path = windows_dir / f"windows_k{k}.csv"
        stats.to_csv(csv_path, index=False)
        print(f"  [global k={k}] Window stats → {csv_path}")


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
        help=(
            f"Root results directory. Subdirs metrics/, windows/, bev/, spectrum/ are created inside. "
            f"Default: {DEFAULT_RESULTS}"
        ),
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
    parser.add_argument(
        "--frames_root",
        default=f"{NEON}/zonghuan/data/conflab/bbox_kp",
        help=(
            "Root containing {batch}/images/{frame_id:08d}.jpg for BEV pixel overlay. "
            f"Default: {NEON}/zonghuan/data/conflab/bbox_kp"
        ),
    )
    parser.add_argument(
        "--spectrum_dir",
        default=DEFAULT_SPECTRUM_PLOTS,
        help=f"Root directory for spectrum PNG output. Subdirs original/, k{{k}}/ are created inside. Default: {DEFAULT_SPECTRUM_PLOTS}",
    )
    parser.add_argument(
        "--per-segment",
        action="store_true",
        help=(
            "Treat every (Cam, Vid, Seg) as an independent sequence (discard concat_ts). "
            "Saves metrics to --results_dir/metrics/per-segment/<CamVidSeg>/ "
            "and window stats to --results_dir/windows/per-segment/<CamVidSeg>/."
        ),
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
    print(f"Per-segment : {args.per_segment}")

    if args.per_segment:
        print("\n--- Per-segment analysis ---")
        for k in ks:
            run_for_k_per_segment(data, k, results_dir, sp_path=args.sp)
    else:
        for k in ks:
            run_for_k(data, k, results_dir, sp_path=args.sp)

    if args.plot:
        finished_path = Path(args.finished)
        print(f"\nBEV plots: loading unsmoothed results from {finished_path}")
        finished_df = pd.read_pickle(finished_path)
        from utils.plot_spacefeat import plot_spacefeat_bev_panels_df
        plot_spacefeat_bev_panels_df(
            finished_df,
            output_dir=args.plot_dir,
            frame_step=args.plot_step,
            frames_root=args.frames_root,
        )

        # Spectrum plots: original + one subdir per k.
        # `data` (smoothed pkl) retains raw {clue}Res columns alongside {clue}Res_k{k}.
        spectrum_root = Path(args.spectrum_dir)
        print(f"\nSpectrum plots  →  {spectrum_root}/{{original,{','.join(f'k{k}' for k in ks)}}}/")
        _plot_spectrum_per_batch(data, spectrum_root / "original", col_suffix="")
        for k in ks:
            _plot_spectrum_per_batch(data, spectrum_root / f"k{k}", col_suffix=f"_k{k}")

    print("\nDone.")
