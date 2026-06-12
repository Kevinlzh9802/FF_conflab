"""
diagnose_bev.py

Print spaceFeat / grouping-result stats from data_finished.pkl and generate
BEV panel plots for one batch.

Run from FF_conflab/python/ (or anywhere — paths resolve relative to this file):

    python python/diagnose_bev.py
    python python/diagnose_bev.py --batch=228
    python python/diagnose_bev.py --batch=228 --step=1   # every frame
    python python/diagnose_bev.py --data=../data/data_finished.pkl --batch=428
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root and add python/ to path so utils imports work
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # .../FF_conflab/python
_REPO_ROOT = _HERE.parent                         # .../FF_conflab
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_DEFAULT_DATA = _REPO_ROOT / "data" / "data_finished.pkl"
_DEFAULT_PLOT = _REPO_ROOT / "data" / "plots" / "bev"
_DEFAULT_BATCH = 428

CLUES = ["head", "shoulder", "hip", "foot"]
CLUE_RES_COLS = [f"{c}Res" for c in CLUES]

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _npeople(sf_dict, clue: str) -> int:
    if not isinstance(sf_dict, dict):
        return 0
    arr = sf_dict.get(clue)
    if arr is None:
        return 0
    a = np.asarray(arr)
    return int(a.shape[0]) if a.ndim == 2 else 0


def _batch_df(df: pd.DataFrame, batch: int) -> pd.DataFrame:
    """Return rows belonging to the given 3-digit batch number."""
    batch_str = f"{batch:03d}"
    cam, vid, seg = int(batch_str[0]), int(batch_str[1]), int(batch_str[2])
    if {"Cam", "Vid", "Seg"}.issubset(df.columns):
        return df[(df["Cam"] == cam) & (df["Vid"] == vid) & (df["Seg"] == seg)].copy()
    return df.copy()


def print_stats(df: pd.DataFrame, batch: int) -> pd.DataFrame:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  data_finished.pkl — global stats")
    print(sep)
    print(f"  Total rows  : {len(df)}")
    print(f"  Columns     : {list(df.columns)}")

    if {"Cam", "Vid", "Seg"}.issubset(df.columns):
        grp = df[["Cam", "Vid", "Seg"]].drop_duplicates()
        batch_ids = sorted(
            f"{int(r.Cam)}{int(r.Vid)}{int(r.Seg)}" for _, r in grp.iterrows()
        )
        print(f"  Batches     : {batch_ids}")

    # Global detection rate per clue
    if "spaceFeat" in df.columns:
        print(f"\n  Detection rate across ALL rows (% frames with ≥1 person per clue):")
        sf_col = df["spaceFeat"].tolist()
        for clue in CLUES:
            counts = [_npeople(sf, clue) for sf in sf_col]
            nonempty = sum(1 for c in counts if c > 0)
            mean_n = float(np.mean([c for c in counts if c > 0])) if nonempty else 0.0
            pct = 100.0 * nonempty / max(len(df), 1)
            print(f"    {clue:10s}: {nonempty}/{len(df)} ({pct:.1f}%)  "
                  f"mean people when present = {mean_n:.2f}")
    else:
        print("\n  WARNING: 'spaceFeat' column not found!")

    # Batch-specific stats
    bdf = _batch_df(df, batch)
    print(f"\n{sep}")
    print(f"  Batch {batch} stats  ({len(bdf)} rows)")
    print(sep)

    if len(bdf) == 0:
        print(f"  WARNING: batch {batch} not found in DataFrame.")
        return bdf

    if "concat_ts" in bdf.columns:
        print(f"  concat_ts range : [{bdf['concat_ts'].min()}, {bdf['concat_ts'].max()}]")
    if "Timestamp" in bdf.columns:
        print(f"  Timestamp range : [{bdf['Timestamp'].min()}, {bdf['Timestamp'].max()}]")

    if "spaceFeat" in bdf.columns:
        print(f"\n  spaceFeat per clue:")
        sf_col = bdf["spaceFeat"].tolist()
        for clue in CLUES:
            counts = [_npeople(sf, clue) for sf in sf_col]
            nonempty = sum(1 for c in counts if c > 0)
            mean_n = float(np.mean([c for c in counts if c > 0])) if nonempty else 0.0
            pct = 100.0 * nonempty / max(len(bdf), 1)
            print(f"    {clue:10s}: {nonempty}/{len(bdf)} frames non-empty ({pct:.1f}%)  "
                  f"mean={mean_n:.2f} people")

        # Inspect first row
        first_sf = bdf.iloc[0].get("spaceFeat")
        print(f"\n  First row spaceFeat:")
        print(f"    type = {type(first_sf)}")
        if isinstance(first_sf, dict):
            print(f"    keys = {list(first_sf.keys())}")
            for clue in CLUES:
                arr = first_sf.get(clue)
                if arr is None:
                    print(f"    [{clue}] → None")
                else:
                    a = np.asarray(arr)
                    print(f"    [{clue}] → shape={a.shape}  dtype={a.dtype}", end="")
                    if a.ndim == 2 and a.shape[0] > 0:
                        print(f"  first_person={a[0]}", end="")
                    print()
        else:
            print(f"    value = {first_sf}")

        # Find first non-empty row per clue
        print(f"\n  First non-empty row per clue:")
        for clue in CLUES:
            found = False
            for row_idx, (_, row) in enumerate(bdf.iterrows()):
                sf = row.get("spaceFeat") or {}
                n = _npeople(sf, clue)
                if n > 0:
                    a = np.asarray(sf[clue])
                    print(f"    {clue:10s}: row_idx={row_idx}  shape={a.shape}  "
                          f"first_person={a[0]}")
                    found = True
                    break
            if not found:
                print(f"    {clue:10s}: all empty in batch {batch}")

    # Grouping result columns
    res_present = [c for c in CLUE_RES_COLS if c in bdf.columns]
    print(f"\n  Grouping result columns ({CLUE_RES_COLS}):")
    if res_present:
        print(f"    Present : {res_present}")
        for col in res_present:
            nonempty = bdf[col].apply(lambda x: bool(x)).sum()
            print(f"    {col:12s}: {nonempty}/{len(bdf)} frames with non-empty grouping")
        print(f"    headRes sample (first 5 rows): "
              f"{[bdf.iloc[i].get('headRes') for i in range(min(5, len(bdf)))]}")
    else:
        print(f"    NONE present — GCFF hasn't been run yet, or wrong pkl loaded.")

    return bdf


# ---------------------------------------------------------------------------
# Inline BEV plotting (mirrors utils/plot_spacefeat.py for local use)
# ---------------------------------------------------------------------------

_COLORS = None

def _group_colors():
    global _COLORS
    if _COLORS is None:
        try:
            _COLORS = plt.colormaps["tab10"].colors
        except AttributeError:
            _COLORS = plt.cm.get_cmap("tab10").colors
    return _COLORS


def _person_color(groups: list) -> Dict[int, tuple]:
    singleton = (0.7, 0.7, 0.7)
    cmap = _group_colors()
    color_map: Dict[int, tuple] = {}
    gi = 0
    for g in (groups or []):
        members = [int(p) for p in (g or [])]
        if len(members) <= 1:
            for p in members:
                color_map[p] = singleton
        else:
            c = cmap[gi % len(cmap)]
            for p in members:
                color_map[p] = c
            gi += 1
    return color_map


def _plot_clue_ax(ax, sf, groups: list, clue: str) -> None:
    ax.set_title(clue, fontsize=9)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)

    if sf is None or (hasattr(sf, "__len__") and len(sf) == 0):
        ax.text(0.5, 0.5, f"no data ({clue})", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=9)
        return
    try:
        arr = np.asarray(sf, dtype=np.float64)
    except Exception as exc:
        ax.text(0.5, 0.5, f"bad data\n{exc}", transform=ax.transAxes,
                ha="center", va="center", color="red", fontsize=7)
        return
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        ax.text(0.5, 0.5, f"no data ({clue}, shape={arr.shape})", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    color_map = _person_color(groups)
    R = 0.15
    for row in arr:
        pid = int(row[0])
        x, y, alpha = float(row[1]), float(row[2]), float(row[3])
        color = color_map.get(pid, (0.7, 0.7, 0.7))
        ax.scatter(x, y, s=80, color=color, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(
            "", xy=(x + R * math.cos(alpha), y + R * math.sin(alpha)), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )
        ax.text(x, y + R * 0.6, str(pid), fontsize=6, ha="center", va="bottom",
                color="black", zorder=4)


def plot_batch_bev(bdf: pd.DataFrame, batch: int, output_dir: Path, step: int = 10) -> None:
    batch_dir = output_dir / f"{batch:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Plotting BEV → {batch_dir}  (every {step} rows, {len(bdf)//step + 1} figures)")

    saved = 0
    for pos, frame_idx in enumerate(range(0, len(bdf), step)):
        row = bdf.iloc[frame_idx]
        sf_dict = row.get("spaceFeat") or {}
        if not isinstance(sf_dict, dict):
            sf_dict = {}

        try:
            cam = int(row.get("Cam", 0))
            vid = int(row.get("Vid", 0))
            seg = int(row.get("Seg", 0))
            ts  = int(row.get("Timestamp", frame_idx))
        except Exception:
            cam = vid = seg = ts = 0

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, clue in zip(axes.flatten(), CLUES):
            sf = sf_dict.get(clue) if isinstance(sf_dict, dict) else None
            groups = row.get(f"{clue}Res") or []
            _plot_clue_ax(ax, sf, groups, clue)

        fig.suptitle(f"BEV  Cam={cam} Vid={vid} Seg={seg}  t={ts}  (row {frame_idx})",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # row_idx in the FULL original df (use the df iloc position relative to full dataset)
        # Use frame_idx as the rownum within the batch
        fig_path = batch_dir / f"{ts:04d}_{frame_idx:06d}.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved += 1
        if saved % 20 == 0:
            print(f"    ... {saved} figures saved")

    print(f"  Done: {saved} BEV figures saved to {batch_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose data_finished.pkl and plot BEV panels for one batch."
    )
    parser.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help=f"Path to data_finished.pkl (default: {_DEFAULT_DATA})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=_DEFAULT_BATCH,
        help=f"3-digit batch number to plot (default: {_DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--output_dir",
        default=str(_DEFAULT_PLOT),
        help=f"BEV plot root directory (default: {_DEFAULT_PLOT})",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Plot every N-th row of the batch (default: 10). Use 1 for all frames.",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Print stats only, skip plotting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"File not found: {data_path}")

    print(f"Loading {data_path} ...")
    df = pd.read_pickle(data_path)
    print(f"  {len(df)} rows loaded.")

    bdf = print_stats(df, args.batch)

    if not args.stats_only and len(bdf) > 0:
        plot_batch_bev(bdf, args.batch, Path(args.output_dir), step=args.step)
    elif len(bdf) == 0:
        print(f"\nNo rows for batch {args.batch} — nothing to plot.")

    print("\nDone.")
