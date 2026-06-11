"""
plot_spacefeat.py

BEV (bird's eye view) panel plots for GCFF ViTPose results.

Generates one figure per sampled frame with a 2×2 grid of subplots — one
per body-part clue (head / shoulder / hip / foot).  Each subplot shows person
positions and orientations from `spaceFeat`, colour-coded by group assignment
from the corresponding `{clue}Res` column.

Requires only `spaceFeat` and `{clue}Res` columns; does not need
`spaceCoords` or `pixelCoords` (unavailable in ViTPose data).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLUES = ["head", "shoulder", "hip", "foot"]
CLUE_RES_COLS = [f"{c}Res" for c in CLUES]

# Radius of orientation arrow (in the same units as spaceFeat x/y, i.e. metres)
_ARROW_RADIUS = 0.15
_POINT_SIZE = 80

_GROUP_COLORS = plt.cm.get_cmap("tab10").colors  # 10 distinct colours
_SINGLETON_COLOR = (0.7, 0.7, 0.7)  # grey for isolated individuals


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def _person_to_group_color(groups: list) -> Dict[int, tuple]:
    """Map person_id → matplotlib colour based on group membership.

    Singletons (groups of size 1) get grey.
    """
    color_map: Dict[int, tuple] = {}
    group_idx = 0
    for g in (groups or []):
        members = [int(p) for p in g] if g else []
        if len(members) <= 1:
            for p in members:
                color_map[p] = _SINGLETON_COLOR
        else:
            color = _GROUP_COLORS[group_idx % len(_GROUP_COLORS)]
            for p in members:
                color_map[p] = color
            group_idx += 1
    return color_map


def _plot_clue_ax(ax: plt.Axes, sf: Optional[np.ndarray], groups: list, clue: str) -> None:
    """Draw one BEV subplot for a single clue."""
    ax.set_title(clue, fontsize=9)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)

    if sf is None or (hasattr(sf, "__len__") and len(sf) == 0):
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    try:
        arr = np.asarray(sf, dtype=np.float64)
    except Exception:
        ax.text(0.5, 0.5, "bad data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    color_map = _person_to_group_color(groups)

    for row in arr:
        pid = int(row[0])
        x, y, alpha = float(row[1]), float(row[2]), float(row[3])
        color = color_map.get(pid, _SINGLETON_COLOR)

        ax.scatter(x, y, s=_POINT_SIZE, color=color, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate("", xy=(x + _ARROW_RADIUS * math.cos(alpha),
                             y + _ARROW_RADIUS * math.sin(alpha)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        ax.text(x, y + _ARROW_RADIUS * 0.6, str(pid),
                fontsize=6, ha="center", va="bottom", color="black", zorder=4)


def _plot_bev_frame(row: pd.Series) -> plt.Figure:
    """Create a 2×2 BEV figure for one DataFrame row."""
    sf_dict = row.get("spaceFeat", {}) or {}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for ax, clue in zip(axes_flat, CLUES):
        res_col = f"{clue}Res"
        sf = sf_dict.get(clue) if isinstance(sf_dict, dict) else None
        groups = row.get(res_col) or []
        _plot_clue_ax(ax, sf, groups, clue)

    try:
        cam = int(row.get("Cam", 0))
        vid = int(row.get("Vid", 0))
        seg = int(row.get("Seg", 0))
        ts = int(row.get("Timestamp", 0))
    except Exception:
        cam = vid = seg = ts = 0

    fig.suptitle(f"BEV  Cam={cam} Vid={vid} Seg={seg} t={ts}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_spacefeat_bev_panels_df(
    data_kp: pd.DataFrame,
    output_dir,
    frame_step: int = 120,
) -> None:
    """Generate BEV panel figures every frame_step rows and save as PNG.

    Parameters
    ----------
    data_kp:
        GCFF DataFrame with at minimum `spaceFeat`, `{clue}Res`,
        `Cam`, `Vid`, `Seg`, `Timestamp` columns.
    output_dir:
        Directory where PNG files are written (created if absent).
    frame_step:
        Save one figure every frame_step rows (default 120).
    """
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total = len(data_kp)
    frame_step = max(1, int(frame_step))
    saved = 0

    for frame_idx in range(0, total, frame_step):
        row = data_kp.iloc[frame_idx]
        try:
            cam = int(row.get("Cam", 0))
            vid = int(row.get("Vid", 0))
            seg = int(row.get("Seg", 0))
            ts = int(row.get("Timestamp", frame_idx))
        except Exception:
            cam = vid = seg = 0
            ts = frame_idx

        filename = f"bev_{cam}{vid}{seg}_{ts}_{frame_idx}.png"
        fig_path = results_dir / filename
        if fig_path.exists():
            continue

        try:
            fig = _plot_bev_frame(row)
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved += 1
        except Exception as exc:
            print(f"  Warning: BEV plot failed for frame {frame_idx}: {exc}")
            try:
                plt.close("all")
            except Exception:
                pass

    print(f"BEV plots: saved {saved} figures to {results_dir}")
