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

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull. Duplicated from utils/plots.py to avoid circular import."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _draw_group_polygons_bev(
    ax: plt.Axes,
    groups: list,
    pid_to_xy: Dict[int, tuple],
    color_map: Dict[int, tuple],
) -> None:
    """Draw convex-hull polygons for non-singleton groups in a BEV subplot."""
    from matplotlib.patches import Polygon as MplPolygon

    for g in (groups or []):
        members = [int(p) for p in g] if g else []
        if len(members) < 2:
            continue
        pts = [pid_to_xy[m] for m in members if m in pid_to_xy]
        if len(pts) < 2:
            continue
        color = color_map.get(members[0], _SINGLETON_COLOR)
        hull = _convex_hull(np.asarray(pts, dtype=float))
        if hull.shape[0] >= 3:
            ax.add_patch(MplPolygon(
                hull, closed=True,
                facecolor=(*color, 0.12), edgecolor=color, linewidth=1.5,
            ))
        elif hull.shape[0] == 2:
            ax.plot(hull[:, 0], hull[:, 1], color=color, linewidth=1.5, alpha=0.6)


def _format_group_label(groups: list) -> str:
    """Format non-singleton groups as a compact string for subplot titles."""
    non_single = [g for g in (groups or []) if g and len(g) > 1]
    if not non_single:
        return "(all singletons)"
    return "  ".join(
        "[" + ",".join(str(int(p)) for p in sorted(g)) + "]"
        for g in non_single
    )


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
    ax.set_title(f"{clue}\n{_format_group_label(groups)}", fontsize=8)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)

    if sf is None or (hasattr(sf, "__len__") and len(sf) == 0):
        ax.text(0.5, 0.5, f"no data ({clue})", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    try:
        arr = np.asarray(sf, dtype=np.float64)
    except Exception:
        ax.text(0.5, 0.5, f"bad data ({clue})", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        ax.text(0.5, 0.5, f"no data ({clue}, shape={np.asarray(sf).shape})", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    color_map = _person_to_group_color(groups)
    pid_to_xy: Dict[int, tuple] = {}

    for row in arr:
        pid = int(row[0])
        x, y, alpha = float(row[1]), float(row[2]), float(row[3])
        color = color_map.get(pid, _SINGLETON_COLOR)
        pid_to_xy[pid] = (x, y)

        ax.scatter(x, y, s=_POINT_SIZE, color=color, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate("", xy=(x + _ARROW_RADIUS * math.cos(alpha),
                             y + _ARROW_RADIUS * math.sin(alpha)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        ax.text(x, y + _ARROW_RADIUS * 0.6, str(pid),
                fontsize=6, ha="center", va="bottom", color="black", zorder=4)

    _draw_group_polygons_bev(ax, groups, pid_to_xy, color_map)


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
        BEV root directory. A per-batch subdirectory ``<Cam><Vid><Seg>/``
        is created inside it for each batch. Files are named
        ``{Timestamp:04d}_{row_idx:06d}.png``.
    frame_step:
        Save one figure every frame_step rows (default 120).
    """
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic: show spaceFeat structure of the first row so empty-plot issues are visible.
    _sf_diag = data_kp.iloc[0].get("spaceFeat")
    print(
        f"[BEV diag] spaceFeat type={type(_sf_diag)}, "
        f"keys={list(_sf_diag.keys()) if isinstance(_sf_diag, dict) else 'not a dict'}"
    )
    if isinstance(_sf_diag, dict):
        for _c, _arr in _sf_diag.items():
            _a = np.asarray(_arr)
            print(f"  [{_c}] shape={_a.shape}  dtype={_a.dtype}")

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

        batch_num = f"{cam}{vid}{seg}"
        batch_dir = results_dir / batch_num
        batch_dir.mkdir(parents=True, exist_ok=True)
        fig_path = batch_dir / f"{ts:04d}_{frame_idx:06d}.png"
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


# ---------------------------------------------------------------------------
# Sample BEV helpers (single-clue panels for diagnostic plots)
# ---------------------------------------------------------------------------

def _plot_bev_single_clue_fig(row: pd.Series, clue: str) -> plt.Figure:
    """Single-panel BEV figure for one clue."""
    sf_dict = row.get("spaceFeat", {}) or {}
    sf = sf_dict.get(clue) if isinstance(sf_dict, dict) else None
    groups = row.get(f"{clue}Res") or []

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    _plot_clue_ax(ax, sf, groups, clue)

    try:
        cam = int(row.get("Cam", 0))
        vid = int(row.get("Vid", 0))
        seg = int(row.get("Seg", 0))
        ts = int(row.get("Timestamp", 0))
    except Exception:
        cam = vid = seg = ts = 0

    fig.suptitle(f"BEV [{clue}]  Cam={cam} Vid={vid} Seg={seg} t={ts}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_sample_bev_per_clue(
    data_kp: pd.DataFrame,
    output_dir,
    clue: str,
) -> None:
    """Save one single-clue BEV PNG per row in data_kp (pass an already-sampled DataFrame).

    Always overwrites existing files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

    for frame_idx in range(len(data_kp)):
        row = data_kp.iloc[frame_idx]
        try:
            cam = int(row.get("Cam", 0))
            vid = int(row.get("Vid", 0))
            seg = int(row.get("Seg", 0))
            ts = int(row.get("Timestamp", frame_idx))
        except Exception:
            cam = vid = seg = ts = frame_idx

        fig_path = out / f"{cam}{vid}{seg}_{ts:04d}_{frame_idx:06d}.png"
        try:
            fig = _plot_bev_single_clue_fig(row, clue)
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved += 1
        except Exception as exc:
            print(f"  Warning: sample BEV [{clue}] frame {frame_idx}: {exc}")
            try:
                plt.close("all")
            except Exception:
                pass

    print(f"Sample BEV [{clue}]: saved {saved} figures to {out}")
