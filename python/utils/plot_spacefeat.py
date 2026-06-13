"""
plot_spacefeat.py

BEV (bird's eye view) panel plots for GCFF ViTPose results.

Generates one figure per sampled frame with a 2×3 grid of subplots:
  - Left 2 columns (2×2): one subplot per body-part clue (head/shoulder/hip/foot).
    Each subplot shows person positions and orientations from `spaceFeat`.
    Nodes and arrows are coloured per person-ID (stable across clues and frames).
    Group polygons are drawn from the corresponding `{clue}Res` column.
  - Top right: all four clues overlaid in one space view, coloured by clue.
  - Bottom right: 2D pixel keypoint overlay on the real frame image
    (placeholder shown until `pixelCoords` / `img_path` columns are available).

Coordinates are in centimetres.
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

# Radius of orientation arrow (in the same units as spaceFeat x/y, i.e. cm)
_ARROW_RADIUS = 50
_POINT_SIZE = 80

_GROUP_COLORS = plt.cm.get_cmap("tab10").colors  # 10 distinct colours for group polygons
_SINGLETON_COLOR = (0.7, 0.7, 0.7)  # grey for isolated individuals

# Per-clue colours for the all-keypoints overlay panel (matches plot_person.py palette)
CLUE_COLORS = {
    "head":     "#d62728",
    "shoulder": "#1f77b4",
    "hip":      "#2ca02c",
    "foot":     "#9467bd",
}
# Short label per clue ("I" for hip avoids clash with head "H")
CLUE_LABELS = {
    "head":     "H",
    "shoulder": "S",
    "hip":      "I",
    "foot":     "F",
}


def _person_id_color(pid: int):
    """Stable colour for a person based on their ID, independent of grouping or clue."""
    return plt.get_cmap("tab20")(pid % 20)


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
    color_map: Dict[int, tuple] = None,  # kept for signature compat; not used
) -> None:
    """Draw convex-hull polygons for non-singleton groups in a BEV subplot.

    Polygon colour is assigned by group index (tab10) so it is independent of
    node/arrow colours (which are now pid-based).
    """
    from matplotlib.patches import Polygon as MplPolygon

    group_idx = 0
    for g in (groups or []):
        members = [int(p) for p in g] if g else []
        if len(members) < 2:
            continue
        pts = [pid_to_xy[m] for m in members if m in pid_to_xy]
        if len(pts) < 2:
            continue
        color = _GROUP_COLORS[group_idx % len(_GROUP_COLORS)]
        group_idx += 1
        hull = _convex_hull(np.asarray(pts, dtype=float))
        if hull.shape[0] >= 3:
            ax.add_patch(MplPolygon(
                hull, closed=True,
                facecolor=(*color[:3], 0.12), edgecolor=color, linewidth=1.5,
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
    ax.set_xlabel("x (cm)", fontsize=7)
    ax.set_ylabel("y (cm)", fontsize=7)
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

    pid_to_xy: Dict[int, tuple] = {}

    for row in arr:
        pid = int(row[0])
        x, y, alpha = float(row[1]), float(row[2]), float(row[3])
        color = _person_id_color(pid)
        pid_to_xy[pid] = (x, y)

        ax.scatter(x, y, s=_POINT_SIZE, color=color, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate("", xy=(x + _ARROW_RADIUS * math.cos(alpha),
                             y + _ARROW_RADIUS * math.sin(alpha)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        ax.text(x, y + _ARROW_RADIUS * 0.6, str(pid),
                fontsize=6, ha="center", va="bottom", color="black", zorder=4)

    _draw_group_polygons_bev(ax, groups, pid_to_xy)


def _plot_all_keypoints_ax(ax: plt.Axes, sf_dict: dict) -> None:
    """All clues' spaceFeat positions overlaid in one BEV axis, coloured by clue."""
    from matplotlib.lines import Line2D

    ax.set_title("All keypoints (space)", fontsize=8)
    ax.set_xlabel("x (cm)", fontsize=7)
    ax.set_ylabel("y (cm)", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)

    if not isinstance(sf_dict, dict):
        ax.text(0.5, 0.5, "no spaceFeat", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    any_data = False
    for clue in CLUES:
        arr = sf_dict.get(clue)
        if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
            continue
        try:
            arr = np.asarray(arr, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            continue
        color = CLUE_COLORS.get(clue, "grey")
        label_str = CLUE_LABELS.get(clue, clue[0].upper())
        any_data = True
        for r in arr:
            pid = int(r[0])
            x, y, alpha = float(r[1]), float(r[2]), float(r[3])
            ax.scatter(x, y, s=_POINT_SIZE * 0.7, color=color, zorder=3,
                       edgecolors="k", linewidths=0.4)
            ax.annotate("", xy=(x + _ARROW_RADIUS * math.cos(alpha),
                                 y + _ARROW_RADIUS * math.sin(alpha)),
                        xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
            ax.text(x, y + _ARROW_RADIUS * 0.5, f"{label_str}{pid}",
                    fontsize=5, ha="center", va="bottom", color=color, zorder=4)

    if not any_data:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=8)
        return

    handles = [
        Line2D([0], [0], color=CLUE_COLORS[c], marker="o", markersize=5,
               linestyle="none", label=c)
        for c in CLUES if c in CLUE_COLORS
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right")


def _plot_pixel_overlay_ax(ax: plt.Axes, row: pd.Series) -> None:
    """2D pixel keypoints on frame image, or a dark placeholder if data is absent.

    pixelCoords is {clue: (n_people, 4) array [person_id, u_rel, v_rel, orientation]}
    where u_rel = u/1920 and v_rel = v/1080 are in [0, 1].  When a background
    image is loaded, coords are scaled to image pixel dimensions automatically.
    """
    ax.set_title("2D keypoints (pixel)", fontsize=8)
    ax.tick_params(labelsize=6)

    pixel_data = row.get("pixelCoords") if hasattr(row, "get") else None
    img_path = row.get("img_path") if hasattr(row, "get") else None

    img_w, img_h = 1.0, 1.0  # relative [0, 1] space when no image
    img_loaded = False
    if img_path is not None:
        try:
            img = plt.imread(str(img_path))
            ax.imshow(img, aspect="auto")
            img_h, img_w = float(img.shape[0]), float(img.shape[1])
            img_loaded = True
        except Exception:
            pass

    if not img_loaded:
        ax.set_facecolor("#222222")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # y=0 at top, matches image convention

    if pixel_data is None or not isinstance(pixel_data, dict):
        ax.text(0.5, 0.5, "2D keypoint data\nnot yet available",
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=8)
        return

    # pixel_data: {clue: (n_people, 4) [person_id, u_rel, v_rel, orientation]}
    for clue, arr in pixel_data.items():
        try:
            arr = np.asarray(arr, dtype=float)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
            continue
        color = CLUE_COLORS.get(clue, "grey")
        u = arr[:, 1] * img_w
        v = arr[:, 2] * img_h
        ax.scatter(u, v, s=15, color=color, zorder=3, edgecolors="k", linewidths=0.3)


def _plot_bev_frame(row: pd.Series) -> plt.Figure:
    """Create a 2×3 BEV figure for one DataFrame row.

    Left two columns (2×2): one subplot per body-clue (head/shoulder/hip/foot).
    Top-right: all clues overlaid in one space plot, coloured by clue.
    Bottom-right: 2D pixel keypoint overlay on the real frame image.
    """
    sf_dict = row.get("spaceFeat", {}) or {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Map the four clue axes: head→[0,0], shoulder→[0,1], hip→[1,0], foot→[1,1]
    clue_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, clue in zip(clue_axes, CLUES):
        res_col = f"{clue}Res"
        sf = sf_dict.get(clue) if isinstance(sf_dict, dict) else None
        groups = row.get(res_col) or []
        _plot_clue_ax(ax, sf, groups, clue)

    _plot_all_keypoints_ax(axes[0, 2], sf_dict)
    _plot_pixel_overlay_ax(axes[1, 2], row)

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

