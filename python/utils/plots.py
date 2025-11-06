from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from utils.groups import equal_groups
from utils.pose import PersonSkeleton, PoseArrow, build_person_skeletons


def _setup_3d(ax=None, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    return fig, ax


def _set_equal_3d(ax, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    # Set equal aspect for 3D by setting limits to the same span
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    if max_range == 0:
        max_range = 1.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


ARROW_LENGTH = 25.0

LABEL_MAP = {
    'head': {0: 'H', 1: 'N'},
    'shoulder': {2: 'LS', 3: 'RS'},
    'hip': {4: 'LI', 5: 'RI'},
    'foot': {6: 'LA', 7: 'RA', 8: 'LF', 9: 'RF'},
}

def _normalize_show_flags(flags) -> Tuple[bool, bool, bool, bool]:
    try:
        if isinstance(flags, (bool, np.bool_)):
            return (bool(flags),) * 4
        seq = list(flags)
        if len(seq) == 0:
            return (False, False, False, False)
        if len(seq) < 4:
            seq = (seq * 4)[:4]
        elif len(seq) > 4:
            seq = seq[:4]
        return tuple(bool(x) for x in seq)
    except Exception:
        return (True, True, True, True)


def _iter_person_skeletons(Coords: np.ndarray, Feats: dict, rows: Sequence[int], base_height: float) -> Iterable[PersonSkeleton]:
    head_feats = None
    if isinstance(Feats, dict):
        head_feats = Feats.get('head')
    for r in rows:
        pts2 = _extract_xy_from_headfeat(Coords[r, :])  # (10, 2)
        skeletons = build_person_skeletons(
            [pts2],
            base_height=base_height,
            head_feats=head_feats,
            row_indices=[r],
        )
        for skel in skeletons:
            yield skel


def _person_label(person: PersonSkeleton) -> str:
    return f"{person.pid}" if person.pid is not None else f"p{person.row_index}"


def _safe_xy(point: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2:
        return None
    xy = arr[:2]
    if np.any(np.isnan(xy)):
        return None
    return xy


def _normalize_groups(groups) -> Sequence[Sequence[int]]:
    normalized = []
    if groups is None:
        return normalized
    try:
        iterator = list(groups)
    except TypeError:
        return normalized
    for g in iterator:
        members = []
        if isinstance(g, (list, tuple, set, np.ndarray)):
            for m in g:
                try:
                    members.append(int(m))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                members.append(int(g))
            except (TypeError, ValueError):
                members = []
        if members:
            normalized.append(members)
    return normalized


def _format_groups_text(groups: Optional[Sequence[Sequence[int]]], prefix: str = 'Groups: ', width: int = 48) -> str:
    if not groups:
        body = '[]'
    else:
        body = ', '.join('[' + ', '.join(str(int(m)) for m in group) + ']' for group in groups)
    text = prefix + body
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False, max_lines=3, placeholder='…')

def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 2:
        return pts
    pts = np.unique(pts, axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _draw_group_polygons(ax,
                         groups: Sequence[Sequence[int]],
                         person_lookup: Dict[int, PersonSkeleton],
                         point_selector):
    if not groups:
        return
    color = 'lime'
    for members in groups:
        pts = []
        for member in members:
            person = person_lookup.get(member)
            if person is None:
                continue
            xy = point_selector(person)
            if xy is None:
                continue
            pts.append(np.asarray(xy, dtype=float))
        if not pts:
            continue
        hull = _convex_hull(np.asarray(pts, dtype=float))
        if hull.shape[0] >= 3:
            ax.add_patch(Polygon(hull, closed=True, facecolor='none', edgecolor=color, linewidth=2))
        elif hull.shape[0] == 2:
            ax.plot(hull[:, 0], hull[:, 1], color=color, linewidth=2)
        elif hull.shape[0] == 1:
            ax.scatter(hull[:, 0], hull[:, 1], color=color, s=60, marker='o')


def _vectors_aligned(vec_a: np.ndarray, vec_b: np.ndarray, tol: float = 0.99) -> bool:
    if vec_a is None or vec_b is None:
        return False
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0 or np.isnan(norm_a) or np.isnan(norm_b):
        return False
    cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return cos_sim >= tol


def _get_pose_arrow(person: PersonSkeleton, pose: str) -> Tuple[List[PoseArrow], bool]:
    arrows: List[PoseArrow] = []
    discrepancy = False
    pts = person.points
    if pose == 'head':
        kp_arrow: Optional[PoseArrow] = None
        if not np.any(np.isnan(pts[[0, 1], :2])):
            start = pts[0]
            vec = np.array([pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]], dtype=float)
            if not (np.any(np.isnan(vec)) or np.allclose(vec, 0.0)):
                kp_arrow = PoseArrow(start=start, vec=vec, kind='head_kp')
        angle = getattr(person, 'exp_head_pose', float('nan'))
        if angle is not None and not np.isnan(angle) and not np.any(np.isnan(pts[0, :2])):
            vec_manual = np.array([math.cos(angle), math.sin(angle)], dtype=float)
            if not (np.any(np.isnan(vec_manual)) or np.allclose(vec_manual, 0.0)):
                manual_arrow = PoseArrow(start=pts[0], vec=vec_manual, kind='head_manual')
                if kp_arrow is not None:
                    if _vectors_aligned(kp_arrow.vec, manual_arrow.vec):
                        arrows.append(kp_arrow)
                    else:
                        discrepancy = True
                        # Only store manual arrow when there is discrepancy
                        arrows.append(manual_arrow)
            elif kp_arrow is not None:
                arrows.append(kp_arrow)
        else:
            if kp_arrow is not None:
                arrows.append(kp_arrow)
        return arrows, discrepancy
    if pose == 'shoulder':
        if np.any(np.isnan(person.mid_shoulder)) or np.any(np.isnan(pts[[2, 3], :2])):
            return arrows, discrepancy
        vx = pts[3, 0] - pts[2, 0]
        vy = pts[3, 1] - pts[2, 1]
        vec = np.array([-vy, vx], dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return arrows, discrepancy
        arrows.append(PoseArrow(start=person.mid_shoulder, vec=vec, kind='shoulder'))
        return arrows, discrepancy
    if pose == 'hip':
        if np.any(np.isnan(person.mid_hip)) or np.any(np.isnan(pts[[4, 5], :2])):
            return arrows, discrepancy
        vx = pts[5, 0] - pts[4, 0]
        vy = pts[5, 1] - pts[4, 1]
        vec = np.array([-vy, vx], dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return arrows, discrepancy
        arrows.append(PoseArrow(start=person.mid_hip, vec=vec, kind='hip'))
        return arrows, discrepancy
    if pose == 'foot':
        if person.foot_vector is None or np.any(np.isnan(person.foot_center[:2])):
            return arrows, discrepancy
        vec = np.asarray(person.foot_vector, dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return arrows, discrepancy
        arrows.append(PoseArrow(start=person.foot_center, vec=vec, kind='foot'))
        return arrows, discrepancy
    return arrows, discrepancy


def _draw_arrow(ax, start: Optional[np.ndarray], vec: Optional[np.ndarray], projection: str,
                color, x_vals: Optional[list] = None, y_vals: Optional[list] = None):
    if start is None or vec is None:
        return None
    sx, sy = float(start[0]), float(start[1])
    vx, vy = float(vec[0]), float(vec[1])
    if any(np.isnan([sx, sy, vx, vy])):
        return None
    if np.allclose([vx, vy], [0.0, 0.0]):
        return None
    norm = math.hypot(vx, vy)
    if norm == 0 or np.isnan(norm):
        return None
    vx = (vx / norm) * ARROW_LENGTH
    vy = (vy / norm) * ARROW_LENGTH

    if projection == '3d':
        sz = float(start[2])
        if np.isnan(sz):
            return None
        try:
            ax.quiver(sx, sy, sz, vx, vy, 0.0, color=color, linewidth=1.5, arrow_length_ratio=0.2)
        except TypeError:
            ax.quiver(sx, sy, sz, vx, vy, 0.0, color=color, linewidth=1.5)
        return np.array([sx, sy, sz], dtype=float)
    else:
        ax.quiver(sx, sy, vx, vy, color=color, angles='xy', scale_units='xy', scale=1.0, width=0.005)
        if x_vals is not None:
            x_vals.extend([sx, sx + vx])
        if y_vals is not None:
            y_vals.extend([sy, sy + vy])
        return np.array([sx, sy], dtype=float)


def _finalize_2d_axis(ax, x_vals, y_vals):
    if not x_vals or not y_vals:
        return
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    valid_mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    if not np.any(valid_mask):
        return
    x_arr = x_arr[valid_mask]
    y_arr = y_arr[valid_mask]
    xmin, xmax = x_arr.min(), x_arr.max()
    ymin, ymax = y_arr.min(), y_arr.max()
    span = max(xmax - xmin, ymax - ymin)
    if span == 0:
        span = 1.0
    pad = span * 0.1
    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)
    lim_half = 0.5 * span + pad
    ax.set_xlim(cx - lim_half, cx + lim_half)
    ax.set_ylim(cy - lim_half, cy + lim_half)
    ax.set_aspect('equal', adjustable='box')


def _plot_person_skeleton(ax,
                          person: PersonSkeleton,
                          color,
                          projection: str,
                          show_flags: Tuple[bool, bool, bool, bool],
                          x_vals,
                          y_vals,
                          legend_label: Optional[str] = None) -> bool:
    points = person.points
    X_all = points[:, 0]
    Y_all = points[:, 1]
    Z_all = points[:, 2]
    mask_xy = ~np.isnan(X_all) & ~np.isnan(Y_all)

    plotted = False
    if projection == '3d':
        mask_pts = mask_xy & ~np.isnan(Z_all)
        if np.any(mask_pts):
            ax.scatter(X_all[mask_pts], Y_all[mask_pts], Z_all[mask_pts], s=20, color=color, label=legend_label)
            plotted = True
        if not np.any(np.isnan(person.mid_shoulder)):
            ax.scatter([person.mid_shoulder[0]], [person.mid_shoulder[1]], [person.mid_shoulder[2]],
                       s=20, color=color)
            plotted = True
        if not np.any(np.isnan(person.mid_hip)):
            ax.scatter([person.mid_hip[0]], [person.mid_hip[1]], [person.mid_hip[2]],
                       s=20, color=color)
            plotted = True
    else:
        if np.any(mask_xy):
            ax.scatter(X_all[mask_xy], Y_all[mask_xy], s=20, color=color, label=legend_label)
            x_vals.extend(X_all[mask_xy].tolist())
            y_vals.extend(Y_all[mask_xy].tolist())
            plotted = True
        if not np.any(np.isnan(person.mid_shoulder[:2])):
            ax.scatter([person.mid_shoulder[0]], [person.mid_shoulder[1]], s=20, color=color)
            x_vals.append(float(person.mid_shoulder[0]))
            y_vals.append(float(person.mid_shoulder[1]))
            plotted = True
        if not np.any(np.isnan(person.mid_hip[:2])):
            ax.scatter([person.mid_hip[0]], [person.mid_hip[1]], s=20, color=color)
            x_vals.append(float(person.mid_hip[0]))
            y_vals.append(float(person.mid_hip[1]))
            plotted = True

    def plot_segment(p1: np.ndarray, p2: np.ndarray):
        nonlocal plotted
        if np.any(np.isnan(p1[:2])) or np.any(np.isnan(p2[:2])):
            return
        if projection == '3d':
            if np.isnan(p1[2]) or np.isnan(p2[2]):
                return
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2)
            x_vals.extend([float(p1[0]), float(p2[0])])
            y_vals.extend([float(p1[1]), float(p2[1])])
        plotted = True

    nose = points[1]
    head = points[0]
    left_shoulder = points[2]
    right_shoulder = points[3]
    left_hip = points[4]
    right_hip = points[5]
    left_ankle = points[6]
    right_ankle = points[7]
    left_foot = points[8]
    right_foot = points[9]
    P_midS = person.mid_shoulder
    P_midH = person.mid_hip

    plot_segment(nose, head)
    plot_segment(head, P_midS)
    plot_segment(P_midS, left_shoulder)
    plot_segment(P_midS, right_shoulder)
    plot_segment(P_midS, P_midH)
    plot_segment(P_midH, left_hip)
    plot_segment(P_midH, right_hip)
    plot_segment(left_hip, left_ankle)
    plot_segment(left_ankle, left_foot)
    plot_segment(right_hip, right_ankle)
    plot_segment(right_ankle, right_foot)

    id_anchor = None
    show_head, show_shoulder, show_hip, show_foot = show_flags
    if show_head:
        arrows, _ = _get_pose_arrow(person, 'head')
        for arrow in arrows:
            arrow_color = 'k' if arrow.kind == 'head_manual' else color
            anchor = _draw_arrow(ax, arrow.start, arrow.vec, projection, arrow_color, x_vals, y_vals)
            if anchor is not None and id_anchor is None:
                id_anchor = anchor
    if show_shoulder:
        arrows, _ = _get_pose_arrow(person, 'shoulder')
        for arrow in arrows:
            anchor = _draw_arrow(ax, arrow.start, arrow.vec, projection, color, x_vals, y_vals)
            if anchor is not None and id_anchor is None:
                id_anchor = anchor
    if show_hip:
        arrows, _ = _get_pose_arrow(person, 'hip')
        for arrow in arrows:
            anchor = _draw_arrow(ax, arrow.start, arrow.vec, projection, color, x_vals, y_vals)
            if anchor is not None and id_anchor is None:
                id_anchor = anchor
    if show_foot:
        arrows, _ = _get_pose_arrow(person, 'foot')
        for arrow in arrows:
            anchor = _draw_arrow(ax, arrow.start, arrow.vec, projection, color, x_vals, y_vals)
            if anchor is not None and id_anchor is None:
                id_anchor = anchor

    if id_anchor is None:
        if projection == '3d':
            valid = ~np.isnan(points[:, :3]).any(axis=1)
            if np.any(valid):
                id_anchor = points[valid][0]
        else:
            valid = ~np.isnan(points[:, :2]).any(axis=1)
            if np.any(valid):
                id_anchor = points[valid][0][:2]

    label = _person_label(person)
    if id_anchor is not None:
        if projection == '3d' and len(id_anchor) >= 3:
            ax.text(float(id_anchor[0]), float(id_anchor[1]), float(id_anchor[2]), label,
                    color=color, fontsize=8)
        elif projection == '2d' or len(id_anchor) == 2:
            ax.text(float(id_anchor[0]) + 5.0, float(id_anchor[1]) + 5.0, label,
                    color=color, fontsize=8)

    return plotted


def plot_skeleton_3d(data_kp: Any,
                     frame_idx: int,
                     key: str = 'headFeat',
                     ax=None,
                     title: Optional[str] = None,
                     show: bool = True):
    """Framework to plot 3D skeleton for one frame.

    - data_kp: mapping or object with key-based access, e.g., data_kp['headFeat'][k]
    - frame_idx: frame index
    - key: which feature key to use ('headFeat', etc.)
    - ax: optional Matplotlib 3D axes; created if None
    - title: optional title
    - show: whether to call plt.show() when ax was created

    Notes:
    - This is a framework: we scatter points; bone connections can be added later.
    - Expects coordinates of shape (N,3) or (3,N). If a different shape is found,
      attempts to reshape to (-1,3) if possible; otherwise raises.
    """
    import matplotlib.pyplot as plt

    coords = data_kp[key][frame_idx]
    arr = np.asarray(coords)
    if arr.ndim == 2 and arr.shape[1] == 3:
        pts = arr
    elif arr.ndim == 2 and arr.shape[0] == 3:
        pts = arr.T
    elif arr.size % 3 == 0:
        pts = arr.reshape((-1, 3))
    else:
        raise ValueError(f"Unsupported coordinate shape for {key}[{frame_idx}]: {arr.shape}")

    fig, ax = _setup_3d(ax, title or f"Skeleton 3D - {key}[{frame_idx}]")
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(X, Y, Z, s=20, c='tab:blue', depthshade=True)
    _set_equal_3d(ax, X, Y, Z)

    if show and fig is not None:
        plt.show()
    return ax


def plot_skeletons_3d_grid(data_kp: Any,
                           frame_indices: Sequence[int],
                           key: str = 'headFeat',
                           ncols: int = 4,
                           figsize: Tuple[int, int] = (12, 8)):
    """Plot multiple frames in a grid of 3D subplots.

    - frame_indices: list of frames to visualize
    - key: which feature array to read
    - ncols: number of columns in the subplot grid
    - figsize: figure size in inches
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(frame_indices)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=figsize)
    axes = []
    for i, k in enumerate(frame_indices):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        plot_skeleton_3d(data_kp, k, key=key, ax=ax, title=f"{key}[{k}]", show=False)
        axes.append(ax)
    plt.tight_layout()
    plt.show()
    return axes


def _extract_xy_from_headfeat(Coords: np.ndarray) -> np.ndarray:
    """Extract XY keypoints from a single row of A (right 24 cols of headFeat).

    MATLAB mapping (1-based) within A (m x 24):
      5=headX,6=headY; 7=noseX,8=noseY;
      9=lShoulderX,10=lShoulderY; 11=rShoulderX,12=rShoulderY;
      13=lHipX,14=lHipY; 15=rHipX,16=rHipY;
      17=lAnkleX,18=lAnkleY; 19=rAnkleX,20=rAnkleY;
      21=lFootX,22=lFootY; 23=rFootX,24=rFootY.
    Zero-based indices inside A: subtract 1.
    Returns array of shape (10, 2) in order:
      head, nose, lShoulder, rShoulder, lHip, rHip, lAnkle, rAnkle, lFoot, rFoot
    """
    # zero-based indices in Coords
    idx_pairs = [
        (0, 1),   # head
        (2, 3),   # nose
        (4, 5),   # left shoulder
        (6, 7), # right shoulder
        (12, 13), # left hip
        (10, 11), # right hip
        (12, 13), # left ankle
        (14, 15), # right ankle
        (16, 17), # left foot
        (18, 19), # right foot
    ]
    pts = []
    for ix, iy in idx_pairs:
        x = Coords[ix] if ix < Coords.shape[0] else np.nan
        y = Coords[iy] if iy < Coords.shape[0] else np.nan
        pts.append((x, y))
    return np.asarray(pts, dtype=float)



def plot_all_skeletons(data_kp: Any,
                       frame_idx: int,
                       source: str = 'space',
                       persons: Optional[Sequence[int]] = None,
                       ax=None,
                       title: Optional[str] = None,
                       show: bool = True,
                       show_poses: Sequence[bool] = (True, True, True, True),
                       base_height: float = 170.0,
                       projection: str = '3d',
                       axis_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None):
    """Plot skeleton keypoints for one frame in 2D or 3D.

    - data_kp[key][frame_idx] must be an (m x 48) array. We take A = [:, 24:48] (0-based slicing)
      and map indices per MATLAB description (subtract 1 already handled here).
    - persons: optional indices of rows (people) to plot; default plots all.
    - Z coordinate is set using a simple height model: Z = base_height * coef, with
      coefficients per keypoint name:
        head:1, nose:0.95, leftShoulder/rightShoulder:0.85, leftHip/rightHip:0.5,
        leftAnkle/rightAnkle:0.02, leftFoot/rightFoot:0.02
    - show_poses: 1x4 booleans (head, shoulder, hip, foot) to toggle arrows.
    - projection: choose '3d' (default) for a 3D scatter or '2d' to plot on the XY plane.
    - axis_limits: optional ((xmin, xmax), (ymin, ymax)) tuple applied when projection='2d'.
    """

    plot_title = title or f"HeadFeat (XY) - {source}[{frame_idx}]"
    projection = projection.lower()
    if projection not in ('2d', '3d'):
        raise ValueError(f"projection must be '2d' or '3d', got {projection}")

    coords_key = source + "Coords"
    feat_key = source + "Feat"
    Coords = np.asarray(data_kp[coords_key][frame_idx])
    Feats = data_kp[feat_key][frame_idx]

    if Coords.ndim != 2 or Coords.shape[1] < 20:
        Warning(f"Expected (m x 20) array for {coords_key}[{frame_idx}], got {Coords.shape}")
        return fig, None
    
    # B = arr[:, 0:24]
    m = Coords.shape[0]
    rows = list(range(m)) if persons is None else list(persons)

    if projection == '3d':
        fig, ax = _setup_3d(ax, plot_title)
        x_vals: list = []
        y_vals: list = []
    else:
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(plot_title)
        x_vals = []
        y_vals = []

    show_flags = _normalize_show_flags(show_poses)
    persons_data = list(_iter_person_skeletons(Coords, Feats, rows, base_height))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))
    any_pts = False

    for idx, person in enumerate(persons_data):
        col = tuple(colors[idx % len(colors)])
        legend_label = _person_label(person)
        plotted = _plot_person_skeleton(
            ax=ax,
            person=person,
            color=col,
            projection=projection,
            show_flags=show_flags,
            x_vals=x_vals,
            y_vals=y_vals,
            legend_label=legend_label,
        )
        any_pts = any_pts or plotted

    if projection == '3d':
        if any_pts:
            scatters = [c for c in ax.collections if hasattr(c, '_offsets3d')] if ax.collections else []
            xdata = np.concatenate([c._offsets3d[0] for c in scatters]) if scatters else np.array([0])
            ydata = np.concatenate([c._offsets3d[1] for c in scatters]) if scatters else np.array([0])
            zdata = np.concatenate([c._offsets3d[2] for c in scatters]) if scatters else np.array([0])
            _set_equal_3d(ax, np.asarray(xdata), np.asarray(ydata), np.asarray(zdata))
    else:
        if axis_limits is not None:
            try:
                x_lim, y_lim = axis_limits
                ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
                ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
            except (TypeError, ValueError):
                _finalize_2d_axis(ax, x_vals, y_vals)
            else:
                ax.set_aspect('equal', adjustable='box')
        elif any_pts:
            _finalize_2d_axis(ax, x_vals, y_vals)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_loc = 'best' if projection == '2d' else 'upper right'
        ax.legend(loc=legend_loc, fontsize=8)
    if show and fig is not None:
        plt.show()
    return ax


def _plot_pose_panel(ax,
                     persons: Sequence[PersonSkeleton],
                     colors,
                     indices: Sequence[int],
                     arrow_key: Optional[str],
                     show_flag: bool,
                     title: str,
                     midpoint_attr: Optional[str] = None,
                     center_attr: Optional[str] = None,
                     groups: Optional[Sequence[Sequence[int]]] = None,
                     prev_groups: Optional[Sequence[Sequence[int]]] = None,
                     person_lookup: Optional[Dict[int, PersonSkeleton]] = None,
                     point_selector=None,
                     clue: Optional[str] = None,
                     show_ids: bool = True,
                     x_lim: Optional[Tuple[float, float]] = None,
                     y_lim: Optional[Tuple[float, float]] = None):
    x_vals: list = []
    y_vals: list = []
    labels_added = set()
    any_content = False
    panel_discrepancy = False
    label_lookup = LABEL_MAP.get(clue or '', {})

    for idx, person in enumerate(persons):
        col = tuple(colors[idx % len(colors)])
        pts = person.points[indices, :]
        mask = ~np.isnan(pts[:, 0]) & ~np.isnan(pts[:, 1])
        if np.any(mask):
            xs = pts[mask, 0]
            ys = pts[mask, 1]
            label = _person_label(person)
            label_to_use = label if label not in labels_added else None
            ax.scatter(xs, ys, s=20, color=col, label=label_to_use)
            if label_to_use:
                labels_added.add(label)
            x_vals.extend(xs.tolist())
            y_vals.extend(ys.tolist())
            any_content = True
            for local_idx, global_idx in enumerate(indices):
                lab = label_lookup.get(global_idx)
                if lab is None:
                    continue
                if local_idx >= pts.shape[0]:
                    continue
                px, py = pts[local_idx, 0], pts[local_idx, 1]
                if np.isnan(px) or np.isnan(py):
                    continue
                ax.text(float(px) + 4.0, float(py) + 4.0, lab, color=col, fontsize=7)
        if midpoint_attr:
            mid = getattr(person, midpoint_attr, None)
            if mid is not None and not np.any(np.isnan(mid[:2])):
                ax.scatter([mid[0]], [mid[1]], s=20, color=col)
                x_vals.append(float(mid[0]))
                y_vals.append(float(mid[1]))
                any_content = True
        if center_attr:
            center = getattr(person, center_attr, None)
            if center is not None and not np.any(np.isnan(center[:2])):
                ax.scatter([center[0]], [center[1]], s=20, color=col)
                x_vals.append(float(center[0]))
                y_vals.append(float(center[1]))
                any_content = True
        id_anchor = None
        if show_flag and arrow_key:
            arrows, mismatch = _get_pose_arrow(person, arrow_key)
            if mismatch:
                panel_discrepancy = True
            for arrow in arrows:
                draw_color = 'k' if arrow.kind == 'head_manual' else col
                anchor = _draw_arrow(ax, arrow.start, arrow.vec, '2d', draw_color, x_vals, y_vals)
                if anchor is not None:
                    any_content = True
                    if show_ids and id_anchor is None:
                        id_anchor = anchor
        if show_ids and id_anchor is None:
            valid = ~np.isnan(person.points[:, :2]).any(axis=1)
            if np.any(valid):
                id_anchor = person.points[valid][0][:2]
        if show_ids and id_anchor is not None:
            ax.text(float(id_anchor[0]) + 6.0, float(id_anchor[1]) + 6.0, _person_label(person), color=col, fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    panel_title = title
    if arrow_key == 'head' and panel_discrepancy:
        if '[O]' not in panel_title:
            panel_title = f"{panel_title} [O]"
    # Detect change vs previous frame's groups and mark with [C]
    try:
        if prev_groups is not None and not equal_groups(groups, prev_groups):
            if '[C]' not in panel_title:
                panel_title = f"{panel_title} [C]"
    except Exception:
        pass
    ax.set_title(panel_title)
    if x_lim is not None and y_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_aspect('equal', adjustable='box')
    else:
        _finalize_2d_axis(ax, x_vals, y_vals)
    if groups and person_lookup and point_selector:
        _draw_group_polygons(ax, groups, person_lookup, point_selector)
    if groups is not None:
        group_text = _format_groups_text(groups)
        ax.text(0.5, -0.18, group_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=8, color='0.3', clip_on=False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=8)
    if not any_content:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='0.5')
    return panel_discrepancy


def plot_pose_panels(data_kp: Any,
                     frame_idx: int,
                     source: str = 'space',
                     persons: Optional[Sequence[int]] = None,
                     show: bool = True,
                     show_poses: Sequence[bool] = (True, True, True, True),
                     base_height: float = 170.0,
                     figsize: Tuple[float, float] = (19.2, 10.8)):
    """Generate a 5-panel 2D overview of pose keypoints and orientations.

    The X axis is fixed to [-50, 600]; the Y axis is centered at
    c_y = 100 * data_kp['Cam'][frame_idx] with a +/- 250 span.
    """

    coords_key = source + "Coords"
    feat_key = source + "Feat"
    Coords = np.asarray(data_kp[coords_key][frame_idx])
    Feats = data_kp[feat_key][frame_idx]

    fig = plt.figure(figsize=figsize)

    if Coords.ndim != 2 or Coords.shape[1] < 20:
        Warning(f"Expected (m x 20) array for {coords_key}[{frame_idx}], got {Coords.shape}")
        return fig, None
    
    # B = arr[:, 0:24]
    m = Coords.shape[0]
    rows = list(range(m)) if persons is None else list(persons)

    try:
        cam_val = data_kp['Cam'][frame_idx]  # type: ignore[index]
    except Exception:
        cam_val = 0.0
    try:
        cam_float = float(cam_val)
    except (TypeError, ValueError):
        cam_float = 0.0
    c_y = 100.0 * cam_float
    x_limits = (-50.0, 600.0)
    y_limits = (c_y - 250.0, c_y + 250.0)

    clue_groups: Dict[str, Sequence[Sequence[int]]] = {}
    for clue in ('head', 'shoulder', 'hip', 'foot'):
        col = f"{clue}Res"
        try:
            raw_groups = data_kp[col][frame_idx]  # type: ignore[index]
        except Exception:
            raw_groups = None
        clue_groups[clue] = _normalize_groups(raw_groups)

    # Previous-frame groups for change detection
    prev_clue_groups: Dict[str, Optional[Sequence[Sequence[int]]]] = {}
    if frame_idx > 0:
        for clue in ('head', 'shoulder', 'hip', 'foot'):
            col = f"{clue}Res"
            try:
                raw_prev = data_kp[col][frame_idx - 1]  # type: ignore[index]
            except Exception:
                raw_prev = None
            prev_clue_groups[clue] = _normalize_groups(raw_prev)
    else:
        prev_clue_groups = {clue: None for clue in ('head', 'shoulder', 'hip', 'foot')}

    # Determine changed flags per clue and overall to drive [C] markers
    changed: Dict[str, bool] = {}
    for clue in ('head', 'shoulder', 'hip', 'foot'):
        prev = prev_clue_groups.get(clue)
        cur = clue_groups.get(clue)
        try:
            changed[clue] = (prev is not None) and (not equal_groups(cur, prev))
        except Exception:
            changed[clue] = False
    any_changed = any(changed.values())

    try:
        raw_gt = data_kp['GT'][frame_idx]  # type: ignore[index]
    except Exception:
        raw_gt = None
    gt_groups = _normalize_groups(raw_gt)

    show_flags = _normalize_show_flags(show_poses)
    persons_data = list(_iter_person_skeletons(Coords, Feats, rows, base_height))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))
    person_lookup: Dict[int, PersonSkeleton] = {}
    for person in persons_data:
        if person.pid is not None:
            person_lookup[int(person.pid)] = person
        if person.row_index not in person_lookup:
            person_lookup[person.row_index] = person
        adj_key = person.row_index + 1
        if adj_key not in person_lookup:
            person_lookup[adj_key] = person

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    ax_head = fig.add_subplot(gs[0, 0])
    ax_shoulder = fig.add_subplot(gs[0, 1])
    ax_hip = fig.add_subplot(gs[0, 2])
    ax_foot = fig.add_subplot(gs[1, 0])
    ax_all = fig.add_subplot(gs[1, 1])
    ax_image = fig.add_subplot(gs[1, 2])

    head_flag, shoulder_flag, hip_flag, foot_flag = show_flags
    title_head = 'Head pose' + (' [C]' if changed.get('head') else '')
    _plot_pose_panel(ax_head, persons_data, colors, [0, 1], 'head', head_flag, title_head,
                     groups=clue_groups['head'], prev_groups=None, person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.points[0]),
                     clue='head',
                     show_ids=False,
                     x_lim=x_limits, y_lim=y_limits)
    title_shoulder = 'Shoulder pose' + (' [C]' if changed.get('shoulder') else '')
    _plot_pose_panel(ax_shoulder, persons_data, colors, [2, 3], 'shoulder', shoulder_flag,
                     title_shoulder, midpoint_attr='mid_shoulder',
                     groups=clue_groups['shoulder'], prev_groups=None, person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.mid_shoulder),
                     clue='shoulder',
                     show_ids=False,
                     x_lim=x_limits, y_lim=y_limits)
    title_hip = 'Hip pose' + (' [C]' if changed.get('hip') else '')
    _plot_pose_panel(ax_hip, persons_data, colors, [4, 5], 'hip', hip_flag, title_hip,
                     midpoint_attr='mid_hip',
                     groups=clue_groups['hip'], prev_groups=None, person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.mid_hip),
                     clue='hip',
                     show_ids=False,
                     x_lim=x_limits, y_lim=y_limits)
    title_foot = 'Foot pose' + (' [C]' if changed.get('foot') else '')
    _plot_pose_panel(ax_foot, persons_data, colors, [6, 7, 8, 9], 'foot', foot_flag,
                     title_foot, center_attr='foot_center',
                     groups=clue_groups['foot'], prev_groups=None, person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.foot_center),
                     clue='foot',
                     show_ids=False,
                     x_lim=x_limits, y_lim=y_limits)

    title_all = 'All skeletons' + (' [C]' if any_changed else '')
    plot_all_skeletons(
        data_kp,
        frame_idx,
        source=source,
        persons=persons,
        ax=ax_all,
        title=title_all,
        show=False,
        show_poses=show_poses,
        base_height=base_height,
        projection='2d',
        axis_limits=(x_limits, y_limits),
    )

    gt_text = _format_groups_text(gt_groups, prefix='GT: ')

    try:
        cam_val = int(cam_float)
    except (TypeError, ValueError):
        cam_val = 0
    try:
        vid_val = int(data_kp['Vid'][frame_idx])  # type: ignore[index]
    except Exception:
        vid_val = 0
    try:
        seg_val = int(data_kp['Seg'][frame_idx])  # type: ignore[index]
    except Exception:
        seg_val = 0
    try:
        t_val = int(data_kp['Timestamp'][frame_idx])  # type: ignore[index]
    except Exception:
        t_val = frame_idx

    frames_dir = Path(__file__).resolve().parents[2] / 'data' / 'export' / 'frames'
    frame_name = f"frame_{cam_val}{vid_val}{seg_val}_{t_val}.png"
    frame_path = frames_dir / frame_name

    PixelCoords = np.asarray(data_kp['pixelCoords'][frame_idx])
    ax_image.set_title('Frame image')
    ax_image.axis('off')
    try:
        if frame_path.exists():
            img = plt.imread(frame_path)
            ax_image.imshow(img)
            img_h, img_w = img.shape[0], img.shape[1]
            for idx_row, person in enumerate(persons_data):
                if idx_row >= PixelCoords.shape[0]:
                    continue
                head_xy = PixelCoords[idx_row, :2].astype(float)
                if head_xy.shape[0] < 2 or np.any(np.isnan(head_xy)):
                    continue
                px = head_xy[0] * img_w
                py = head_xy[1] * img_h
                ax_image.text(px, py, _person_label(person), color='yellow', fontsize=9,
                              ha='left', va='bottom',
                              bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))
        else:
            ax_image.text(0.5, 0.5, f"Image not found:\n{frame_name}", ha='center', va='center',
                          fontsize=8, color='0.4')
    except Exception as exc:
        ax_image.text(0.5, 0.5, f"Error loading image:\n{exc}", ha='center', va='center',
                      fontsize=8, color='0.4')

    ax_image.text(0.5, -0.18, gt_text, transform=ax_image.transAxes,
                  ha='center', va='top', fontsize=8, color='0.3', clip_on=False)

    axes = [ax_head, ax_shoulder, ax_hip, ax_foot, ax_all, ax_image]
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes

def plot_panels_df(data_kp):
    results_dir = Path(__file__).resolve().parents[2] / 'data' / 'results' / 'panel_plots'
    results_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(data_kp)
    for frame_idx in range(total_frames):
        try:
            cam_val = int(data_kp['Cam'].iloc[frame_idx])
        except Exception:
            cam_val = 0
        try:
            vid_val = int(data_kp['Vid'].iloc[frame_idx])
        except Exception:
            vid_val = 0
        try:
            seg_val = int(data_kp['Seg'].iloc[frame_idx])
        except Exception:
            seg_val = 0
        try:
            timestamp_val = int(data_kp['Timestamp'].iloc[frame_idx])
        except Exception:
            timestamp_val = frame_idx

        filename = f"panel_{cam_val}{vid_val}{seg_val}_{timestamp_val}_{frame_idx}.png"
        fig_path = results_dir / filename
        if not fig_path.exists():
            fig, _ = plot_pose_panels(data_kp=data_kp, frame_idx=frame_idx, show=False)
            fig.savefig(fig_path, dpi=150, bbox_inches=None)
            plt.close(fig)
        else:
            print(f"File {fig_path} already exists")
