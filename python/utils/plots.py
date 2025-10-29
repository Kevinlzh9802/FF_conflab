from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


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


SKELETON_Z_COEFS = np.array([1.0, 0.95, 0.85, 0.85, 0.5, 0.5, 0.02, 0.02, 0.02, 0.02], dtype=float)


@dataclass
class PersonSkeleton:
    row_index: int
    pid: Optional[int]
    points: np.ndarray
    mid_shoulder: np.ndarray
    mid_hip: np.ndarray
    foot_center: np.ndarray
    foot_vector: Optional[np.ndarray]


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


def _mid_point(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    if any(np.isnan([X[idx_a], Y[idx_a], Z[idx_a], X[idx_b], Y[idx_b], Z[idx_b]])):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.array([
        (X[idx_a] + X[idx_b]) / 2.0,
        (Y[idx_a] + Y[idx_b]) / 2.0,
        (Z[idx_a] + Z[idx_b]) / 2.0,
    ], dtype=float)


def _compute_foot_properties(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    idxs_center = [6, 7, 8, 9]
    valid_center = [i for i in idxs_center if not (np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i]))]
    if valid_center:
        cx = float(np.nanmean([X[i] for i in valid_center]))
        cy = float(np.nanmean([Y[i] for i in valid_center]))
        cz = float(np.nanmean([Z[i] for i in valid_center]))
        center = np.array([cx, cy, cz], dtype=float)
    else:
        center = np.array([np.nan, np.nan, np.nan], dtype=float)

    def valid_xy(idx: int) -> bool:
        return not (np.isnan(X[idx]) or np.isnan(Y[idx]))

    vecs = []
    if valid_xy(6) and valid_xy(7):
        ax_vx = X[7] - X[6]
        ax_vy = Y[7] - Y[6]
        vecs.append(np.array([-ax_vy, ax_vx], dtype=float))
    if valid_xy(8) and valid_xy(9):
        ft_vx = X[9] - X[8]
        ft_vy = Y[9] - Y[8]
        vecs.append(np.array([-ft_vy, ft_vx], dtype=float))

    foot_vector = None
    if vecs:
        stack = np.stack(vecs, axis=0)
        v_avg = np.nanmean(stack, axis=0)
        if not np.any(np.isnan(v_avg)) and not np.allclose(v_avg, 0.0):
            foot_vector = v_avg.astype(float)
    return center, foot_vector


def _iter_person_skeletons(A: np.ndarray, rows: Sequence[int], base_height: float) -> Iterable[PersonSkeleton]:
    for r in rows:
        pts2 = _extract_xy_from_headfeat_A(A[r, :])  # (10, 2)
        points_xy = pts2.astype(float)
        X_all = points_xy[:, 0]
        Y_all = points_xy[:, 1]
        Z_all = base_height * SKELETON_Z_COEFS.copy()
        missing = np.isnan(X_all) | np.isnan(Y_all)
        Z_all[missing] = np.nan
        if np.all(missing):
            continue
        mid_shoulder = _mid_point(X_all, Y_all, Z_all, 2, 3)
        mid_hip = _mid_point(X_all, Y_all, Z_all, 4, 5)
        foot_center, foot_vector = _compute_foot_properties(X_all, Y_all, Z_all)
        points = np.column_stack((X_all, Y_all, Z_all))

        pid = None
        if A.shape[1] > 0:
            pid_raw = A[r, 0]
            try:
                if not np.isnan(pid_raw):
                    pid = int(pid_raw)
            except TypeError:
                if isinstance(pid_raw, (int, np.integer)):
                    pid = int(pid_raw)
            except ValueError:
                pass

        yield PersonSkeleton(
            row_index=r,
            pid=pid,
            points=points,
            mid_shoulder=mid_shoulder,
            mid_hip=mid_hip,
            foot_center=foot_center,
            foot_vector=foot_vector,
        )


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
    for g in groups:
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


def _draw_group_polygons(ax,
                         groups: Sequence[Sequence[int]],
                         person_lookup: Dict[int, PersonSkeleton],
                         point_selector):
    if not groups:
        return
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(groups), 1)))
    for idx, members in enumerate(groups):
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
        pts_arr = np.asarray(pts, dtype=float)
        color = colors[idx % len(colors)]
        if pts_arr.shape[0] >= 3:
            poly = Polygon(pts_arr, closed=True, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.add_patch(poly)
        elif pts_arr.shape[0] == 2:
            ax.plot(pts_arr[:, 0], pts_arr[:, 1], color=color, linewidth=2, linestyle='--')
        else:
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], color=color, s=80, facecolors='none', edgecolors=color, linewidth=1.5)


def _get_pose_arrow(person: PersonSkeleton, pose: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    pts = person.points
    if pose == 'head':
        if np.any(np.isnan(pts[[0, 1], :2])):
            return None, None
        start = pts[0]
        vec = np.array([pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]], dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return None, None
        return start, vec
    if pose == 'shoulder':
        if np.any(np.isnan(person.mid_shoulder)) or np.any(np.isnan(pts[[2, 3], :2])):
            return None, None
        vx = pts[3, 0] - pts[2, 0]
        vy = pts[3, 1] - pts[2, 1]
        vec = np.array([-vy, vx], dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return None, None
        return person.mid_shoulder, vec
    if pose == 'hip':
        if np.any(np.isnan(person.mid_hip)) or np.any(np.isnan(pts[[4, 5], :2])):
            return None, None
        vx = pts[5, 0] - pts[4, 0]
        vy = pts[5, 1] - pts[4, 1]
        vec = np.array([-vy, vx], dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return None, None
        return person.mid_hip, vec
    if pose == 'foot':
        if person.foot_vector is None or np.any(np.isnan(person.foot_center[:2])):
            return None, None
        vec = np.asarray(person.foot_vector, dtype=float)
        if np.any(np.isnan(vec)) or np.allclose(vec, 0.0):
            return None, None
        return person.foot_center, vec
    return None, None


def _draw_arrow(ax, start: Optional[np.ndarray], vec: Optional[np.ndarray], projection: str,
                color, x_vals: Optional[list] = None, y_vals: Optional[list] = None):
    if start is None or vec is None:
        return
    sx, sy = float(start[0]), float(start[1])
    vx, vy = float(vec[0]), float(vec[1])
    if any(np.isnan([sx, sy, vx, vy])):
        return
    if np.allclose([vx, vy], [0.0, 0.0]):
        return
    if projection == '3d':
        sz = float(start[2])
        if np.isnan(sz):
            return
        try:
            ax.quiver(sx, sy, sz, vx, vy, 0.0, color=color, linewidth=1.5, arrow_length_ratio=0.2)
        except TypeError:
            ax.quiver(sx, sy, sz, vx, vy, 0.0, color=color, linewidth=1.5)
    else:
        ax.quiver(sx, sy, vx, vy, color=color, angles='xy', scale_units='xy', scale=1.0, width=0.005)
        if x_vals is not None:
            x_vals.extend([sx, sx + vx])
        if y_vals is not None:
            y_vals.extend([sy, sy + vy])


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

    show_head, show_shoulder, show_hip, show_foot = show_flags
    if show_head:
        start, vec = _get_pose_arrow(person, 'head')
        _draw_arrow(ax, start, vec, projection, color, x_vals, y_vals)
    if show_shoulder:
        start, vec = _get_pose_arrow(person, 'shoulder')
        _draw_arrow(ax, start, vec, projection, color, x_vals, y_vals)
    if show_hip:
        start, vec = _get_pose_arrow(person, 'hip')
        _draw_arrow(ax, start, vec, projection, color, x_vals, y_vals)
    if show_foot:
        start, vec = _get_pose_arrow(person, 'foot')
        _draw_arrow(ax, start, vec, projection, color, x_vals, y_vals)

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


def _extract_xy_from_headfeat_A(A_row: np.ndarray) -> np.ndarray:
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
    # zero-based indices in A
    idx_pairs = [
        (4, 5),   # head
        (6, 7),   # nose
        (8, 9),   # left shoulder
        (10, 11), # right shoulder
        (12, 13), # left hip
        (14, 15), # right hip
        (16, 17), # left ankle
        (18, 19), # right ankle
        (20, 21), # left foot
        (22, 23), # right foot
    ]
    pts = []
    for ix, iy in idx_pairs:
        x = A_row[ix] if ix < A_row.shape[0] else np.nan
        y = A_row[iy] if iy < A_row.shape[0] else np.nan
        pts.append((x, y))
    return np.asarray(pts, dtype=float)



def plot_all_skeletons(data_kp: Any,
                       frame_idx: int,
                       key: str = 'headFeat',
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

    plot_title = title or f"HeadFeat (XY) - {key}[{frame_idx}]"
    projection = projection.lower()
    if projection not in ('2d', '3d'):
        raise ValueError(f"projection must be '2d' or '3d', got {projection}")

    arr = np.asarray(data_kp[key][frame_idx])
    if arr.ndim != 2 or arr.shape[1] < 48:
        raise ValueError(f"Expected (m x 48) array for {key}[{frame_idx}], got {arr.shape}")
    A = arr[:, 24:48]
    m = A.shape[0]
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
    persons_data = list(_iter_person_skeletons(A, rows, base_height))
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
                     person_lookup: Optional[Dict[int, PersonSkeleton]] = None,
                     point_selector=None,
                     x_lim: Optional[Tuple[float, float]] = None,
                     y_lim: Optional[Tuple[float, float]] = None):
    x_vals: list = []
    y_vals: list = []
    labels_added = set()
    any_content = False

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
        if show_flag and arrow_key:
            start, vec = _get_pose_arrow(person, arrow_key)
            if start is not None and vec is not None:
                _draw_arrow(ax, start, vec, '2d', col, x_vals, y_vals)
                any_content = True

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    if x_lim is not None and y_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_aspect('equal', adjustable='box')
    else:
        _finalize_2d_axis(ax, x_vals, y_vals)
    if groups and person_lookup and point_selector:
        _draw_group_polygons(ax, groups, person_lookup, point_selector)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=8)
    if not any_content:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='0.5')


def plot_pose_panels(data_kp: Any,
                     frame_idx: int,
                     key: str = 'headFeat',
                     persons: Optional[Sequence[int]] = None,
                     show: bool = True,
                     show_poses: Sequence[bool] = (True, True, True, True),
                     base_height: float = 170.0,
                     figsize: Tuple[int, int] = (14, 8)):
    """Generate a 5-panel 2D overview of pose keypoints and orientations.

    The X axis is fixed to [-50, 450]; the Y axis is centered at
    c_y = 100 * data_kp['Cam'][frame_idx] with a +/- 250 span.
    """

    arr = np.asarray(data_kp[key][frame_idx])
    if arr.ndim != 2 or arr.shape[1] < 48:
        raise ValueError(f"Expected (m x 48) array for {key}[{frame_idx}], got {arr.shape}")
    A = arr[:, 24:48]
    m = A.shape[0]
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
    x_limits = (-50.0, 450.0)
    y_limits = (c_y - 250.0, c_y + 250.0)

    clue_groups: Dict[str, Sequence[Sequence[int]]] = {}
    for clue in ('head', 'shoulder', 'hip', 'foot'):
        col = f"{clue}Res"
        try:
            raw_groups = data_kp[col][frame_idx]  # type: ignore[index]
        except Exception:
            raw_groups = None
        clue_groups[clue] = _normalize_groups(raw_groups)

    show_flags = _normalize_show_flags(show_poses)
    persons_data = list(_iter_person_skeletons(A, rows, base_height))
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

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    ax_head = fig.add_subplot(gs[0, 0])
    ax_shoulder = fig.add_subplot(gs[0, 1])
    ax_hip = fig.add_subplot(gs[0, 2])
    ax_foot = fig.add_subplot(gs[1, 0])
    ax_all = fig.add_subplot(gs[1, 1:])

    head_flag, shoulder_flag, hip_flag, foot_flag = show_flags
    _plot_pose_panel(ax_head, persons_data, colors, [0, 1], 'head', head_flag, 'Head pose',
                     groups=clue_groups['head'], person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.points[0]),
                     x_lim=x_limits, y_lim=y_limits)
    _plot_pose_panel(ax_shoulder, persons_data, colors, [2, 3], 'shoulder', shoulder_flag,
                     'Shoulder pose', midpoint_attr='mid_shoulder',
                     groups=clue_groups['shoulder'], person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.mid_shoulder),
                     x_lim=x_limits, y_lim=y_limits)
    _plot_pose_panel(ax_hip, persons_data, colors, [4, 5], 'hip', hip_flag, 'Hip pose',
                     midpoint_attr='mid_hip',
                     groups=clue_groups['hip'], person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.mid_hip),
                     x_lim=x_limits, y_lim=y_limits)
    _plot_pose_panel(ax_foot, persons_data, colors, [6, 7, 8, 9], 'foot', foot_flag,
                     'Foot pose', center_attr='foot_center',
                     groups=clue_groups['foot'], person_lookup=person_lookup,
                     point_selector=lambda p: _safe_xy(p.foot_center),
                     x_lim=x_limits, y_lim=y_limits)

    plot_all_skeletons(
        data_kp,
        frame_idx,
        key=key,
        persons=persons,
        ax=ax_all,
        title='All skeletons',
        show=False,
        show_poses=show_poses,
        base_height=base_height,
        projection='2d',
        axis_limits=(x_limits, y_limits),
    )

    axes = [ax_head, ax_shoulder, ax_hip, ax_foot, ax_all]
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
