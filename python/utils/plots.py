from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def plot_all_skeletons_3d(data_kp: Any,
                    frame_idx: int,
                    key: str = 'headFeat',
                    persons: Optional[Sequence[int]] = None,
                    ax=None,
                    title: Optional[str] = None,
                    show: bool = True,
                    base_height: float = 170.0):
    """Plot 3D scatter of 2D keypoints (Z=0) extracted from the right 24 columns of feature set.

    - data_kp[key][frame_idx] must be an (m x 48) array. We take A = [:, 24:48] (0-based slicing)
      and map indices per MATLAB description (subtract 1 already handled here).
    - persons: optional indices of rows (people) to plot; default plots all.
    - Z coordinate is set using a simple height model: Z = base_height * coef, with
      coefficients per keypoint name:
        head:1, nose:0.95, leftShoulder/rightShoulder:0.85, leftHip/rightHip:0.5,
        leftAnkle/rightAnkle:0.02, leftFoot/rightFoot:0.02
    """

    arr = np.asarray(data_kp[key][frame_idx])
    if arr.ndim != 2 or arr.shape[1] < 48:
        raise ValueError(f"Expected (m x 48) array for {key}[{frame_idx}], got {arr.shape}")
    A = arr[:, 24:48]
    m = A.shape[0]
    rows = list(range(m)) if persons is None else list(persons)

    fig, ax = _setup_3d(ax, title or f"HeadFeat (XY) - {key}[{frame_idx}]")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))
    any_pts = False
    # Coefficients aligned with _extract_xy_from_headfeat_A order
    coefs = np.array([1.0, 0.95, 0.85, 0.85, 0.5, 0.5, 0.02, 0.02, 0.02, 0.02], dtype=float)
    for ci, r in enumerate(rows):
        pts2 = _extract_xy_from_headfeat_A(A[r, :])  # (10,2)
        # Full arrays with NaNs for missing
        X_all = pts2[:, 0].astype(float)
        Y_all = pts2[:, 1].astype(float)
        Z_all = base_height * coefs.copy()
        missing = np.isnan(X_all) | np.isnan(Y_all)
        Z_all[missing] = np.nan
        if np.all(missing):
            continue
        any_pts = True

        # Compute midpoints: midShoulder between left(2) and right(3) shoulder, midHip between hips(4,5)
        def mid_point(idx_a: int, idx_b: int):
            if not (np.isnan(X_all[idx_a]) or np.isnan(X_all[idx_b]) or np.isnan(Y_all[idx_a]) or np.isnan(Y_all[idx_b])):
                return (X_all[idx_a] + X_all[idx_b]) / 2.0, (Y_all[idx_a] + Y_all[idx_b]) / 2.0, (Z_all[idx_a] + Z_all[idx_b]) / 2.0
            return (np.nan, np.nan, np.nan)

        midS = mid_point(2, 3)   # mid-shoulder
        midH = mid_point(4, 5)   # mid-hip

        # Scatter keypoints and midpoints
        col = tuple(colors[ci % len(colors)])
        # Person ID from first column of A (if available)
        pid = A[r, 0] if A.shape[1] > 0 else np.nan
        try:
            pid_int = int(pid) if not np.isnan(pid) else None
        except Exception:
            pid_int = None
        legend_label = f"id{pid_int}" if pid_int is not None else f"p{r}"
        mask_pts = ~np.isnan(X_all) & ~np.isnan(Y_all) & ~np.isnan(Z_all)
        ax.scatter(X_all[mask_pts], Y_all[mask_pts], Z_all[mask_pts], s=20, color=col, label=legend_label)
        if not np.any(np.isnan(midS)):
            ax.scatter([midS[0]], [midS[1]], [midS[2]], s=20, color=col)
        if not np.any(np.isnan(midH)):
            ax.scatter([midH[0]], [midH[1]], [midH[2]], s=20, color=col)

        # Helper to draw a line if both endpoints are valid
        def pl(a, b):
            if (not np.any(np.isnan(a))) and (not np.any(np.isnan(b))):
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col, linewidth=2)

        # Assemble points
        P = [
            (X_all[1], Y_all[1], Z_all[1]),  # nose
            (X_all[0], Y_all[0], Z_all[0]),  # head
            midS,                            # mid-shoulder
            (X_all[2], Y_all[2], Z_all[2]),  # left shoulder
            (X_all[3], Y_all[3], Z_all[3]),  # right shoulder
            midH,                            # mid-hip
            (X_all[4], Y_all[4], Z_all[4]),  # left hip
            (X_all[5], Y_all[5], Z_all[5]),  # right hip
            (X_all[6], Y_all[6], Z_all[6]),  # left ankle
            (X_all[7], Y_all[7], Z_all[7]),  # right ankle
            (X_all[8], Y_all[8], Z_all[8]),  # left foot
            (X_all[9], Y_all[9], Z_all[9]),  # right foot
        ]
        # Bone connections:
        # [nose,head]
        pl(P[0], P[1])
        # [head,midShoulder]
        pl(P[1], P[2])
        # [midShoulder, leftShoulder], [midShoulder, rightShoulder]
        pl(P[2], P[3]); pl(P[2], P[4])
        # [midShoulder, midHip]
        pl(P[2], P[5])
        # [midHip, leftHip], [midHip, rightHip]
        pl(P[5], P[6]); pl(P[5], P[7])
        # [leftHip-leftAnkle-leftFoot]
        pl(P[6], P[8]); pl(P[8], P[10])
        # [rightHip-rightAnkle-rightFoot]
        pl(P[7], P[9]); pl(P[9], P[11])
    if any_pts:
        # try to set equal aspect from all plotted points
        # collect limits from current data
        xdata = np.concatenate([c._offsets3d[0] for c in ax.collections]) if ax.collections else np.array([0])
        ydata = np.concatenate([c._offsets3d[1] for c in ax.collections]) if ax.collections else np.array([0])
        zdata = np.concatenate([c._offsets3d[2] for c in ax.collections]) if ax.collections else np.array([0])
        _set_equal_3d(ax, np.asarray(xdata), np.asarray(ydata), np.asarray(zdata))
        ax.legend(loc='upper right', fontsize=8)
    if show and fig is not None:
        plt.show()
    return ax
