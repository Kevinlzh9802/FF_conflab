from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
# pd.options.mode.copy_on_write = True

def get_foot_orientation(coords: Sequence[float], is_left_handed: bool) -> Tuple[float, Tuple[float, float]]:
    arr = np.asarray(coords, dtype=float).copy()
    arr[arr == 0] = np.nan
    lx, ly, rx, ry, lfx, lfy, rfx, rfy = arr
    left_theta = np.nan
    right_theta = np.nan
    if np.all(~np.isnan([lx, ly, lfx, lfy])):
        dy = lfy - ly
        dx = lfx - lx
        left_theta = np.arctan2(dy, dx)
    if np.all(~np.isnan([rx, ry, rfx, rfy])):
        dy = rfy - ry
        dx = rfx - rx
        right_theta = np.arctan2(dy, dx)
    if not np.isnan(left_theta) and not np.isnan(right_theta):
        theta = np.arctan2(np.mean([np.sin(left_theta), np.sin(right_theta)]), np.mean([np.cos(left_theta), np.cos(right_theta)]))
    elif not np.isnan(left_theta):
        theta = left_theta
    elif not np.isnan(right_theta):
        theta = right_theta
    else:
        theta = np.nan
    xs = np.array([lx, rx, lfx, rfx])
    ys = np.array([ly, ry, lfy, rfy])
    valid = ~np.isnan(xs) & ~np.isnan(ys)
    if np.any(valid):
        pos = (float(xs[valid].mean()), float(ys[valid].mean()))
    else:
        pos = (float('nan'), float('nan'))
    return float(theta), pos


def process_foot_data(T):
    """Process a table-like object T with column 'footFeat'.

    For each row, expects an n x 48 features array. Adjusts columns [2:3] and [4]
    using foot orientation and position.
    """
    
    for i in range(len(T)):
        features = T['footFeat'][i]
        if features is None or features.shape[1] == 0:
            continue
        M1 = features[:, 0:24].copy()
        M2 = features[:, 24:48].copy()
        for j in range(M1.shape[0]):
            theta1, pos1 = get_foot_orientation(M1[j, 16:24], True)
            M1[j, 1:3] = np.array(pos1) * np.array([1920.0, 1080.0])
            M1[j, 3] = theta1
            theta2, pos2 = get_foot_orientation(M2[j, 16:24], False)
            M2[j, 1:3] = np.array(pos2)
            M2[j, 3] = theta2
        T['footFeat'][i] = np.hstack([M1, M2])
    return T


def read_pose_info(info: Dict[str, Any], person: Sequence[int]):  # pragma: no cover - project specific I/O
    """Placeholder for reading pose info from JSON.

    TODO: implement if/when input format is defined in Python environment.
    """
    raise NotImplementedError("read_pose_info depends on external JSON files; implement in your environment.")

