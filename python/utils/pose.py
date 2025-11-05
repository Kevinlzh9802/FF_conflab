from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
# pd.options.mode.copy_on_write = True

def back_project(frame_data: np.ndarray,
                 K: np.ndarray,
                 R: np.ndarray,
                 t: np.ndarray,
                 dist_coeffs: Sequence[float],
                 body_height: float,
                 img_size: Sequence[float],
                 height_ratios_map: Dict[str, float],
                 part_column_map: Dict[str, Sequence[int]]):
    """Backproject pixel points onto a horizontal plane at specified height.

    Returns array of shape (num_people, 2, num_parts) with X,Y per part.
    """
    num_people = frame_data.shape[0]
    parts = list(height_ratios_map.keys())
    num_parts = len(parts)
    worldXY_all = np.zeros((num_people, 2, num_parts), dtype=float)
    Rinv = R.T
    camCenter = -Rinv @ t.reshape(3, 1)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    img_w, img_h = float(img_size[0]), float(img_size[1])
    k1, k2, p1, p2, k3 = dist_coeffs
    for p in range(num_people):
        for pi, name in enumerate(parts):
            cols = part_column_map[name]
            u = frame_data[p, cols[0]] * img_w
            v = frame_data[p, cols[1]] * img_h
            x = (u - cx) / fx
            y = (v - cy) / fy
            r2 = x * x + y * y
            x_dist = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            y_dist = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            dir_cam = np.array([x_dist, y_dist, 1.0])
            dir_world = Rinv @ dir_cam.reshape(3, 1)
            Z = float(height_ratios_map[name]) * float(body_height)
            lam = (Z - camCenter[2, 0]) / dir_world[2, 0]
            world_point = camCenter[:, 0] + lam * dir_world[:, 0]
            worldXY_all[p, :, pi] = world_point[0:2]
    return worldXY_all

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
