from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


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

