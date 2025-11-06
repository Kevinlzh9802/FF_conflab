from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, List, Sequence, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
# pd.options.mode.copy_on_write = True

SKELETON_THRES = 100
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
    exp_head_pose: float
    is_missing: Optional[List[bool]] = None
    is_illed: Optional[List[bool]] = None

    FEATURE_ORDER: ClassVar[Tuple[str, ...]] = ('head', 'shoulder', 'hip', 'foot')
    FEATURE_KEYPOINTS: ClassVar[Dict[str, Tuple[int, ...]]] = {
        'head': (0, 1),
        'shoulder': (2, 3),
        'hip': (4, 5),
        'foot': (6, 7, 8, 9),
    }

    def _get_xy(self, idx: int) -> np.ndarray:
        try:
            xy = np.asarray(self.points[idx, :2], dtype=float)
        except (IndexError, TypeError, ValueError):
            return np.array([np.nan, np.nan], dtype=float)
        return xy

    def _is_keypoint_missing(self, idx: int) -> bool:
        xy = self._get_xy(idx)
        return bool(np.any(np.vstack([np.isnan(xy), xy==0])))

    # @property
    def missing_features(self) -> List[bool]:
        flags: List[bool] = []
        for feature in self.FEATURE_ORDER:
            indices = self.FEATURE_KEYPOINTS[feature]
            flags.append(any(self._is_keypoint_missing(i) for i in indices))
        self.is_missing = flags
        return flags

    def _pair_length(self, idx_a: int, idx_b: int) -> float:
        a_xy = self._get_xy(idx_a)
        b_xy = self._get_xy(idx_b)
        if np.any(np.isnan(a_xy)) or np.any(np.isnan(b_xy)):
            return float('nan')
        return float(np.linalg.norm(b_xy - a_xy))

    def feature_lengths(self) -> np.ndarray:
        head_len = self._pair_length(0, 1)
        shoulder_len = self._pair_length(2, 3)
        hip_len = self._pair_length(4, 5)
        ankle_len = self._pair_length(6, 7)
        foot_len = self._pair_length(8, 9)
        if np.isnan(ankle_len) and np.isnan(foot_len):
            combined_foot = float('nan')
        elif np.isnan(ankle_len):
            combined_foot = foot_len
        elif np.isnan(foot_len):
            combined_foot = ankle_len
        else:
            combined_foot = max(ankle_len, foot_len)
        return np.array([head_len, shoulder_len, hip_len, combined_foot], dtype=float)

    def illed_features(self, thresholds: Union[Sequence[float], float]) -> List[bool]:
        lengths = self.feature_lengths()
        if np.isscalar(thresholds):
            thresh_arr = np.full(len(self.FEATURE_ORDER), float(thresholds), dtype=float)
        else:
            thresh_arr = np.asarray(thresholds, dtype=float)
            if thresh_arr.shape[0] != len(self.FEATURE_ORDER):
                raise ValueError(f"thresholds must have length {len(self.FEATURE_ORDER)}")
        flags: List[bool] = []
        for length, limit in zip(lengths, thresh_arr):
            if np.isnan(length) or np.isnan(limit):
                flags.append(False)
            else:
                flags.append(length > limit)
        self.is_illed = flags
        return flags

@dataclass
class PoseArrow:
    start: np.ndarray
    vec: np.ndarray
    kind: str

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

def mid_point(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    if any(np.isnan([X[idx_a], Y[idx_a], Z[idx_a], X[idx_b], Y[idx_b], Z[idx_b]])):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.array([
        (X[idx_a] + X[idx_b]) / 2.0,
        (Y[idx_a] + Y[idx_b]) / 2.0,
        (Z[idx_a] + Z[idx_b]) / 2.0,
    ], dtype=float)


def compute_foot_properties(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def build_person_skeletons(coords_entries: Iterable[Any],
                           base_height: float = 170.0,
                           head_feats: Optional[Any] = None,
                           row_indices: Optional[Sequence[int]] = None) -> List[PersonSkeleton]:
    skeletons: List[PersonSkeleton] = []
    if coords_entries is None:
        return skeletons

    coords_list = list(coords_entries)
    if row_indices is None:
        row_index_list = list(range(len(coords_list)))
    else:
        row_index_list = list(row_indices)

    head_array: Optional[np.ndarray]
    if head_feats is None:
        head_array = None
    else:
        try:
            head_array = np.asarray(head_feats, dtype=float)
            if head_array.ndim != 2:
                head_array = None
        except Exception:
            head_array = None

    for idx, entry in enumerate(coords_list):
        if entry is None:
            continue
        arr = np.asarray(entry, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                continue
            arr = arr.reshape(-1, 2)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] < 2:
            continue
        xy = arr[:, :2]
        if xy.shape[0] < 10:
            pad = np.full((10 - xy.shape[0], 2), np.nan)
            xy = np.vstack([xy, pad])
        elif xy.shape[0] > 10:
            xy = xy[:10]

        X = xy[:, 0]
        Y = xy[:, 1]
        Z = base_height * SKELETON_Z_COEFS.copy()
        missing_mask = np.isnan(X) | np.isnan(Y)
        Z[missing_mask] = np.nan
        if np.all(missing_mask):
            continue

        mid_shoulder = mid_point(X, Y, Z, 2, 3)
        mid_hip = mid_point(X, Y, Z, 4, 5)
        foot_center, foot_vector = compute_foot_properties(X, Y, Z)
        points = np.column_stack((X, Y, Z))

        pid = None
        exp_head_pose = float('nan')
        if head_array is not None and idx < head_array.shape[0]:
            pid = _safe_int(head_array[idx, 0])
            if head_array.shape[1] > 3:
                try:
                    exp_head_pose = float(head_array[idx, 3])
                except (TypeError, ValueError):
                    exp_head_pose = float('nan')

        row_index = row_index_list[idx] if idx < len(row_index_list) else idx

        skeletons.append(
            PersonSkeleton(
                row_index=row_index,
                pid=pid,
                points=points,
                mid_shoulder=mid_shoulder,
                mid_hip=mid_hip,
                foot_center=foot_center,
                foot_vector=foot_vector,
                exp_head_pose=exp_head_pose,
            )
        )

    return skeletons
