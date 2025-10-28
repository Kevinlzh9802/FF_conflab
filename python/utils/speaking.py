from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
from groups import filter_group_by_members
from scipy.io import loadmat


def read_speaking_status(data_path):
    mat = loadmat(data_path, squeeze_me=True, struct_as_record=False)
    S = mat['speaking_status']
    speaking = {fn: getattr(S.speaking, fn) for fn in S.index.speaking_keys}
    confidence = {fn: getattr(S.confidence, fn) for fn in S.index.confidence_keys}
    return {'speaking': speaking, 'confidence': confidence}


def get_speaking_status_single(sp_status: Dict[str, Any], vid: int, seg: int, n: int, window: int):
    """Read speaking status row(s) from a nested dict structure similar to MATLAB.

    sp_status['speaking'] contains keys like 'vidX_segy' mapping to arrays.
    Returns (sp, cf) arrays or -1000 on failure.
    """
    key = f"vid{vid}_seg{seg}"
    try:
        sp = np.asarray(sp_status['speaking'][key])
        cf = np.asarray(sp_status['confidence'][key])
    except Exception:
        return -1000, -1000
    if window <= 1:
        try:
            return sp[n, :], cf[n, :]
        except Exception:
            return -1000, -1000
    else:
        w_start = n - int(round(window * 0.5))
        w_end = w_start + window
        try:
            return sp[w_start:w_end, :].mean(axis=0), cf[w_start:w_end, :].mean(axis=0)
        except Exception:
            return -1000, -1000


def merge_speaking_status(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    idsA, dataA = A[0, :], A[1:, :]
    idsB, dataB = B[0, :], B[1:, :]
    all_ids = np.unique(np.concatenate([idsA, idsB]).astype(int))
    mergedA = np.full((dataA.shape[0], all_ids.size), np.nan)
    mergedB = np.full((dataB.shape[0], all_ids.size), np.nan)
    locA = {int(pid): i for i, pid in enumerate(idsA)}
    locB = {int(pid): i for i, pid in enumerate(idsB)}
    for j, pid in enumerate(all_ids):
        if int(pid) in locA:
            mergedA[:, j] = dataA[:, locA[int(pid)]]
        if int(pid) in locB:
            mergedB[:, j] = dataB[:, locB[int(pid)]]
    merged = np.vstack([all_ids.reshape(1, -1), mergedA, mergedB])
    return merged


def get_status_for_group(person: Sequence[int], status: Sequence[float], group: Sequence[Sequence[int]]):
    missing_value = -1000
    status_group: List[np.ndarray] = []
    idx_map = {int(p): i for i, p in enumerate(person)}
    for g in group:
        values = np.full((len(g),), missing_value, dtype=float)
        for i, pid in enumerate(g):
            if int(pid) in idx_map:
                values[i] = status[idx_map[int(pid)]]
        status_group.append(values)
    return status_group


def collect_matching_groups(vector: Sequence[int], ts: Iterable[int], vid: int, cam: int,
                            results, feat_name: str, speaking_status: Dict[int, np.ndarray], window_len: int):
    combinedGroups: List[List[Any]] = [[], [], [], []]
    feat_res = f"{feat_name}Res"
    ss = speaking_status[vid]
    matched = [i for i, ok in enumerate((results['concat_ts'] == ts) & (results['Vid'] == vid) & (results['Cam'] == cam)) if ok]
    id_list = ss[0, :]
    speaking_data = ss[1:, :]
    max_t = speaking_data.shape[0]
    half_win = window_len // 2
    for idx in matched:
        t = results['concat_ts'][idx]
        groups = results[feat_res][idx]
        matched_groups = filter_group_by_members(vector, groups)
        # original speaking status over window for these IDs
        _, col_inds = np.unique([np.where(id_list == pid)[0][0] for pid in vector], return_index=True)
        t_start = max(1, t - half_win)
        t_end = min(max_t, t + half_win)
        orig_status = speaking_data[t_start:t_end, col_inds].mean(axis=0)
        split_status = []
        simu_sp = []
        for g in matched_groups:
            cols = [np.where(id_list == pid)[0][0] for pid in g]
            vals = speaking_data[t_start:t_end, cols].mean(axis=0)
            split_status.append(vals)
            simu_sp.append(vals.sum())
        combinedGroups[0].append(matched_groups)
        combinedGroups[1].append(orig_status)
        combinedGroups[2].append(split_status)
        combinedGroups[3].append(np.array(simu_sp))
    return combinedGroups


# Complex window aggregation across cameras is outside current scope.
def count_speaker_groups(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError("countSpeakerGroups requires project-specific table structures; TODO later.")


if __name__ == '__main__': 
    read_speaking_status('../data/export/speaking_status_py.mat')