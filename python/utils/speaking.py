from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import os
import re
import pickle
import pandas as pd
from utils.groups import filter_group_by_members
from scipy.io import loadmat


def write_sp_merged(read_path, write_path):
    """Read exported speaking_status .mat and write merged per-video pickles.

    - read_path: path to .mat produced by GCFF/convertData.m (contains struct 'speaking_status')
    - write_path: path to .pkl to write; saved object is a dict:
        {
          'speaking': { vid: merged_matrix },
          'confidence': { vid: merged_matrix }
        }
      where merged_matrix has shape ((1 + total_time), num_ids)
      with first row the ID list and subsequent rows the concatenated timeline.
    """
    ss = read_speaking_status(read_path)
    speaking = ss['speaking']
    confidence = ss['confidence']

    # Collect vids from keys like 'vid2_seg8'
    def parse_key(k: str):
        m = re.match(r"vid(\d+)_seg(\d+)$", k)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    # Map vid -> list of (seg, key)
    vids: Dict[int, List[Tuple[int, str]]] = {}
    for k in speaking.keys():
        p = parse_key(k)
        if p is None:
            continue
        vid, seg = p
        vids.setdefault(vid, []).append((seg, k))

    merged_sp: Dict[int, np.ndarray] = {}
    merged_cf: Dict[int, np.ndarray] = {}

    for vid, pairs in vids.items():
        # sort by seg increasing
        pairs.sort(key=lambda x: x[0])
        # Merge speaking
        sp_arr = np.asarray(speaking[pairs[0][1]])
        for _, key in pairs[1:]:
            sp_arr = merge_speaking_status(sp_arr, np.asarray(speaking[key]))
        # Merge confidence
        cf_arr = np.asarray(confidence[pairs[0][1]])
        for _, key in pairs[1:]:
            cf_arr = merge_speaking_status(cf_arr, np.asarray(confidence[key]))
        # Optional: ensure both have identical ID rows
        if sp_arr.shape[1] != cf_arr.shape[1] or not np.array_equal(sp_arr[0, :], cf_arr[0, :]):
            # reconcile by union of IDs
            ids_sp = sp_arr[0, :].astype(int)
            ids_cf = cf_arr[0, :].astype(int)
            all_ids = np.unique(np.concatenate([ids_sp, ids_cf]))
            # reindex helper
            def reindex(arr, all_ids):
                ids = arr[0, :].astype(int)
                data = arr[1:, :]
                out = np.full((data.shape[0], all_ids.size), np.nan)
                loc = {int(pid): i for i, pid in enumerate(ids)}
                for j, pid in enumerate(all_ids):
                    if int(pid) in loc:
                        out[:, j] = data[:, loc[int(pid)]]
                return np.vstack([all_ids.reshape(1, -1), out])
            sp_arr = reindex(sp_arr, all_ids)
            cf_arr = reindex(cf_arr, all_ids)
        merged_sp[vid] = sp_arr.astype(np.float32)
        merged_cf[vid] = cf_arr.astype(np.float32)

    out = {'speaking': merged_sp, 'confidence': merged_cf}
    os.makedirs(os.path.dirname(write_path) or '.', exist_ok=True)
    with open(write_path, 'wb') as f:
        pickle.dump(out, f)

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


if __name__ == '__main__': 
    # TODO: make the two arguments configurable in config.yaml
    write_sp_merged('../data/export/speaking_status_py.mat', '../data/export/sp_merged.pkl')
