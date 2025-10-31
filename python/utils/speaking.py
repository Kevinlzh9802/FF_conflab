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
        merged_sp[vid] = sp_arr
        merged_cf[vid] = cf_arr

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


def groups_speaker_belongs_clues(row, speakers: List):
    group_nums = {}
    for clue in ['headRes', 'shoulderRes', 'hipRes', 'footRes']:
        detection = row.get(clue, 0)
        group_nums[clue] = groups_speaker_belongs(detection, speakers)
    return group_nums


def groups_speaker_belongs(groups: List, speakers: List):
    groups_contained = groups.copy()
    for idx, g in enumerate(groups_contained):
        if len(set(g).intersection(set(speakers))) == 0:
            groups_contained.pop(idx)
    return groups_contained


def count_speaker_groups(
        breakpoints: pd.DataFrame,
        speaking_status_path: Optional[str] = "../data/export/sp_merged.pkl", 
        group_counter: Optional[Callable[[Sequence[int], Dict[str, Any]], Any]] = None,
) -> pd.DataFrame:
    """Augment breakpoint windows with speaking metadata.

    Parameters
    ----------
    breakpoints:
        DataFrame produced by `detect_group_num_breakpoints` (or similar) with at
        least the columns `Vid`, `time` (start/end frame pair), and optional `Cam`.
    speaking_status_path:
        Path to a merged speaking-status pickle generated by `write_sp_merged`.
    group_counter:
        Optional callable `(speakers: Sequence[int], context: Dict) -> Any` used to
        derive per-window group membership information. The return value is stored in
        the `scene_groups` column, and its length/size (if applicable) populates
        `num_scene_groups`. When omitted, `scene_groups` is set to ``None`` and the
        count defaults to zero.

    Returns
    -------
    pandas.DataFrame
        Copy of ``breakpoints`` with the additional columns:
          - ``speaking_window``: list of IDs speaking throughout the window.
          - ``speaking_in_scene``: filtered “in scene” speakers.
          - ``num_speaking_in_scene``: count of in-scene speakers.
          - ``scene_groups``: result from ``group_counter`` (or ``None``).
          - ``num_scene_groups``: number of groups (derived from ``scene_groups``).
    """
    if not isinstance(breakpoints, pd.DataFrame):
        raise TypeError("breakpoints must be a pandas DataFrame.")

    try:
        with open(speaking_status_path, "rb") as fh:
            sp_payload = pickle.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Speaking-status file not found: {speaking_status_path}") from exc

    speaking: Dict[int, np.ndarray] = sp_payload.get("speaking", {}) if isinstance(sp_payload, dict) else {}
    df = breakpoints.copy()
    df = df.reset_index(drop=True)

    def _normalize_groups_cell(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, float) and np.isnan(value):
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return [value]

    groups_columns = ['headRes', 'shoulderRes', 'hipRes', 'footRes']
    for col in groups_columns:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        else:
            df[col] = df[col].apply(_normalize_groups_cell)

    speaking_col: List[List[int]] = []
    in_scene_col: List[List[int]] = []
    num_scene_col: List[int] = []
    groups_sp_belong_col: List[Any] = []
    num_groups_sp_belong_col: List[int] = []
    all_people_col: List[List[int]] = []

    for idx, row in df.iterrows():
        vid = int(row.get("Vid", 0))
        time_window = row.get("time", [0, 0])
        if not isinstance(time_window, (list, tuple)) or len(time_window) < 2:
            time_window = [0, 0]
        start = int(time_window[0])
        end = int(time_window[1])
        if end < start:
            start, end = end, start

        status_matrix = speaking.get(vid)
        speakers_in_window: List[int] = []
        if isinstance(status_matrix, np.ndarray) and status_matrix.ndim == 2 and status_matrix.shape[0] >= 2:
            ids = status_matrix[0, :].astype(int)
            speaking_data = status_matrix[1:, :]
            # Convert to zero-based indices; ensure bounds.
            start_idx = max(0, start - 1)
            end_idx = min(speaking_data.shape[0] - 1, end - 1)
            if end_idx >= start_idx:
                window_slice = speaking_data[start_idx:end_idx + 1, :]
                if window_slice.size:
                    mask = np.all(window_slice == 1, axis=0)
                    speakers_in_window = ids[mask].astype(int).tolist()

        speaking_col.append(speakers_in_window)

        group_detection = row.get('headRes')
        people_in_scene = sorted({pid for group in group_detection for pid in group})
        speakers_in_scene = [pid for pid in speakers_in_window if pid in people_in_scene]
        all_people_col.append(people_in_scene)


        speakers_in_scene = [pid for pid in speakers_in_window if pid in people_in_scene]
        in_scene_col.append(speakers_in_scene)
        num_scene = len(speakers_in_scene)
        num_scene_col.append(num_scene)

        groups_speaker_belong = groups_speaker_belongs_clues(row, speakers_in_scene)
        num_groups_belong = {clue:len(g) for clue, g in groups_speaker_belong.items()}

        groups_sp_belong_col.append(groups_speaker_belong)
        num_groups_sp_belong_col.append(num_groups_belong)

    df["all_people_in_scene"] = all_people_col
    df["speaking"] = speaking_col
    df["speaking_in_scene"] = in_scene_col
    df["num_speaking_in_scene"] = num_scene_col
    df["groups_speaker_belong"] = groups_sp_belong_col
    df["num_groups_speaker_belong"] = num_groups_sp_belong_col

    return df


if __name__ == '__main__': 
    write_sp_merged('../data/export/speaking_status_py.mat', '../data/export/sp_merged.pkl')
