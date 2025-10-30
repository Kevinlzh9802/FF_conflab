"""Placeholders for MATLAB script files referenced by example_GCFF.m.

These are stubs with function signatures so the pipeline can call them.
Fill in implementations when ready or integrate your project-specific logic.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# TODO: move all functions in this file to other files
def _add_merged_column(df: pd.DataFrame) -> pd.DataFrame:  # TODO: move to table.py
    ids = pd.unique(df['Seg'])
    offset = 0.0
    concat_ts = np.full((len(df),), np.nan, dtype=float)
    for seg_id in ids:
        idx = df['Seg'] == seg_id
        frames = df.loc[idx, 'Timestamp'].to_numpy(dtype=float)
        merged = offset + frames
        concat_ts[idx.to_numpy()] = merged
        offset = merged[-1]
    out = df.copy()
    out['concat_ts'] = concat_ts
    return out


def concat_segs(data: pd.DataFrame) -> pd.DataFrame:
    """Port of utils/concatSegs.m.

    For each (Vid, Cam), treat rows as a continuous time sequence and
    concatenate their timestamps based on the order of Seg values.

    Adds a new column 'concat_ts' to the DataFrame without altering
    original 'Seg' or 'Timestamp' values.
    """
    df = data.copy()
    df['concat_ts'] = np.nan
    if 'Vid' not in df.columns or 'Cam' not in df.columns:
        return df
    groups = df.groupby(['Vid', 'Cam'], sort=False)
    for (vid, cam), idx in groups.groups.items():
        sub = df.loc[idx, ['Seg', 'Timestamp']].copy()
        merged = _add_merged_column(sub)
        df.loc[idx, 'concat_ts'] = merged['concat_ts'].values
    return df


def construct_formations(results: dict, data: pd.DataFrame, speaking_status: Dict[str, Any] | None = None):
    """Simplified port of utils/constructFormations.m.

    Builds a formations table from unique groups per Vid for the 'GT' column.
    Adds cardinality, id, participants, timestamps (from concat_ts), Cam, Vid.
    If speaking_status provided with merged arrays per vid, computes avg_speaker.
    Returns a pandas DataFrame 'formations'.
    """
    from utils.groups import record_unique_groups

    col_name = 'GT'
    vids = sorted(pd.unique(data['Vid']))
    rows: List[Dict[str, Any]] = []
    for vid in vids:
        ana = data.loc[data['Vid'] == vid]
        uniq = record_unique_groups(ana, col_name)
        for entry in uniq:
            rows.append({
                'participants': entry['participants'],
                'timestamps': entry['timestamps'],
                'timestamps_all': entry['timestamps'],
                'Cam': entry['Cam'],
                'Vid': vid,
            })
    if not rows:
        return pd.DataFrame(columns=['participants', 'timestamps', 'timestamps_all', 'Cam', 'Vid', 'cardinality', 'id', 'avg_speaker'])
    formations = pd.DataFrame(rows)
    formations['cardinality'] = formations['participants'].apply(lambda x: len(x))
    formations['id'] = np.arange(1, len(formations) + 1)
    # Filter basic constraints
    formations = formations[formations['cardinality'] >= 1]
    # Optional: compute avg_speaker if speaking_status provided as dict per vid: array with first row IDs, following rows status
    def _avg_speaker(row):
        if speaking_status is None:
            return np.nan
        vid = int(row['Vid'])
        actions = speaking_status.get(vid)
        if actions is None:
            return np.nan
        ids = actions[0, :]
        ts = np.asarray(row['timestamps_all'], dtype=int)
        cols = []
        for p in row['participants']:
            loc = np.where(ids == p)[0]
            if loc.size:
                cols.append(int(loc[0]))
        if not cols or ts.size == 0:
            return np.nan
        sp = actions[1:, :]
        ts = np.clip(ts, 1, sp.shape[0])
        vals = sp[ts - 1][:, cols]
        return float(np.sum(vals) / max(vals.size, 1))

    formations['avg_speaker'] = formations.apply(_avg_speaker, axis=1)
    return formations


def detect_group_num_breakpoints(data: pd.DataFrame, clues: List[str] | None = None):
    """Port of utils/detectGroupNumBreakpoints.m (simplified).

    - Finds breakpoints where any clue's groups change for each (Vid, Cam)
    - Generates windows between consecutive breakpoints
    - Returns a DataFrame window_table with columns: id, Vid, Cam, time, length, speaking_all_time
    """
    if clues is None:
        clues = ['headRes', 'shoulderRes', 'hipRes', 'footRes']
    videos = sorted(pd.unique(data['Vid']))
    cameras = sorted(pd.unique(data['Cam']))
    rows: List[Dict[str, Any]] = []
    for vid in videos:
        for cam in cameras:
            cam_data = data[(data['Vid'] == vid) & (data['Cam'] == cam)]
            if cam_data.empty or 'concat_ts' not in cam_data.columns:
                continue
            ts = sorted(pd.unique(cam_data['concat_ts'].dropna()))
            if len(ts) <= 1:
                continue
            breakpoints: List[int] = []
            prev_groups = None
            for t in ts:
                current_groups = []
                for clue in clues:
                    row_idx = cam_data.index[cam_data['concat_ts'] == t]
                    if len(row_idx) > 0 and clue in cam_data.columns:
                        current_groups.append(cam_data.loc[row_idx[0], clue])
                    else:
                        current_groups.append([])
                if prev_groups is None or any(not _equal_groups(a, b) for a, b in zip(current_groups, prev_groups)):
                    breakpoints.append(t)
                prev_groups = current_groups
            # ensure first and last
            if breakpoints and breakpoints[0] != ts[0]:
                breakpoints = [ts[0]] + breakpoints
            if breakpoints and breakpoints[-1] != ts[-1]:
                breakpoints = breakpoints + [ts[-1]]
            # windows
            for i in range(len(breakpoints) - 1):
                start = int(breakpoints[i])
                end = int(breakpoints[i + 1])
                rows.append({
                    'id': i + 1,
                    'Vid': int(vid),
                    'Cam': int(cam),
                    'time': [start, end],
                    'length': end - start + 1,
                    'speaking_all_time': [],
                })
    return pd.DataFrame(rows)


def _equal_groups(a, b):
    try:
        if a is None and b is None:
            return True
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            norm_a = sorted(sorted(x) for x in a)
            norm_b = sorted(sorted(x) for x in b)
            return norm_a == norm_b
        return a == b
    except Exception:
        return False


# Aliases matching original MATLAB script names
def constructFormations(results: dict, data=None):  # pragma: no cover - placeholder
    return construct_formations(results, data=data)
