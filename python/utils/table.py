import json, os
import pandas as pd
import numpy as np
import re


from typing import Any, Iterable, List, Sequence, Tuple, Optional

ALL_CLUES = ["head", "shoulder", "hip", "foot"]

def filter_and_concat_table(data, used_segs=None): 
    df = data
    if isinstance(df, pd.DataFrame) and used_segs:
        sel = df.iloc[0:0].copy()
        for key in used_segs:
            if isinstance(key, str) and len(key) == 3 and key.isdigit():
                cam, vid, seg = int(key[0]), int(key[1]), int(key[2])
                mask = (df['Cam'] == cam) & (df['Vid'] == vid) & (df['Seg'] == seg)
                if mask.any():
                    sel = pd.concat([sel, df.loc[mask]], ignore_index=True)
        if len(sel) > 0:
            df = sel
    return df


def convert_cell_array_to_table(cell_array: List[List[Any]]):
    """Convert a MATLAB-like cell array to a dict-of-lists table.

    Expects cell_array columns: [FrameData, "camX_vidY_segZ", Timestamp]
    Returns dict with keys: FrameData, Cam, Vid, Seg, Timestamp
    """
    out = {
        'FrameData': [row[0] for row in cell_array],
        'Cam': [],
        'Vid': [],
        'Seg': [],
        'Timestamp': [row[2] for row in cell_array],
    }
    for i, row in enumerate(cell_array):
        m = re.search(r"cam(\d+)_vid(\d+)_seg(\d+)", str(row[1]))
        if m:
            out['Cam'].append(int(m.group(1)))
            out['Vid'].append(int(m.group(2)))
            out['Seg'].append(int(m.group(3)))
        else:
            out['Cam'].append(None)
            out['Vid'].append(None)
            out['Seg'].append(None)
    return out


# def filter_and_concat_table(T, keys: Optional[Sequence[str]] = None):
#     """Filter and concatenate rows where (Cam,Vid,Seg) matches keys like '233'."""
#     if keys is None or len(keys) == 0:
#         return T
#     out = []
#     for k in keys:
#         if not (isinstance(k, str) and len(k) == 3 and k.isdigit()):
#             raise ValueError("Each key must be a 3-digit numeric string")
#         cam, vid, seg = int(k[0]), int(k[1]), int(k[2])
#         for row in T:
#             if row['Cam'] == cam and row['Vid'] == vid and row['Seg'] == seg:
#                 out.append(row)
#     return out


def filter_table(data_table, cam_values, vid_values, seg_values):
    """Filter a list of dict rows by Cam/Vid/Seg values or 'all'."""
    def as_set(v):
        if v == 'all':
            return None
        return set(v if isinstance(v, (list, tuple)) else [v])

    cams = as_set(cam_values)
    vids = as_set(vid_values)
    segs = as_set(seg_values)
    out = []
    for row in data_table:
        if cams is not None and row['Cam'] not in cams:
            continue
        if vids is not None and row['Vid'] not in vids:
            continue
        if segs is not None and row['Seg'] not in segs:
            continue
        out.append(row)
    return out


def find_matching_frame(table1, table2, n: int):
    """Find matching frame row in table2 for row n of table1.

    Both tables are lists of dict with keys: FrameData, Cam, Vid, Seg, Timestamp.
    """
    if n < 0 or n >= len(table1):
        raise IndexError("n out of bounds")
    r = table1[n]
    for row in table2:
        if row['Cam'] == r['Cam'] and row['Vid'] == r['Vid'] and row['Seg'] == r['Seg'] and row['Timestamp'] == r['Timestamp']:
            return row['FrameData']
    return None


