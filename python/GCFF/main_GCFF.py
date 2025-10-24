"""
Python port of GCFF/example_GCFF.m as an executable module.

This script orchestrates the GCFF pipeline over a sequence of frames.
It expects pre-loaded data structures and calls into Python ports of the
original MATLAB functions. Script-style .m files are represented here as
function calls with parameters (see utils.python.scripts stubs).

Typical usage (pseudo-code):
    from GCFF.python.main_GCFF import run_gcff_sequence, Params
    results, data_out = run_gcff_sequence(data, Params(stride=40, mdl=6000),
                                          clue="head", speaking_status=sp, use_real=True)

Data expectations:
    data: a dict-like or pandas.DataFrame with columns:
        - headFeat, shoulderFeat, hipFeat, footFeat (per-frame arrays)
        - GT (per-frame list of groups) [optional]
        - Cam, Vid, Seg, Timestamp (scalars per frame)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import os
import numpy as np
import argparse
import pandas as pd

from gcff_core import ff_deletesingletons, ff_evalgroups, gc
from utils.speaking import read_speaking_status, get_status_for_group
from utils.scripts import constructFormations, detectGroupNumBreakpoints
from utils.data import filter_and_concat_table

ALL_CLUES = ["head", "shoulder", "hip", "foot"]
USED_SEGS = ["429"]

@dataclass
class Params:
    stride: float
    mdl: float
    use_real: bool
    used_parts: Optional[List[str]] = None  # e.g., ["233", "429"]


def display_frame_results(idx_frame: int, total_frames: int, groups, GTgroups) -> None:
    print(f"Frame: {idx_frame}/{total_frames}")
    # Found:
    print("   FOUND:-- ", end="")
    if groups:
        for g in groups:
            print(f" {g}", end=" |")
    else:
        print(" No Groups ", end="")
    print("")
    # GT:
    print("   GT   :-- ", end="")
    if GTgroups:
        for g in GTgroups:
            print(f" {g}", end=" |")
    else:
        print(" No Groups ", end="")
    print("")


def gcff_experiments(data: pd.DataFrame, params: Params, speaking_status: Any):
    # Optional: filter and concat table by 3-digit keys in params.used_parts
    data = filter_and_concat_table(data, params.used_parts)

    # Build features per frame for the selected clue
    for clue in ALL_CLUES:
        feat_col = f"{clue}Feat"
        features = list(data[feat_col]) if hasattr(data, '__getitem__') else []
        GTgroups = list(data['GT']) if ('GT' in getattr(data, 'columns', [])) else [None] * len(features)
        timestamps = list(data['Timestamp']) if ('Timestamp' in getattr(data, 'columns', [])) else list(range(len(features)))

        results, data_returned = gcff_sequence(features, GTgroups, params, speaking_status)

    # Translate remaining scripts to function calls (placeholders for now)
    # _formations = constructFormations(results, data=data)
    # _breakpoints = detectGroupNumBreakpoints(results, data=data)
    return results, data_returned

def gcff_sequence(features, GTgroups, params, speaking_status):
    """High-level pipeline adapted from example_GCFF.m.

    Returns (results_dict, data_out)
    """
    T = len(features)
    TP = np.zeros(T)
    FP = np.zeros(T)
    FN = np.zeros(T)
    precision = np.zeros(T)
    recall = np.zeros(T)
    groups_out: List[List[List[int]]] = [None] * T
    s_speaker: List[float] = []
    group_sizes: List[int] = []

    for idx in range(T):
        feat = features[idx]
        if feat is None or len(feat) == 0:
            groups_out[idx] = []
            continue
        # MATLAB: feat = features{idx}(:, [1:24] + 24 * use_real)
        # Expect feat as 2D array where columns include [ID,x,y,alpha,...]
        cols = np.arange(0 if not params.use_real else 24, 24 + (24 if params.use_real else 0))
        cols = cols[: min(feat.shape[1], 24)] if feat.shape[1] >= 24 else np.arange(feat.shape[1])
        F = feat[:, cols]
        labels = gc(F[:, :4], params.stride, params.mdl)
        groups = []
        for lab in range(int(labels.max()) + 1 if labels.size else 0):
            members = F[labels == lab, 0].astype(int).tolist()
            groups.append(members)
        # Delete singletons
        groups = ff_deletesingletons(groups) if groups else []
        groups_out[idx] = groups
        # Apply GT filtering too
        GT = ff_deletesingletons(GTgroups[idx]) if (GTgroups and GTgroups[idx]) else []
        # Evaluate
        pr, re, tp, fp, fn = ff_evalgroups(groups, GT, TH='card', cardmode=0)
        precision[idx], recall[idx], TP[idx], FP[idx], FN[idx] = pr, re, tp, fp, fn

        # Optionally collect speaking status per-frame (if structure available)
        try:
            if isinstance(data, pd.DataFrame):
                info = data.iloc[idx][['Cam', 'Vid', 'Seg', 'Timestamp']]
                sp, cf = read_speaking_status(speaking_status, int(info.Vid), int(info.Seg), int(info.Timestamp) + 1, 1)
                if isinstance(sp, (list, np.ndarray)) and groups:
                    ss = get_status_for_group(np.arange(len(sp)), sp, groups)
                    for vals in ss:
                        ssv = float(np.sum(vals))
                        if not (ssv > 10 or ssv < 0):
                            group_sizes.append(len(vals))
                            s_speaker.append(ssv)
        except Exception:
            pass

        display_frame_results(idx + 1, T, groups, GT)

    pr_avg = float(np.nanmean(precision)) if precision.size else float('nan')
    re_avg = float(np.nanmean(recall)) if recall.size else float('nan')
    F1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
    F1_avg = float(np.nanmean(F1)) if F1.size else float('nan')

    results = {
        'precisions': precision,
        'recalls': recall,
        'F1s': F1,
        'F1_avg': F1_avg,
        'groups': groups_out,
        'group_sizes': np.array(group_sizes),
        's_speaker': np.array(s_speaker),
        # 'body_orientations': clue,
    }



    return results, data


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Run GCFF main pipeline.')
    parser.add_argument('--data', type=str, required=False, help='Path to parent directory with features and metadata', default="../data/export/")
    parser.add_argument('--clue', type=str, default='head', choices=['head', 'shoulder', 'hip', 'foot'])
    parser.add_argument('--stride', type=float, default=40.0)
    parser.add_argument('--mdl', type=float, default=6000.0)
    parser.add_argument('--use-real', type=bool, default=True)
    args = parser.parse_args()

    data = pd.read_pickle(args.data + "data.pkl")
    params = Params(args.stride, args.mdl, args.use_real, used_parts=USED_SEGS)
    res, _ = gcff_experiments(data, params, speaking_status=None)
    print('F1_avg:', res['F1_avg'])
