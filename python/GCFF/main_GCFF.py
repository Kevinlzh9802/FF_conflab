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
from pathlib import Path

import os
import numpy as np
import argparse
import pandas as pd

from gcff_core import ff_deletesingletons, ff_evalgroups, graph_cut
from utils.scripts import constructFormations, detect_group_num_breakpoints
from utils.data import filter_and_concat_table
from utils.groups import turn_singletons_to_groups
from utils.plots import plot_all_skeletons, plot_panels_df

ALL_CLUES = ["head", "shoulder", "hip", "foot"]
USED_SEGS = []

@dataclass
class Params:
    stride: float
    mdl: float
    use_real: bool
    used_parts: Optional[List[str]] = None  # e.g., ["233", "429"]
    data_paths: Optional[Dict] = None


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


def gcff_experiments(params: Params):
    force_rerun = True  # TODO: make this an argument
    # read keypoint data, prioritize finished data with detections
    if force_rerun:
        data_kp = pd.read_pickle(params.data_paths["kp"])
        rerun = True
    else:
        try:
            data_kp = pd.read_pickle(params.data_paths["kp_finished"])
            rerun = False
        except:
            data_kp = pd.read_pickle(params.data_paths["kp"])
            rerun = True
        
    # filter and concat table by 3-digit keys in params.used_parts
    data_kp = filter_and_concat_table(data_kp, params.used_parts)

    # Build features per frame for the selected clue
    if rerun:
        for clue in ALL_CLUES:
            feat_col = f"{clue}Feat"
            features = list(data_kp[feat_col]) if hasattr(data_kp, '__getitem__') else []
            GTgroups = list(data_kp['GT']) if ('GT' in getattr(data_kp, 'columns', [])) else [None] * len(features)

            results = gcff_sequence(features, GTgroups, params)
            data_kp[f"{clue}Res"] = results['groups']

        data_kp.to_pickle(params.data_paths["kp_finished"])
    
    # Translate remaining scripts to function calls (placeholders for now)
    # breakpoints = detect_group_num_breakpoints(data=data_kp)

    # Save detection results as panels
    plot_panels_df(data_kp)
    return data_kp

def gcff_sequence(features, GTgroups, params):
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
        if feat is None or len(feat) == 0 or feat.shape[1] == 0:
            groups_out[idx] = []
            continue
        # MATLAB: feat = features{idx}(:, [1:24] + 24 * use_real)
        # Expect feat as 2D array where columns include [ID,x,y,alpha,...]
        if params.use_real:
            F = feat[:, 24:28]
        else:
            F = feat[:, 0:4]
        labels = graph_cut(F[:, :4], params.stride, params.mdl)
        groups = []
        for lab in range(int(labels.max()) + 1 if labels.size else 0):
            members = F[labels == lab, 0].astype(int).tolist()
            groups.append(members)

        # Deal with
        if not ff_deletesingletons(groups):  # which means groups are all singletons
            groups = []
        groups = turn_singletons_to_groups(groups)
        GT = turn_singletons_to_groups(GTgroups[idx])
        groups_out[idx] = groups

        # GT = ff_deletesingletons(GTgroups[idx]) if (GTgroups and GTgroups[idx]) else []
        # Evaluate
        pr, re, tp, fp, fn = ff_evalgroups(groups, GT, TH='card', cardmode=0)
        precision[idx], recall[idx], TP[idx], FP[idx], FN[idx] = pr, re, tp, fp, fn

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
    }

    return results


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Run GCFF main pipeline.')
    parser.add_argument('--data', type=str, required=False, help='Path to parent directory with features and metadata', default="../data/export/")
    parser.add_argument('--clue', type=str, default='head', choices=['head', 'shoulder', 'hip', 'foot'])
    parser.add_argument('--stride', type=float, default=40.0)
    parser.add_argument('--mdl', type=float, default=6000.0)
    parser.add_argument('--use-real', type=bool, default=True)
    args = parser.parse_args()

    params = Params(args.stride, args.mdl, args.use_real, used_parts=USED_SEGS)
    params.data_paths = {
        "kp": args.data + "data.pkl",
        "kp_finished": args.data + "data_finished.pkl",
        "sp": args.data + "sp_merged.pkl",
        "frames": args.data + "frames/",
    }
    res = gcff_experiments(params)
    # print('F1_avg:', res['F1_avg'])
