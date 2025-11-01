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
import yaml
from munch import Munch

from gcff_core import ff_deletesingletons, ff_evalgroups, graph_cut
from utils.scripts import detect_group_num_breakpoints
from utils.data import filter_and_concat_table
from utils.groups import turn_singletons_to_groups
from utils.plots import plot_all_skeletons, plot_panels_df
from utils.speaking import count_speaker_groups


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


def gcff_experiments(config: Munch) -> pd.DataFrame:
    # read keypoint data, prioritize finished data with detections
    if config.force_rerun:
        data_kp = pd.read_pickle(config.data_paths.kp)
        rerun = True
    else:
        try:
            data_kp = pd.read_pickle(config.data_paths.kp_finished)
            rerun = False
        except:
            data_kp = pd.read_pickle(config.data_paths.kp)
            rerun = True

    # filter and concat table by 3-digit keys in params.used_parts
    data_kp = filter_and_concat_table(data_kp, config.used_segs)

    # Build features per frame for the selected clue
    if rerun:
        for clue in config.all_clues:
            if config.use_space:
                features = [data_kp["spaceFeat"][k][clue] for k in range(len(data_kp))]
            else:
                features = [data_kp["pixelFeat"][k][clue] for k in range(len(data_kp))]
            GTgroups = list(data_kp['GT']) if ('GT' in getattr(data_kp, 'columns', [])) else [None] * len(features)

            results = gcff_sequence(features, GTgroups, config.params)
            data_kp[f"{clue}Res"] = results['groups']

        data_kp.to_pickle(config.data_paths.kp_finished)
    
    # Translate remaining scripts to function calls (placeholders for now)
    breakpoints = detect_group_num_breakpoints(data=data_kp)
    breakpoints = count_speaker_groups(breakpoints)
    
    # Save detection results as panels
    # plot_panels_df(data_kp)
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

        labels = graph_cut(feat, params.stride, params.mdl)
        groups = []
        for lab in range(int(labels.max()) + 1 if labels.size else 0):
            members = feat[labels == lab, 0].astype(int).tolist()
            groups.append(members)

        # Deal with singletons
        if not ff_deletesingletons(groups):  # which means groups are all singletons
            groups = []
        groups = turn_singletons_to_groups(groups)
        GT = turn_singletons_to_groups(GTgroups[idx])
        groups_out[idx] = groups

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
    parser.add_argument('--config', type=str, default="./configs/config_GCFF.yaml", help='Path to config YAML file')
    parser.add_argument('--stride', type=float, default=None)
    parser.add_argument('--mdl', type=float, default=None)
    parser.add_argument('--use-space', type=bool, default=None)
    parser.add_argument('--force-rerun', type=bool, default=None)
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    
    config = Munch(config)
    for key, item in config.items():
        if isinstance(item, dict):
            config[key] = Munch(item)
    
    if args.stride is not None:
        config.params.stride = args.stride
    if args.mdl is not None:
        config.params.mdl = args.mdl
    if args.use_space is not None:
        config.use_space = args.use_space
    if args.force_rerun is not None:
        config.force_rerun = args.force_rerun

    res = gcff_experiments(config)
    # print('F1_avg:', res['F1_avg'])
