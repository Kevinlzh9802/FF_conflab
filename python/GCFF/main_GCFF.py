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
from analysis.cross_modal import detect_group_num_breakpoints, count_speaker_groups
from utils.table import filter_and_concat_table
from utils.groups import turn_singletons_to_groups
from utils.plots import plot_all_skeletons, plot_panels_df
import sys
import shutil
import math

# Optional: tqdm for clean multi-line progress display
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def display_frame_results(idx_frame: int, total_frames: int, groups, GTgroups) -> None:
    """Render a 3-line, in-place status that continuously updates.

    Line 1: Frame n/N with tqdm progress bar (if available)
    Line 2: FOUND: [found groups]
    Line 3: GT:    [GT groups]

    - Uses tqdm multi-bars when available for robust, non-flooding updates.
    - Falls back to ANSI cursor control to update 3 logical lines even if they wrap.
    """
    found_txt = " |".join(str(g) for g in (groups or [])) or "No Groups"
    gt_txt = " |".join(str(g) for g in (GTgroups or [])) or "No Groups"

    # Preferred: tqdm stacked bars (positions 0..2)
    if _tqdm is not None:
        if not hasattr(display_frame_results, "_bars"):
            # Initialize three bars on first call
            p0 = _tqdm(total=total_frames, position=0, dynamic_ncols=True,
                       leave=True, bar_format="Frame {n_fmt}/{total_fmt} {bar} {percentage:3.0f}%")
            p1 = _tqdm(total=1, position=1, dynamic_ncols=True, leave=True,
                       bar_format="FOUND: {desc}")
            p2 = _tqdm(total=1, position=2, dynamic_ncols=True, leave=True,
                       bar_format="GT:    {desc}")
            display_frame_results._bars = (p0, p1, p2)  # type: ignore[attr-defined]

        p0, p1, p2 = display_frame_results._bars  # type: ignore[attr-defined]

        # Update progress and lines
        p0.n = min(idx_frame, total_frames)
        p0.refresh()
        p1.set_description_str(found_txt)
        p1.refresh()
        p2.set_description_str(gt_txt)
        p2.refresh()

        if idx_frame >= total_frames:
            p2.close(); p1.close(); p0.close()
            delattr(display_frame_results, "_bars")  # type: ignore[attr-defined]
        return

    # Fallback: manual 3-line update using ANSI, handling wrapping
    msg1 = f"Frame {idx_frame}/{total_frames}"
    msg2 = f"FOUND: {found_txt}"
    msg3 = f"GT:    {gt_txt}"

    width = shutil.get_terminal_size(fallback=(80, 20)).columns or 80

    def phys_rows(s: str) -> int:
        # Estimate how many terminal rows the string will occupy when wrapped
        # Avoid zero; treat empty as one row
        return max(1, math.ceil(len(s) / max(1, width)))

    new_h = phys_rows(msg1) + phys_rows(msg2) + phys_rows(msg3)
    prev_h = getattr(display_frame_results, "_prev_h", 0)

    if prev_h:
        # Move cursor up to the beginning of the previous block
        sys.stdout.write(f"\x1b[{prev_h}A")
        # Clear previous block line by line
        for i in range(prev_h):
            sys.stdout.write("\r\x1b[2K")
            if i < prev_h - 1:
                sys.stdout.write("\n")
        # Move back up to start position
        sys.stdout.write(f"\x1b[{prev_h - 1}A" if prev_h > 1 else "\r")

    # Print new block
    print(msg1)
    print(msg2)
    print(msg3, end="", flush=True)

    display_frame_results._prev_h = new_h  # type: ignore[attr-defined]
    if idx_frame >= total_frames:
        print("")
        display_frame_results._prev_h = 0  # type: ignore[attr-defined]


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

    for idx in range(35, T):
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
