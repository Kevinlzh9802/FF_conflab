"""Pythonized equivalents of example_GCFF.m and example_GCFF_noGT.m.

These scripts depend on project-specific datasets and utilities that are
outside this folder. The structure matches the MATLAB examples and calls
into the Python ports where available.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from gcff_core import gc, ff_deletesingletons, ff_evalgroups


@dataclass
class Params:
    stride: float
    mdl: float


def example_GCFF(features: List[np.ndarray], GTgroups: List[List[List[int]]], params: Params):
    """Run GCFF pipeline with GT evaluation.

    features: list of per-frame arrays, columns [ID,x,y,alpha,...]
    GTgroups: list of per-frame list-of-groups
    params: Params(stride, mdl)
    Returns a dict with metrics and per-frame groups
    """
    T = len(features)
    precision = np.zeros(T)
    recall = np.zeros(T)
    TP = np.zeros(T)
    FP = np.zeros(T)
    FN = np.zeros(T)
    groups: List[List[List[int]]] = [[] for _ in range(T)]
    for t in range(T):
        feat = features[t]
        if feat is None or feat.size == 0:
            continue
        gg = gc(feat[:, :4], params.stride, params.mdl)
        groups_t: List[List[int]] = []
        for lab in range(int(gg.max()) + 1):
            members = feat[gg == lab, 0].astype(int).tolist()
            groups_t.append(members)
        groups_t = ff_deletesingletons(groups_t) if groups_t else []
        groups[t] = groups_t
        GT_t = ff_deletesingletons(GTgroups[t]) if GTgroups and GTgroups[t] else []
        pr, re, tp, fp, fn = ff_evalgroups(groups_t, GT_t, TH='card', cardmode=0)
        precision[t], recall[t], TP[t], FP[t], FN[t] = pr, re, tp, fp, fn
    pr_avg = np.nanmean(precision)
    re_avg = np.nanmean(recall)
    F1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
    F1_avg = np.nanmean(F1)
    return {
        'precisions': precision,
        'recalls': recall,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'F1s': F1,
        'F1_avg': F1_avg,
        'groups': groups,
    }


def example_GCFF_noGT(features: List[np.ndarray], params: Params):
    """Run GCFF when ground truth is not available. Returns detected groups.
    """
    T = len(features)
    groups: List[List[List[int]]] = [[] for _ in range(T)]
    for t in range(T):
        feat = features[t]
        if feat is None or feat.size == 0:
            continue
        gg = gc(feat[:, :4], params.stride, params.mdl)
        groups_t: List[List[int]] = []
        for lab in range(int(gg.max()) + 1):
            members = feat[gg == lab, 0].astype(int).tolist()
            groups_t.append(members)
        groups[t] = ff_deletesingletons(groups_t) if groups_t else []
    return groups

