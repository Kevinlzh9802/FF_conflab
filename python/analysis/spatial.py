from __future__ import annotations

from typing import List

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

FEATURE_COLS = ['headRes', 'shoulderRes', 'hipRes', 'footRes']

def compute_homogeneity(G1: List[List[int]], G2: List[List[int]]) -> float:
    def as_col(groups):
        return [list(map(int, g)) for g in groups]

    G1 = as_col(G1)
    G2 = as_col(G2)
    all_elems = sorted(set(sum(G1, []) + sum(G2, [])))
    if not all_elems:
        return 1.0
    # pad with singletons
    G1_elems = set(sum(G1, []))
    G2_elems = set(sum(G2, []))
    for e in (set(all_elems) - G1_elems):
        G1.append([e])
    for e in (set(all_elems) - G2_elems):
        G2.append([e])
    C = len(G1)
    K = len(G2)
    N = len(all_elems)
    # contingency
    ack = np.zeros((K, C), dtype=int)
    for k in range(K):
        B = set(G2[k])
        for c in range(C):
            ack[k, c] = len(B.intersection(G1[c]))
    ac = ack.sum(axis=0)
    H_C = 0.0
    for c in range(C):
        pc = ac[c] / N
        if pc > 0:
            H_C -= pc * math.log(pc)
    H_C_given_K = 0.0
    for k in range(K):
        a_k = ack[k, :].sum()
        for c in range(C):
            a_ck = ack[k, c]
            if a_ck > 0:
                H_C_given_K -= (a_ck / N) * math.log(a_ck / a_k)
    if H_C == 0:
        return 1.0
    return 1.0 - H_C_given_K / H_C


def compute_split_score(G1: List[List[int]], G2: List[List[int]]) -> float:
    G1 = [list(map(int, g)) for g in G1]
    G2 = [list(map(int, g)) for g in G2]
    all_elems = sorted(set(sum(G1, []) + sum(G2, [])))
    if not all_elems:
        return 0.0
    G1_elems = set(sum(G1, []))
    G2_elems = set(sum(G2, []))
    for e in (set(all_elems) - G1_elems):
        G1.append([e])
    for e in (set(all_elems) - G2_elems):
        G2.append([e])
    total = 0.0
    for A in G1:
        if not A:
            continue
        subgroup_count = 0
        Aset = set(A)
        for B in G2:
            overlap = Aset.intersection(B)
            if len(overlap) > 1:
                subgroup_count += 1
        total += max(0, subgroup_count - 1)
    return total / max(len(G1), 1)


def compute_hic_matrix(GTgroups: List[List[int]], Detgroups: List[List[int]]) -> np.ndarray:
    if not GTgroups or not Detgroups:
        return np.array(0)
    max_gt = max(len(g) for g in GTgroups)
    max_det = max(len(g) for g in Detgroups)
    HIC = np.zeros((max_gt, max_det), dtype=float)
    person_to_gt = {}
    for g in GTgroups:
        for p in g:
            person_to_gt[int(p)] = len(g)
    for g in Detgroups:
        det_card = len(g)
        for p in g:
            p = int(p)
            if p in person_to_gt:
                gt_card = person_to_gt[p]
                HIC[gt_card - 1, det_card - 1] += 1
    # normalize rows
    row_sums = HIC.sum(axis=1, keepdims=True)
    nz = row_sums[:, 0] > 0
    HIC[nz] = HIC[nz] / row_sums[nz]
    return HIC

def getHIC(used_data: pd.DataFrame) -> pd.DataFrame:
    """Populate a 'HIC' column by computing a head interaction consistency matrix.

    Mirrors GCFF/getHIC.m behavior:
    used_data.HIC = cell(height(used_data), 1);
    used_data.HIC{k} = computeHICMatrix(used_data.GT{k}, used_data.headRes{k});

    Unknown dependency: computeHICMatrix — TODO: provide implementation in utilities.
    """
    df = used_data.copy()
    if 'HIC' not in df.columns:
        df['HIC'] = [None] * len(df)
    for idx, row in df.iterrows():
        df.at[idx, 'HIC'] = compute_hic_matrix(row.get('GT'), row.get('headRes'))
    return df


def spatial_scores_df(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if df.empty:
        return np.array([]), np.array([])
    
    n = len(feature_cols)
    hom_sums = np.zeros((n, n), dtype=float)
    split_sums = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)
    for idx, row in df.iterrows():
        groups_cache = {feature: row.get(feature, []) for feature in feature_cols}
        for i, fa in enumerate(feature_cols):
            groups_a = groups_cache[fa]
            if not groups_a:
                continue
            for j, fb in enumerate(feature_cols):
                groups_b = groups_cache[fb]
                if not groups_b:
                    continue
                h_score = compute_homogeneity(groups_a, groups_b)
                s_score = compute_split_score(groups_a, groups_b)
                if np.isnan(h_score) or np.isnan(s_score):
                    continue
                length_coeff = (row.get('length', []) - 1) / 60
                hom_sums[i, j] += h_score * length_coeff
                split_sums[i, j] += s_score * length_coeff
                counts[i, j] += length_coeff
        c = 9
    with np.errstate(divide='ignore', invalid='ignore'):
        hom_matrix = np.divide(
            hom_sums,
            counts,
            out=np.full_like(hom_sums, np.nan, dtype=float),
            where=counts > 0
        )
        split_matrix = np.divide(
            split_sums,
            counts,
            out=np.full_like(split_sums, np.nan, dtype=float),
            where=counts > 0
        )
    if np.all(np.isnan(hom_matrix)) and np.all(np.isnan(split_matrix)):
        return hom_matrix, split_matrix
    fig_h, ax_h = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        hom_matrix,
        ax=ax_h,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=feature_cols,
        yticklabels=feature_cols,
        cbar_kws={"label": "Homogeneity"}
    )
    ax_h.set_title("Average Homogeneity Scores")
    fig_h.tight_layout()
    fig_s, ax_s = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        split_matrix,
        ax=ax_s,
        annot=True,
        fmt=".3f",
        cmap="magma",
        xticklabels=feature_cols,
        yticklabels=feature_cols,
        cbar_kws={"label": "Split Score"}
    )
    ax_s.set_title("Average Split Scores")
    fig_s.tight_layout()
    plt.show()
    return fig_h, fig_s