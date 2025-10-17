from __future__ import annotations

from typing import List

import math
import numpy as np


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
        total += len(A) * max(0, subgroup_count - 1)
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

