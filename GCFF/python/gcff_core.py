import math
from typing import List, Sequence, Tuple, Union

import numpy as np

from GCFF.graphopt.python.graphcut import expand as expand_mex


def ff_deletesingletons(groups: List[Sequence[int]]) -> List[List[int]]:
    """Delete groups with a single member.

    groups: list of iterables of ints
    returns: filtered list with only groups of size > 1
    """
    return [list(g) for g in groups if len(g) > 1]


def ff_evalgroups(
    group: List[Sequence[int]],
    GT: List[Sequence[int]],
    TH: Union[str, float] = "card",
    cardmode: Union[int, str] = 0,
):
    """Compute precision/recall and TP/FP/FN given detected and GT groups.

    - group, GT: lists of lists of subject IDs (ints)
    - TH: 'card' (2/3), 'all' (1.0), or float in [0,1]
    - cardmode: 0/1 or 'cardmode' for per-cardinality evaluation

    Returns: (precision, recall, TP, FP, FN)
      If cardmode==0: floats and ints
      If cardmode==1: numpy arrays per cardinality
    """
    # Normalize inputs
    group = [list(map(int, g)) for g in (group or [])]
    GT = [list(map(int, g)) for g in (GT or [])]

    # TH handling
    if isinstance(TH, str):
        if TH == "card":
            THv = 2.0 / 3.0
        elif TH == "all":
            THv = 1.0
        else:
            raise ValueError("TH must be in [0,1] or 'card'/'all'")
    else:
        THv = float(TH)
        if not (0.0 <= THv <= 1.0):
            raise ValueError("TH must be in [0,1]")

    # cardmode handling
    if cardmode in (1, "cardmode"):
        cm = 1
    else:
        cm = 0

    if cm == 0:
        # Degenerate cases
        if not group and not GT:
            return (1.0, 1.0, 0, 0, 0)
        if not group and GT:
            return (1.0, 0.0, 0, 0, len(GT))
        if group and not GT:
            return (0.0, 1.0, 0, len(group), 0)

        TP = 0
        # For each GT, search any matching detected group
        for GTg in GT:
            GTcard = len(GTg)
            matched = False
            GTset = set(GTg)
            for Gg in group:
                groupcard = len(Gg)
                interscard = len(GTset.intersection(Gg))
                if (interscard / max(GTcard, groupcard)) >= THv:
                    TP += 1
                    matched = True
                    break
        FP = len(group) - TP
        FN = len(GT) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
        recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        return precision, recall, TP, FP, FN

    # cardmode per-cardinality evaluation
    c_GT = np.array([len(g) for g in GT], dtype=int) if GT else np.array([], dtype=int)
    c_group = np.array([len(g) for g in group], dtype=int) if group else np.array([], dtype=int)

    if GT:
        GTcardmax = int(c_GT.max())
        h_GT = np.bincount(c_GT, minlength=GTcardmax + 1)[1:]
    else:
        GTcardmax = 0
        h_GT = np.array([], dtype=int)
    if group:
        groupcardmax = int(c_group.max())
        h_group = np.bincount(c_group, minlength=groupcardmax + 1)[1:]
    else:
        groupcardmax = 0
        h_group = np.array([], dtype=int)

    if not group and not GT:
        precision = np.nan
        recall = np.nan
        return precision, recall, np.nan, np.nan, np.nan
    if not group and GT:
        TP = np.zeros(GTcardmax, dtype=int)
        FP = 0
        FN = h_GT
        precision = np.zeros_like(TP, dtype=float)
        denom = (TP + FP)
        precision = np.divide(TP, denom, out=np.full_like(TP, np.nan, dtype=float), where=denom != 0)
        recall = np.divide(TP, TP + FN, out=np.full_like(TP, np.nan, dtype=float), where=(TP + FN) != 0)
        return precision, recall, TP, FP, FN
    if group and not GT:
        TP = np.zeros(groupcardmax, dtype=int)
        FP = h_group
        FN = 0
        recall = np.zeros_like(TP, dtype=float)
        denom = (TP + FP)
        precision = np.divide(TP, denom, out=np.full_like(TP, np.nan, dtype=float), where=denom != 0)
        return precision, recall, TP, FP, FN

    TP = np.zeros((GTcardmax, groupcardmax), dtype=int)
    # Match per GT group
    for GTg in GT:
        GTcard = len(GTg)
        GTset = set(GTg)
        for Gg in group:
            groupcard = len(Gg)
            interscard = len(GTset.intersection(Gg))
            if (interscard / max(GTcard, groupcard)) >= THv:
                TP[GTcard - 1, groupcard - 1] += 1
                break
    FP = h_group - TP.sum(axis=0)
    FN = h_GT - TP.sum(axis=1)
    precision = np.divide(TP.sum(axis=0), TP.sum(axis=0) + FP, out=np.full_like(FP, np.nan, dtype=float), where=(TP.sum(axis=0) + FP) != 0)
    recall = np.divide(TP.sum(axis=1), TP.sum(axis=1) + FN, out=np.full_like(FN, np.nan, dtype=float), where=(TP.sum(axis=1) + FN) != 0)
    return precision, recall, TP, FP, FN


def ff_gengrid(features: List[np.ndarray], param, quant: float = 1.0):
    """Generate Hough voting grid with quantization.

    features: list of arrays per frame; columns (id, x, y, alpha, ...)
    param.radius: used to extend bounds
    returns (votegrid: (Y,X,2), votegrid_pos: (N,2))
    """
    # find first non-empty frame
    idx = 0
    while idx < len(features) and (features[idx] is None or features[idx].size == 0):
        idx += 1
    if idx == len(features):
        raise ValueError("All features are empty")
    x_min = np.min(features[idx][:, 1])
    x_max = np.max(features[idx][:, 1])
    y_min = np.min(features[idx][:, 2])
    y_max = np.max(features[idx][:, 2])
    for fr in features:
        if fr is not None and fr.size:
            x_min = min(x_min, np.min(fr[:, 1]))
            x_max = max(x_max, np.max(fr[:, 1]))
            y_min = min(y_min, np.min(fr[:, 2]))
            y_max = max(y_max, np.max(fr[:, 2]))
    xmin = x_min - 2 * float(getattr(param, "radius", 1.0))
    xmax = x_max + 2 * float(getattr(param, "radius", 1.0))
    ymin = y_min - 2 * float(getattr(param, "radius", 1.0))
    ymax = y_max + 2 * float(getattr(param, "radius", 1.0))
    xs = np.arange(xmin, xmax + 1e-9, quant)
    ys = np.arange(ymin, ymax + 1e-9, quant)
    X, Y = np.meshgrid(xs, ys)
    votegrid = np.stack([X, Y], axis=2)
    votegrid_pos = votegrid.reshape(-1, 2)
    return votegrid, votegrid_pos


def ff_plot_person_tv(xx_feat, yy_feat=None, alpha_feat=None, col="b", scale=1.0, ax=None):
    """Plot person(s) from top view using matplotlib.

    Supports two call styles similar to MATLAB:
    - ff_plot_person_tv(features, col='b', scale=1)
      where features is Nx4 (ID, x, y, alpha)
    - ff_plot_person_tv(xx, yy, alpha, col='b', scale=1)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(xx_feat, np.ndarray) and xx_feat.ndim == 2 and xx_feat.shape[1] >= 4:
        arr = xx_feat
        xx = arr[:, 1]
        yy = arr[:, 2]
        alpha = arr[:, 3]
        IDs = arr[:, 0].astype(int)
        if yy_feat is not None:
            col = yy_feat
        if alpha_feat is not None:
            scale = alpha_feat
    else:
        xx = np.asarray(xx_feat)
        yy = np.asarray(yy_feat)
        alpha = np.asarray(alpha_feat)
        IDs = None

    bw = 0.5 * scale
    bh = 0.2 * scale
    hd = 0.15 * scale

    for i in range(len(xx)):
        e1 = Ellipse((xx[i], yy[i]), 2 * bw, 2 * bh, angle=(alpha[i] + np.pi / 2) * 180 / np.pi, fill=False, edgecolor=col)
        e2 = Ellipse((xx[i], yy[i]), 2 * hd, 2 * hd, angle=(alpha[i] + np.pi / 2) * 180 / np.pi, fill=False, edgecolor=col)
        ax.add_patch(e1)
        ax.add_patch(e2)
        ax.quiver(xx[i], yy[i], bw * np.cos(alpha[i]), bw * np.sin(alpha[i]), angles='xy', scale_units='xy', scale=1, color=col)
        if IDs is not None:
            ax.text(
                xx[i] - (bh * 1.5) * np.cos(alpha[i]),
                yy[i] - (bh * 1.5) * np.sin(alpha[i]),
                f"P_{{{IDs[i]}}}",
                color=col,
                ha="center",
                fontweight="bold",
            )
    ax.set_aspect("equal")
    return ax


def gc(f: np.ndarray, stride: float, MDL_in: float) -> np.ndarray:
    """Runs graph-cuts clustering as in GCFF/gc.m.

    f: Nx4 array where columns are [ID, x, y, alpha]
    Returns: labels vector of length N with 0-based label indices
    """
    f = np.asarray(f, dtype=float)

    def find_locs(feat, stride_):
        locs = np.zeros((feat.shape[0], 2), dtype=float)
        locs[:, 0] = feat[:, 1] + np.cos(feat[:, 3]) * stride_
        locs[:, 1] = feat[:, 2] + np.sin(feat[:, 3]) * stride_
        return locs

    def calc_distance(loc, feat, labels, mdl):
        u = np.unique(labels)
        distmat = np.zeros((loc.shape[0], len(u)), dtype=float)
        labels_out = labels.copy()
        for ii, lab in enumerate(u):
            means = loc[labels == lab, :].mean(axis=0)
            labels_out[labels == lab] = ii
            disp = feat[:, 1:3] - means.reshape(1, 2)
            d2 = np.sum((loc - means.reshape(1, 2)) ** 2, axis=1)
            distmat[:, ii] = d2
            mask = np.where(d2 < mdl)[0]
            for j in mask:
                for k in mask:
                    distk = np.linalg.norm(disp[k])
                    distj = np.linalg.norm(disp[j])
                    if distk > distj and distj > 0:
                        inner = float(disp[k].dot(disp[j]))
                        norma = distk * distj
                        if (inner / norma) > 0.75:
                            distmat[k, ii] += 100 ** (inner / norma * distk / distj)
        return distmat, labels_out

    locs = find_locs(f, stride)
    unary, _ = calc_distance(locs, f, np.arange(locs.shape[0]), MDL_in)
    N = f.shape[0]
    neigh = np.zeros((0, N), dtype=float)
    weight = np.zeros((0, N), dtype=float)
    seg = np.arange(N, dtype=float)
    segold = np.zeros_like(seg)
    MAX_ITER = 10
    numiter = 1
    while not np.array_equal(seg, segold) and numiter <= MAX_ITER:
        segold = seg.copy()
        mdl = np.ones(unary.shape[1], dtype=float) * MDL_in
        _, seg_new = expand_mex(unary, neigh, weight, seg, mdl, float("inf"))
        # Refit distances
        unary, seg = calc_distance(locs, f, seg_new.astype(int), MDL_in)
        numiter += 1
    return seg.astype(int)

