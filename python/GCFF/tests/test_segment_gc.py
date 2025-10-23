import os
import sys
import numpy as np

# Ensure repo root is on sys.path so that 'GCFF' package resolves when running this file directly
_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from graphopt.segment_gc import segment_gc
from graphopt import plotoverlap


def make_synthetic(width=20, height=20, nlabels=2):
    W, H = width, height
    N = W * H
    # Image: two color regions
    im = np.zeros((W, H, 3), dtype=np.uint8)
    im[:, : H // 2, :] = [200, 50, 50]
    im[:, H // 2 :, :] = [50, 200, 50]

    # Unary: squared distance to two centers
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W))
    centers = np.array([[W // 2, H // 4], [W // 2, 3 * H // 4]], dtype=float)
    unary = np.empty((N, nlabels), dtype=float)
    coords = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    for l in range(nlabels):
        d2 = np.sum((coords - centers[l]) ** 2, axis=1)
        unary[:, l] = d2

    # Initial labels: alternating or random
    current_labels = (coords[:, 1] >= H // 2).astype(float) + 1  # 1 or 2 (MATLAB style)
    pairc = 1.0
    MDL = 150.0
    return unary, current_labels, pairc, MDL, im


def test_segment_gc_runs():
    unary, current_labels, pairc, MDL, im = make_synthetic()
    ov, labelling, out = segment_gc(unary, current_labels, pairc, MDL, im)
    assert labelling.shape == (im.shape[0], im.shape[1])
    assert ov.shape[0] == im.shape[0] * im.shape[1]
    assert ov.shape[1] == unary.shape[1]
    # Expect at least 2 labels present
    assert len(np.unique(labelling)) >= 2


if __name__ == "__main__":
    test_segment_gc_runs()
    print("segment_gc sanity test passed.")
