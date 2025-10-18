import numpy as np

from .graphcut import allgc


def _median3(img: np.ndarray) -> np.ndarray:
    """3x3 median filter per-channel. Falls back to no-op if import fails."""
    try:
        from scipy.signal import medfilt2d  # type: ignore

        if img.ndim == 2:
            return medfilt2d(img, kernel_size=3)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[..., c] = medfilt2d(img[..., c], kernel_size=3)
        return out
    except Exception:
        # Lightweight fallback: manual pad + median of 3x3 using strides
        if img.ndim == 2:
            img_c = img[..., None]
        else:
            img_c = img
        pad = np.pad(img_c, ((1, 1), (1, 1), (0, 0)), mode="edge")
        H, W, C = img_c.shape
        out = np.empty_like(img_c)
        for y in range(H):
            for x in range(W):
                patch = pad[y : y + 3, x : x + 3]
                out[y, x] = np.median(patch.reshape(-1, C), axis=0)
        return out[..., 0] if img.ndim == 2 else out


def _build_8nbh_indices(rows: int, cols: int) -> np.ndarray:
    """Return 8 x (rows*cols) neighbor indices (1-based like MATLAB)."""
    idx = np.arange(1, rows * cols + 1, dtype=np.int64).reshape(rows, cols)
    pad = np.pad(idx, ((1, 1), (1, 1)), mode="edge")
    pair = np.zeros((8, rows * cols), dtype=np.int64)
    # Exact translations of MATLAB slices to Python 0-based indexing
    pair[0, :] = pad[0:rows, 1:cols + 1].reshape(-1)      # 1:rows, 2:cols+1
    pair[1, :] = pad[1:rows + 1, 0:cols].reshape(-1)      # 2:rows+1, 1:cols
    pair[2, :] = pad[2:rows + 2, 1:cols + 1].reshape(-1)  # 3:rows+2, 2:cols+1
    pair[3, :] = pad[1:rows + 1, 2:cols + 2].reshape(-1)  # 2:rows+1, 3:cols+2
    pair[4, :] = pad[0:rows, 0:cols].reshape(-1)          # 1:rows, 1:cols
    pair[5, :] = pad[2:rows + 2, 0:cols].reshape(-1)      # 3:rows+2, 1:cols
    pair[6, :] = pad[0:rows, 2:cols + 2].reshape(-1)      # 1:rows, 3:cols+2
    pair[7, :] = pad[2:rows + 2, 2:cols + 2].reshape(-1)  # 3:rows+2, 3:cols+2
    return pair


def _compute_pair_costs_from_image(pair: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Compute Nx8 pair costs from color differences as in MATLAB code, then exp/scale.

    pair: 8 x N (1-based indices)
    im: (W, H, 3) uint8 or float
    returns: 8 x N pair costs (transposed later as needed)
    """
    W, H = im.shape[:2]
    N = W * H
    img = _median3(im.astype(np.float64))
    flat = img.reshape(N, -1)  # N x 3
    # Reference per pixel
    ref = np.repeat(flat[:, None, :], 8, axis=1)  # N x 8 x 3
    # Neighbor colors: pair is 1-based, convert to 0-based
    n_inds = np.maximum(pair, 1) - 1  # 8 x N -> 8 x N
    neigh = flat[n_inds.T, :]  # N x 8 x 3
    diff = np.abs(neigh - ref)
    pcost = -np.sum(diff, axis=2)  # N x 8
    pcost = np.exp(pcost / 20.0)
    return pcost.T  # 8 x N


def segment_gc(unary: np.ndarray,
               current_labels: np.ndarray,
               pairc: float,
               MDL: float,
               im: np.ndarray):
    """Python port of GCFF/graphopt/segment_gc.m.

    Inputs mirror the MATLAB function. Returns (overlap, labelling, out) where
    - overlap: points x hyp annotation matrix (as from allgc)
    - labelling: (W, H) label image (1-based like MATLAB on return)
    - out: points x hyp outlier indicator matrix (currently zeros)
    """
    unary = np.asarray(unary, dtype=float)
    if unary.ndim == 3:
        # MATLAB sometimes passes (rows, cols, L)
        rows, cols, L = unary.shape
        unary2 = unary.reshape(rows * cols, L)
    else:
        unary2 = unary
        # If im given, deduce rows, cols
        rows, cols = int(im.shape[0]), int(im.shape[1])

    current_labels = np.asarray(current_labels, dtype=float).reshape(-1)
    if current_labels.size != rows * cols:
        raise ValueError("current_labels size must equal number of pixels")

    pair = _build_8nbh_indices(rows, cols)  # 8 x N (1-based)

    if im is None or im.size == 0:
        pcost = np.ones((8, rows * cols), dtype=float) * float(pairc)
    else:
        pcost = _compute_pair_costs_from_image(pair, im)

    # Following MATLAB: (1 + pcost)' * 10 for pair costs, lambda=1, hyp weights=150, thresh=220
    pair_costs = (1.0 + pcost) * 10.0
    lam = 1.0
    hyp = unary2.shape[1]
    hypweight = np.full((hyp,), 150.0, dtype=float)
    thresh = 220.0

    overlap, lab_vec, out = allgc(unary2, pair, pair, pair_costs, current_labels - 1, lam, hypweight, thresh)

    # MATLAB increments labels by 1
    labelling = (lab_vec + 1).astype(int).reshape(rows, cols)
    return overlap, labelling, out


