import numpy as np


def plotoverlap(ov: np.ndarray, out: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Python port of plotoverlap.m returning a montage image.

    ov: points x hyp annotation matrix
    out: points x hyp outlier matrix
    im: (rows, cols, 3) image (uint8 or float 0..255)
    returns: image array representing [edge | edge2 | out | im_masked]
    """
    rows, cols = im.shape[:2]
    N = rows * cols
    ov = ov.reshape(N, -1)
    out = out.reshape(N, -1)
    ov_map = ov.sum(axis=1).reshape(rows, cols)
    out_map = (out.sum(axis=1) != 0).reshape(rows, cols)

    edge = 2 * (ov_map > 1) + (ov_map == 1)
    edge = (edge / 2.0).astype(float)
    edge = np.repeat(edge[:, :, None], 3, axis=2)

    out_rgb = np.repeat(out_map[:, :, None].astype(float), 3, axis=2)

    imf = im.astype(float) / 255.0
    im_mask2 = (ov_map > 1)
    im_mask1 = (ov_map >= 1)
    edge2 = imf.copy()
    im2 = imf.copy()
    im2[~im_mask1] = 0
    im3 = imf.copy()
    im3[~im_mask2] = 0

    # Concatenate horizontally: [edge, edge2, out_rgb, im3]
    return np.concatenate([edge, edge2, out_rgb, im3], axis=1)


def plot_patches_heights(W2d: np.ndarray,
                         points: list,
                         color_matrix: np.ndarray | None = None,
                         npoints: np.ndarray | None = None,
                         ax=None):
    """Port of plotPatchesHeights.m using matplotlib 3D.

    - W2d: shape (2, N) array of 2D coordinates.
    - points: list of 1D index arrays (1-based or 0-based). Each entry is a group to plot.
    - color_matrix: (M, 3) RGB in [0,1] or [0,255]; if None/empty, random colors.
    - npoints: optional 1D indices of additional points to plot as '+' markers.
    - ax: optional matplotlib 3D axes; created if None. Returns ax.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    W2d = np.asarray(W2d, dtype=float)
    assert W2d.shape[0] == 2, "W2d must be (2, N)"
    N = W2d.shape[1]

    Nm = len(points)
    if color_matrix is None or (hasattr(color_matrix, "size") and color_matrix.size == 0):
        rng = np.random.default_rng(0)
        color_matrix = rng.random((Nm, 3))
    color_matrix = np.asarray(color_matrix, dtype=float)
    if color_matrix.max() > 1.0:
        color_matrix = color_matrix / 255.0

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    zspan = (W2d.max() - W2d.min())
    dh = 5.0 * (zspan / max(Nm, 1))
    h = np.zeros(N, dtype=float)

    # normalize indices to 0-based
    def to_zero_based(idx):
        arr = np.asarray(idx).astype(int)
        if arr.size == 0:
            return arr
        if arr.min() == 0:
            return arr
        return arr - 1

    # First group
    p0 = to_zero_based(points[0])
    if p0.size:
        ax.scatter(W2d[0, p0], W2d[1, p0], h[p0], s=100, c=[color_matrix[0]], edgecolors=[color_matrix[0]], linewidths=1, marker='o')
    # Remaining groups with increasing height
    for m in range(1, Nm):
        h += dh
        p = to_zero_based(points[m])
        if p.size:
            ax.scatter(W2d[0, p], W2d[1, p], h[p], s=100, c=[color_matrix[m]], edgecolors=[color_matrix[m]], linewidths=1, marker='o')

    if npoints is not None and np.asarray(npoints).size:
        h += dh
        npz = to_zero_based(npoints)
        ax.scatter(W2d[0, npz], W2d[1, npz], h[npz], s=100, c='k', marker='+', linewidths=1)

    ax.set_box_aspect((1, 1, 0.3))
    ax.set_aspect('auto')
    return ax

