import numpy as np

# -------------------- basic helpers -------------------------------------

def grayscale(image):
    if image.ndim == 2:
        return image
    weights = np.array([0.114, 0.587, 0.299])
    return (image[..., :3] @ weights).astype(np.uint8)


def binarization(image, factor):
    gray = grayscale(image)
    P = np.mean(gray)
    thr = P / factor
    out = np.where(gray < thr, 255, 0).astype(np.uint8)  # dark = foreground
    return out

def _logical_shift(img_bool: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return a view of img_bool shifted by (dy,dx) with zero padding."""
    h, w = img_bool.shape
    # y slice
    if dy < 0:
        ys = slice(0, h + dy)
        yt = slice(-dy, h)
    elif dy > 0:
        ys = slice(dy, h)
        yt = slice(0, h - dy)
    else:
        ys = yt = slice(0, h)
    # x slice
    if dx < 0:
        xs = slice(0, w + dx)
        xt = slice(-dx, w)
    elif dx > 0:
        xs = slice(dx, w)
        xt = slice(0, w - dx)
    else:
        xs = xt = slice(0, w)

    out = np.zeros_like(img_bool)
    out[yt, xt] = img_bool[ys, xs]
    return out


def dilatation(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    fg = (img == 255)
    ky, kx = np.where(kernel == 1)
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
    acc = np.zeros_like(fg)
    for dy, dx in zip(ky - cy, kx - cx):
        acc |= _logical_shift(fg, dy, dx)
    return acc.astype(np.uint8) * 255


def erosion(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    fg = (img == 255)
    ky, kx = np.where(kernel == 1)
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
    acc = np.ones_like(fg)
    for dy, dx in zip(ky - cy, kx - cx):
        acc &= _logical_shift(fg, dy, dx)
    return acc.astype(np.uint8) * 255

def opening(mask, k):
    return dilatation(erosion(mask, k), k)

def closing(mask, k):
    return erosion(dilatation(mask, k), k)

def refine_circle_and_score(bin_mask: np.ndarray):
    if bin_mask.ndim != 2:
        raise ValueError("binary mask must be 2‑D")

    fg = bin_mask // 255
    if fg.sum() == 0:
        h, w = bin_mask.shape
        return np.zeros((h, w), dtype=np.uint8), 0.0

    # centre via projections (same method as UI)
    proj_y = np.sum(fg, axis=1)
    rows = np.where(proj_y > 0)[0]
    proj_x = np.sum(fg, axis=0)
    cols = np.where(proj_x > 0)[0]
    cy = (rows[0] + rows[-1]) // 2
    cx = (cols[0] + cols[-1]) // 2
    ry = (rows[-1] - rows[0]) / 2.0
    rx = (cols[-1] - cols[0]) / 2.0
    r  = (ry + rx) / 2.0

    # build ideal circle mask (NumPy broadcasting)
    h, w = bin_mask.shape
    yy, xx = np.ogrid[:h, :w]
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
    ideal = circle.astype(np.uint8) * 255

    # Jaccard score (intersection / union)
    inter = np.logical_and(fg, circle).sum()
    union = np.logical_or(fg, circle).sum()
    score = inter / union if union else 0.0

    return ideal, score


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def extract_contour(binary):
    """
    Vectorized: external contour = foreground pixels (A==1)
    that have at least one 4‐neighbour background pixel.
    """
    A = (binary > 0).astype(np.uint8)
    up    = np.pad(A, ((1,0),(0,0)))[:-1, :]
    down  = np.pad(A, ((0,1),(0,0)))[1: , :]
    left  = np.pad(A, ((0,0),(1,0)))[:, :-1]
    right = np.pad(A, ((0,0),(0,1)))[:, 1: ]
    bg_n = (up==0)|(down==0)|(left==0)|(right==0)
    return (A==1) & bg_n

def circularity_and_completeness(binary):
    """
    Returns a combined score ∈ [0,1]:
      score = (min_r/max_r) * angular_coverage.

    - min_r/max_r penalizes radial irregularity.
    - angular_coverage = 1 - (largest empty angle)/(2π).
    """
    cnt = extract_contour(binary)
    ys, xs = np.nonzero(cnt)
    if xs.size < 5:
        return 0.0

    # centroid
    cx, cy = xs.mean(), ys.mean()
    # radii
    rs = np.hypot(xs - cx, ys - cy)
    rmin, rmax = rs.min(), rs.max()
    circ = rmin/rmax

    # angles in [0,2π)
    ang = np.mod(np.arctan2(ys-cy, xs-cx), 2*np.pi)
    sa = np.sort(ang)
    # compute gaps between successive angles (with wrap)
    gaps = np.diff(np.concatenate([sa, sa[:1] + 2*np.pi]))
    largest_gap = gaps.max()
    coverage = 1 - largest_gap/(2*np.pi)

    return float(circ * coverage)



# --- Usage Example ---
# import imageio
# bin1 = imageio.imread('/mnt/data/7d833d2a-537f-4cb3-ab47-e06c4293dcf4.png')
# bin2 = imageio.imread('/mnt/data/c0ee743b-9937-4778-908f-301bd0eaf16e.png')
# print("Score1:", circle_score(bin1))
# print("Score2:", circle_score(bin2))


# -------------------- hole-filling & closing ----------------------------
def close_and_fill(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Morphological closing (dilation→erosion) then flood-fill to fill holes."""
    closed = erosion(dilatation(binary, kernel), kernel)

    # flood-fill 0-pixels connected to the border → background mask
    h, w = closed.shape
    bg = np.zeros_like(closed, dtype=bool)
    stack = [(0, x) for x in range(w) if closed[0, x] == 0] + \
            [(h-1, x) for x in range(w) if closed[h-1, x] == 0] + \
            [(y, 0) for y in range(h) if closed[y, 0] == 0] + \
            [(y, w-1) for y in range(h) if closed[y, w-1] == 0]

    while stack:
        y, x = stack.pop()
        if bg[y, x]:
            continue
        bg[y, x] = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and closed[ny, nx] == 0 and not bg[ny, nx]:
                    stack.append((ny, nx))

    holes = (~bg) & (closed == 0)
    filled = closed.copy()
    filled[holes] = 255
    return filled


# -------------------- projection helpers --------------------------------

def _projection(binary, axis):
    proj = np.sum(binary // 255, axis=axis)
    max_val = proj.max()
    thresh = 0.7 * max_val
    idx = np.where(proj >= thresh)[0]
    if idx.size:
        length = idx[-1] - idx[0]
        center = (idx[-1] + idx[0]) // 2
        radius = length // 2
    else:
        center = proj.argmax()
        radius = 0
    return center, radius


def horizontal_projection(binary, factor):
    return _projection(binary, axis=1)


def vertical_projection(binary, factor):
    return _projection(binary, axis=0)


# ----------------------- polar helper -----------------------------------

def polar_to_cartesian(r, theta, cx, cy):
    x = int(r * np.cos(theta) + cx)
    y = int(r * np.sin(theta) + cy)
    return x, y