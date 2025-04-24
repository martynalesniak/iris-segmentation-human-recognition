import numpy as np
from numpy.lib._stride_tricks_impl import as_strided
import cv2


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
    h, w = img_bool.shape
    if dy < 0:
        ys = slice(0, h + dy)
        yt = slice(-dy, h)
    elif dy > 0:
        ys = slice(dy, h)
        yt = slice(0, h - dy)
    else:
        ys = yt = slice(0, h)
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

def extract_contour(binary):

    A = (binary > 0).astype(np.uint8)
    up    = np.pad(A, ((1,0),(0,0)))[:-1, :]
    down  = np.pad(A, ((0,1),(0,0)))[1: , :]
    left  = np.pad(A, ((0,0),(1,0)))[:, :-1]
    right = np.pad(A, ((0,0),(0,1)))[:, 1: ]
    bg_n = (up==0)|(down==0)|(left==0)|(right==0)
    return (A==1) & bg_n

def circularity_and_completeness(binary):

    cnt = extract_contour(binary)
    ys, xs = np.nonzero(cnt)
    if xs.size < 5:
        return 0.0

    cx, cy = xs.mean(), ys.mean()
    rs = np.hypot(xs - cx, ys - cy)
    rmin, rmax = rs.min(), rs.max()
    circ = rmin/rmax

    ang = np.mod(np.arctan2(ys-cy, xs-cx), 2*np.pi)
    sa = np.sort(ang)
    gaps = np.diff(np.concatenate([sa, sa[:1] + 2*np.pi]))
    largest_gap = gaps.max()
    coverage = 1 - largest_gap/(2*np.pi)

    return float(circ * coverage)

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


def _apply_kernel(image, kernel, mode='reflect'):

    kernel = np.array(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    if image.ndim == 2:

        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)
        H, W = image.shape
        shape = (H, W, kh, kw)
        strides = padded.strides * 2
        windows = as_strided(padded, shape=shape, strides=strides)
        result = np.einsum('ijkl,kl->ij', windows, kernel)
        return result.clip(0, 255).astype(np.uint8)
    elif image.ndim == 3:
        # Color image
        H, W, C = image.shape
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=mode)
        output = np.empty((H, W, C), dtype=np.float32)
        for c in range(C):
            channel = padded[:, :, c]
            shape = (H, W, kh, kw)
            strides = channel.strides * 2
            windows = as_strided(channel, shape=shape, strides=strides)
            result = np.einsum('ijkl,kl->ij', windows, kernel)
            output[:, :, c] = result
        return output.clip(0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape.")


def detect_iris_from_pupil(gray_image, pupil_center, pupil_radius):
    h, w = gray_image.shape
    max_search_radius = int(pupil_radius * 3.2)

    num_angles = 36
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    iris_radius_estimates = []

    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        intensity_profile = []
        radius_values = []

        for r in range(pupil_radius + 5, max_search_radius, 2):
            x = int(pupil_center[0] + dx * r)
            y = int(pupil_center[1] + dy * r)

            if 0 <= x < w and 0 <= y < h:
                intensity = gray_image[y, x]
                intensity_profile.append(intensity)
                radius_values.append(r)

        if len(intensity_profile) > 10:
            smoothed = np.convolve(intensity_profile, np.ones(5)/5, mode='valid')
            gradient = np.gradient(smoothed)

            search_range = len(gradient) // 3
            if search_range < len(gradient):
                max_grad_idx = search_range + np.argmax(np.abs(gradient[search_range:]))
                if max_grad_idx < len(radius_values):
                    iris_radius_estimates.append(radius_values[max_grad_idx])

    if iris_radius_estimates:
        sorted_estimates = sorted(iris_radius_estimates)
        num_to_remove = len(sorted_estimates) // 5
        filtered_estimates = sorted_estimates[num_to_remove:-num_to_remove] if num_to_remove > 0 else sorted_estimates

        iris_radius = int(np.mean(filtered_estimates))
        return iris_radius
    else:
        return int(pupil_radius * 2.5)


# def remove_eyelids(image, iris_center, iris_radius):
#     result = image.copy()
#     h, w = image.shape[:2]
#
#     eyelid_mask = np.zeros((h, w), dtype=np.uint8)
#     angle_step = 20
#
#     for angle in range(-60, 61, angle_step):
#         rad = np.radians(angle)
#         start_x = int(iris_center[0] + (iris_radius * 0.3) * np.cos(rad))
#         start_y = int(iris_center[1] + (iris_radius * 0.3) * np.sin(rad))
#         end_x = int(iris_center[0] + iris_radius * np.cos(rad))
#         end_y = int(iris_center[1] + iris_radius * np.sin(rad))
#
#         cv2.line(eyelid_mask, (start_x, start_y), (end_x, end_y), 255, 2)
#
#     for angle in range(120, 241, angle_step):
#         rad = np.radians(angle)
#         start_x = int(iris_center[0] + (iris_radius * 0.3) * np.cos(rad))
#         start_y = int(iris_center[1] + (iris_radius * 0.3) * np.sin(rad))
#         end_x = int(iris_center[0] + iris_radius * np.cos(rad))
#         end_y = int(iris_center[1] + iris_radius * np.sin(rad))
#
#         cv2.line(eyelid_mask, (start_x, start_y), (end_x, end_y), 255, 2)
#
#     kernel = np.ones((7, 7), np.uint8)
#     eyelid_mask = cv2.dilate(eyelid_mask, kernel, iterations=3)
#
#     if len(image.shape) > 2:
#         eyelid_mask_color = cv2.merge([eyelid_mask, eyelid_mask, eyelid_mask])
#
#         white_bg = np.ones_like(result) * 255
#         result = np.where(eyelid_mask_color > 0, white_bg, result)
#     else:
#         result[eyelid_mask > 0] = 255
#
#     return result, cv2.bitwise_not(eyelid_mask)


def largest_connected_component(bin_img):

    h, w = bin_img.shape
    visited = np.zeros_like(bin_img, dtype=bool)
    best_sz = 0; best_mask = None
    stack = []

    for y, x in zip(*np.where(bin_img)):
        if visited[y, x]:
            continue
        cur_mask = []
        stack.append((y, x)); visited[y, x] = True
        while stack:
            cy, cx = stack.pop(); cur_mask.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and bin_img[ny, nx]:
                        visited[ny, nx] = True; stack.append((ny, nx))
        if len(cur_mask) > best_sz:
            best_sz = len(cur_mask)
            best_mask = cur_mask
    mask = np.zeros_like(bin_img, dtype=bool)
    if best_mask is not None:
        ys, xs = zip(*best_mask)
        mask[ys, xs] = True
    return mask

def fit_circle_bottom_anchor(mask, cover_thresh = 0.80, r_min = 5, r_max = 400, step = 1, use_perimeter = True, return_largest = True):

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    yb = ys.max()
    xb = int(np.round(xs[ys == yb].mean()))

    H, W = mask.shape
    if r_max is None:
        r_max = int(min(yb, H-1, W-1))

    yy, xx = np.indices(mask.shape)
    best = None
    for r in range(r_min, r_max + 1, step):
        yc = yb - r
        if yc < 0:
            break

        if use_perimeter:
            band = np.abs(np.hypot(xx - xb, yy - yc) - r) <= 0.5
        else:
            band = np.hypot(xx - xb, yy - yc) <= r

        overlap = (band & (mask > 0)).sum() / band.sum()

        if overlap >= cover_thresh:
            best = (xb, yc, r)
            if not return_largest:
                return best

    return best
