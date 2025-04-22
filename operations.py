import numpy as np

def grayscale(image):
    if image.ndim == 2:
        return image
    weights = np.array([0.114, 0.587, 0.299])
    gray = np.dot(image[..., :3], weights)
    return gray.astype(np.uint8)

def binarization(image, factor):
    gray = grayscale(image)
    P = np.mean(gray)
    threshold = P / factor
    binary = np.where(gray >= threshold, 255, 0)
    return binary.astype(np.uint8)
