import numpy as np
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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

def erosion(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.array_equal(region[kernel == 1], 255 * np.ones(np.sum(kernel))):
                output[i, j] = 255
            else:
                output[i, j] = 0
    return output

def dilatation(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0
    return output

def horizontal_projection(image, factor):
    binary = binarization(image, factor)
    binary[binary == 0] = 1
    binary[binary == 255] = 0

    projection = np.sum(binary, axis=1)
    max_val = np.max(projection)
    threshold = 0.7 * max_val  # 70% maksimum jako próg

    indices = np.where(projection >= threshold)[0]
    if len(indices) > 0:
        height = indices[-1] - indices[0]
        center_y = (indices[-1] + indices[0]) // 2
        radius_y = height // 2
    else:
        center_y = np.argmax(projection)
        radius_y = 0  # fallback

    return center_y, radius_y

def vertical_projection(image, factor):
    binary = binarization(image, factor)
    binary[binary == 0] = 1
    binary[binary == 255] = 0

    projection = np.sum(binary, axis=0)
    max_val = np.max(projection)
    threshold = 0.7 * max_val

    indices = np.where(projection >= threshold)[0]
    if len(indices) > 0:
        width = indices[-1] - indices[0]
        center_x = (indices[-1] + indices[0]) // 2
        radius_x = width // 2
    else:
        center_x = np.argmax(projection)
        radius_x = 0

    return center_x, radius_x



def polar_to_cartesian(r, theta, center_x, center_y):
    x = int(r * np.cos(theta) + center_x)
    y = int(r * np.sin(theta) + center_y)
    return x, y
