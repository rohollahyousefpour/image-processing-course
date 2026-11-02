"""Add known noise to a clean image (Week 4)."""
import cv2, numpy as np

def add_gaussian(img, sigma=20):
    n = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

def add_salt_pepper(img, p=0.06):
    out = img.copy()
    out[np.random.rand(*img.shape) < p/2] = 255
    out[np.random.rand(*img.shape) < p/2] = 0
    return out
