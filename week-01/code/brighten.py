"""A correct brighten vs the naive (overflowing) one (Week 1)."""
import cv2
import numpy as np


def brighten_buggy(img: np.ndarray, k: int) -> np.ndarray:
    return img + k                                 # uint8 wraps around -> WRONG


def brighten_correct(img: np.ndarray, k: int) -> np.ndarray:
    # promote to int, add, clip to [0,255], back to uint8
    return np.clip(img.astype(np.int16) + k, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    px = np.array([[100, 150], [200, 250]], dtype=np.uint8)
    print("input      :\n", px)
    print("buggy  +60 :\n", brighten_buggy(px, 60))     # note the wrap-around
    print("correct+60 :\n", brighten_correct(px, 60))
    print("cv2.add+60 :\n", cv2.add(px, 60))            # saturates like 'correct'
