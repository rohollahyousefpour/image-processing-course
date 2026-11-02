"""Load an image and report its basic properties + color conversions (Week 1)."""
import sys
import cv2
import numpy as np


def inspect(path: str) -> None:
    img = cv2.imread(path)                       # BGR, uint8
    if img is None:
        raise SystemExit(f"could not read {path}")

    print(f"shape : {img.shape}   (H, W, channels)")
    print(f"dtype : {img.dtype}")
    print(f"min/max/mean : {img.min()} / {img.max()} / {img.mean():.1f}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # perceptual luminance
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(f"gray shape : {gray.shape}")
    print(f"hsv  shape : {hsv.shape}  (hue, saturation, value)")

    cv2.imwrite("out_gray.png", gray)             # PNG = lossless
    print("wrote out_gray.png")


if __name__ == "__main__":
    default = r"E:/data/abfa_image/numerators/num_1332.jpg"  # real dataset image
    inspect(sys.argv[1] if len(sys.argv) > 1 else default)
