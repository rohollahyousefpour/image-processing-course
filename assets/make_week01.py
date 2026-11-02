"""Generate Week-1 figure images from the real E:/data datasets.
Run:  python make_week01.py
Outputs PNGs into ../week-01/assets/ for \\includegraphics in the slides/notes.
"""
import os
import cv2
import numpy as np

DATA = "E:/data"
RAW = "E:/programs/image-processing-course/assets/raw"
OUT = "E:/programs/image-processing-course/week-01/assets"
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# Representative raw images reused across the course.
SRC = {
    "meter":  f"{DATA}/abfa_image/numerators/num_1332.jpg",
    "meter2": f"{DATA}/abfa_image/numerators/num_3052.jpg",
    "plate":  f"{DATA}/plate/unique_maxsize/11147L.jpg",
}


def save(name, img):
    cv2.imwrite(f"{OUT}/{name}.png", img)
    print("wrote", name, img.shape)


def main():
    meter = cv2.imread(SRC["meter"])
    plate = cv2.imread(SRC["plate"])
    for k, p in SRC.items():
        im = cv2.imread(p)
        if im is not None:
            cv2.imwrite(f"{RAW}/{k}.png", im)

    # 1) original + grayscale + HSV value channel of the meter
    save("meter_bgr", meter)
    gray = cv2.cvtColor(meter, cv2.COLOR_BGR2GRAY)
    save("meter_gray", gray)
    hsv = cv2.cvtColor(meter, cv2.COLOR_BGR2HSV)
    save("meter_value", hsv[:, :, 2])

    # 2) brighten done right (clip) vs wrong (uint8 overflow) on the meter gray
    right = np.clip(gray.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    wrong = (gray + 60).astype(np.uint8)  # wraps around
    save("meter_bright_ok", right)
    save("meter_bright_overflow", wrong)

    # 3) the plate original + grayscale (variety)
    save("plate_bgr", plate)
    save("plate_gray", cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY))


if __name__ == "__main__":
    main()
