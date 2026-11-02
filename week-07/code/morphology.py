"""Morphological operations on a real binarized image (Week 7)."""
import sys, cv2, numpy as np

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outs = {
        "binary": b,
        "erode": cv2.erode(b, k),
        "dilate": cv2.dilate(b, k),
        "open": cv2.morphologyEx(b, cv2.MORPH_OPEN, k),
        "close": cv2.morphologyEx(b, cv2.MORPH_CLOSE, k),
        "gradient": cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k),
        "tophat": cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))),
    }
    for n, im in outs.items():
        cv2.imwrite(f"out_{n}.png", im); print("wrote", n)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/abfa_image/numerators/num_3494.jpg")
