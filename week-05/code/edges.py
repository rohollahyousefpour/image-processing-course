"""Gradient and Canny edge detection on a real image (Week 5)."""
import sys, cv2, numpy as np

def norm(x): return cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    b = cv2.GaussianBlur(g, (5, 5), 0)
    gx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
    outs = {
        "sobelx": norm(gx), "sobely": norm(gy),
        "sobelmag": norm(np.sqrt(gx**2 + gy**2)),
        "laplacian": norm(cv2.Laplacian(b, cv2.CV_64F, ksize=3)),
        "canny": cv2.Canny(b, 80, 160),
    }
    for n, im in outs.items():
        cv2.imwrite(f"out_{n}.png", im); print("wrote", n)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/abfa_image/numerators/num_1332.jpg")
