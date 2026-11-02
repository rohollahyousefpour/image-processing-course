"""Denoise with several filters and measure PSNR vs the clean image (Week 4)."""
import sys, cv2, numpy as np
from noise import add_gaussian, add_salt_pepper

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 99.0 if mse == 0 else 10*np.log10(255.0**2/mse)

def main(path):
    clean = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    clean = cv2.resize(clean, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for label, noisy in [("gaussian", add_gaussian(clean, 20)),
                         ("salt-pepper", add_salt_pepper(clean, 0.06))]:
        print(f"\n== {label} noise (PSNR of noisy = {psnr(clean, noisy):.1f} dB) ==")
        outs = {
            "gaussianBlur": cv2.GaussianBlur(noisy, (5, 5), 0),
            "median": cv2.medianBlur(noisy, 3),
            "bilateral": cv2.bilateralFilter(noisy, 9, 60, 60),
            "nlm": cv2.fastNlMeansDenoising(noisy, None, 15, 7, 21),
        }
        for name, out in outs.items():
            print(f"  {name:14s} PSNR = {psnr(clean, out):.1f} dB")

if __name__ == "__main__":
    np.random.seed(0)
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/plate/unique_maxsize/11147L.jpg")
