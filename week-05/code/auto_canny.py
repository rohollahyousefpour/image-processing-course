"""Median-based automatic Canny thresholds (Week 5)."""
import sys, cv2, numpy as np

def auto_canny(img, sigma=0.33):
    m = np.median(img)
    lo = int(max(0, (1.0 - sigma) * m))
    hi = int(min(255, (1.0 + sigma) * m))
    return cv2.Canny(img, lo, hi), lo, hi

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else r"E:/data/plate/unique_maxsize/11147L.jpg"
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e, lo, hi = auto_canny(g)
    print(f"auto thresholds: low={lo} high={hi}, edge pixels={(e>0).sum()}")
    cv2.imwrite("out_autocanny.png", e)
