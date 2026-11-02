"""Histogram + exposure diagnosis (Week 2)."""
import sys, cv2, numpy as np

def diagnose(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    lo, hi, mean = int(g.min()), int(g.max()), float(g.mean())
    spread = hi - lo
    print(f"min={lo} max={hi} mean={mean:.1f} spread={spread}")
    if mean < 80:   print("-> under-exposed (dark): try gamma>1 or stretch")
    elif mean > 180: print("-> over-exposed (bright)")
    if spread < 120: print("-> low contrast: try stretch / CLAHE")
    else: print("-> contrast looks ok")

if __name__ == "__main__":
    diagnose(sys.argv[1] if len(sys.argv) > 1
             else r"E:/data/abfa_image/numerators/num_3494.jpg")
