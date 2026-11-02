"""Compare smoothing/sharpening filters on a real image (Week 3)."""
import sys, cv2, numpy as np

def main(p):
    g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    outs = {
        "box": cv2.blur(g, (5, 5)),
        "gauss": cv2.GaussianBlur(g, (7, 7), 1.5),
        "median": cv2.medianBlur(g, 3),
        "bilateral": cv2.bilateralFilter(g, 9, 50, 50),
    }
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    outs["sharp"] = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    for n, im in outs.items():
        cv2.imwrite(f"out_{n}.png", im); print("wrote", n)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/plate/unique_maxsize/11147L.jpg")
