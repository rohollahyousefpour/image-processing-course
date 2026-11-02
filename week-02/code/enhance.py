"""Compare contrast enhancements on a real meter image (Week 2)."""
import sys, cv2, numpy as np

def gamma(g, gm):
    lut = np.array([((i/255.0)**(1.0/gm))*255 for i in range(256)], np.uint8)
    return cv2.LUT(g, lut)

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    outs = {
        "01_gray": g,
        "02_stretch": cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX),
        "03_equalized": cv2.equalizeHist(g),
        "04_gamma2": gamma(g, 2.0),
        "05_clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g),
    }
    for name, im in outs.items():
        cv2.imwrite(f"out_{name}.png", im)
        print("wrote", name)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else r"E:/data/abfa_image/numerators/num_3494.jpg")
