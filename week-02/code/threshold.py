"""Global Otsu vs adaptive thresholding (Week 2)."""
import sys, cv2

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    t, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("Otsu chose threshold =", t)
    adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 10)
    cv2.imwrite("out_otsu.png", otsu)
    cv2.imwrite("out_adaptive.png", adapt)
    print("wrote out_otsu.png, out_adaptive.png (foreground = white digits)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else r"E:/data/abfa_image/numerators/num_3494.jpg")
