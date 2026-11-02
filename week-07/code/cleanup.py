"""The DocScan cleanup: threshold -> open (specks) -> close (gaps) (Week 7)."""
import sys, cv2, numpy as np

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)     # remove specks first
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)  # then repair digits
    cv2.imwrite("out_clean_digits.png", clean)
    print("wrote out_clean_digits.png (each digit a solid blob, ready for contours)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/abfa_image/numerators/num_3494.jpg")
