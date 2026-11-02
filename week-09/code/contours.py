"""Find, filter, crop and order digit contours (Week 9)."""
import sys, cv2, numpy as np

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 200 and 0.15 < w / h < 1.3:          # keep digit-like shapes
            digits.append((x, g[y:y + h, x:x + w]))

    digits = [img for x, img in sorted(digits)]        # left-to-right
    print(f"found {len(digits)} digit contours")
    for i, d in enumerate(digits):
        cv2.imwrite(f"out_digit_{i+1}.png", d)
    print("wrote one cropped image per digit, ordered")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/abfa_image/numerators/num_3494.jpg")
