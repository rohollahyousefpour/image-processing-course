"""Separate touching digits with distance transform + watershed (Week 8)."""
import sys, cv2, numpy as np

def main(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0); fg = np.uint8(fg)
    bg = cv2.dilate(b, k, iterations=3)
    unknown = cv2.subtract(bg, fg)
    n, markers = cv2.connectedComponents(fg)
    markers = markers + 1; markers[unknown == 255] = 0
    color = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    color[markers == -1] = (0, 0, 255)
    cv2.imwrite("out_watershed.png", color)
    print(f"separated into {n-1} digit regions")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/abfa_image/numerators/num_3494.jpg")
