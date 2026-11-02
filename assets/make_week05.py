"""Week-5 figures: edge detection on real images."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-05/assets"
os.makedirs(OUT, exist_ok=True)

meter = cv2.cvtColor(cv2.imread(f"{DATA}/abfa_image/numerators/num_1332.jpg"), cv2.COLOR_BGR2GRAY)
meter = cv2.resize(meter, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
plate = cv2.cvtColor(cv2.imread(f"{DATA}/plate/unique_maxsize/11147L.jpg"), cv2.COLOR_BGR2GRAY)
plate = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)
def norm(x): return cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def main():
    for tag, g in [("meter", meter), ("plate", plate)]:
        save(f"{tag}_gray", g)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)   # horizontal gradient
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)   # vertical gradient
        save(f"{tag}_sobelx", norm(gx))
        save(f"{tag}_sobely", norm(gy))
        mag = np.sqrt(gx**2 + gy**2)
        save(f"{tag}_sobelmag", norm(mag))
        save(f"{tag}_laplacian", norm(cv2.Laplacian(g, cv2.CV_64F, ksize=3)))
        save(f"{tag}_canny", cv2.Canny(g, 80, 160))
        # effect of blur before Canny (noise -> false edges)
        save(f"{tag}_canny_noblur", cv2.Canny(g, 40, 80))

if __name__=="__main__":
    main()
