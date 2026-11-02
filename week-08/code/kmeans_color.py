"""k-means colour segmentation (Week 8)."""
import sys, cv2, numpy as np

def kmeans_seg(img, k):
    Z = img.reshape((-1, 3)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, lbl, center = cv2.kmeans(Z, k, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    return center.astype(np.uint8)[lbl.flatten()].reshape(img.shape)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else \
        r"E:/data/Vehicle_classes/car_images_/65cf04a8-121b-4175-a105-b3beba6453d2.jpg"
    img = cv2.resize(cv2.imread(p), (200, 200))
    for k in (2, 4, 8):
        cv2.imwrite(f"out_kmeans_k{k}.png", kmeans_seg(img, k)); print("wrote k =", k)
