"""Week-8 figures: segmentation on real images."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-08/assets"
os.makedirs(OUT, exist_ok=True)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

# --- thresholding methods on a meter ---
g = cv2.cvtColor(cv2.imread(f"{DATA}/abfa_image/numerators/num_3494.jpg"), cv2.COLOR_BGR2GRAY)
g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
save("gray", g)
_, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
save("otsu", otsu)
adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
save("adaptive", adapt)

# --- distance transform + watershed to separate the digits ---
b = otsu
b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
save("distance", cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
_, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
sure_bg = cv2.dilate(b, np.ones((3,3), np.uint8), iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)
n, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
color = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(color, markers)
seg = color.copy(); seg[markers == -1] = [0, 0, 255]  # boundaries in red
save("watershed", seg)

# --- connected components labelling (color each blob) ---
num, labels = cv2.connectedComponents(b)
label_hue = np.uint8(179*labels/max(1, labels.max()))
blank = 255*np.ones_like(label_hue)
labeled = cv2.merge([label_hue, blank, blank])
labeled = cv2.cvtColor(labeled, cv2.COLOR_HSV2BGR); labeled[label_hue == 0] = 0
save("components", labeled)
print("connected components (incl. bg):", num)

# --- k-means colour segmentation on a car image ---
car = cv2.imread(f"{DATA}/Vehicle_classes/car_images_/65cf04a8-121b-4175-a105-b3beba6453d2.jpg")
if car is not None:
    car = cv2.resize(car, (200, 200))
    save("car", car)
    Z = car.reshape((-1, 3)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, lbl, center = cv2.kmeans(Z, 4, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    seg2 = center.astype(np.uint8)[lbl.flatten()].reshape(car.shape)
    save("car_kmeans", seg2)
else:
    print("car image missing")
