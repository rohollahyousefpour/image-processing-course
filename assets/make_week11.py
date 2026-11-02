"""Week-11 figures: homography & perspective warp (the DocScan core)."""
import os, cv2, numpy as np
DATA = "E:/data"; OUT = "E:/programs/image-processing-course/week-11/assets"
os.makedirs(OUT, exist_ok=True)


def save(n, im):
    cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)


# a real water-meter numerator: this is exactly what DocScan must rectify
img = cv2.imread(f"{DATA}/abfa_image/numerators/num_1332.jpg")
img = cv2.resize(img, (360, 160))
save("original", img)

h, w = img.shape[:2]

# --- 1. simulate a perspective photo (as if snapped at an angle) ---
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
# push the top edge inward + tilt -> a trapezoid, like an angled phone shot
dst = np.float32([[40, 25], [w - 15, 5], [w - 5, h - 10], [25, h - 20]])
Hmat = cv2.getPerspectiveTransform(src, dst)
tilted = cv2.warpPerspective(img, Hmat, (w, h), borderValue=(30, 30, 30))
# draw the 4 corners we will "detect"
marked = tilted.copy()
for p in dst.astype(int):
    cv2.circle(marked, tuple(p), 5, (0, 0, 255), -1)
save("tilted", marked)

# --- 2. recover: 4 corners -> rectangle (the rectification) ---
Hinv = cv2.getPerspectiveTransform(dst, src)
recovered = cv2.warpPerspective(tilted, Hinv, (w, h))
save("recovered", recovered)

# --- 3. affine vs perspective: a grid shows the difference ---
grid = np.full((h, w, 3), 255, np.uint8)
for x in range(0, w, 30):
    cv2.line(grid, (x, 0), (x, h), (200, 120, 60), 1)
for y in range(0, h, 30):
    cv2.line(grid, (0, y), (w, y), (200, 120, 60), 1)
# affine: parallel lines stay parallel
A = cv2.getAffineTransform(src[:3], np.float32([[20, 20], [w - 10, 5], [w - 30, h - 5]]))
save("grid_affine", cv2.warpAffine(grid, A, (w, h), borderValue=(255, 255, 255)))
# perspective: parallel lines converge
save("grid_perspective", cv2.warpPerspective(grid, Hmat, (w, h), borderValue=(255, 255, 255)))

# --- 4. full DocScan-style pipeline using feature matching + RANSAC ---
# match the tilted photo back to the template automatically (no manual corners)
orb = cv2.ORB_create(1000)
g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(g1, None)
kp2, des2 = orb.detectAndCompute(g2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
knn = bf.knnMatch(des2, des1, k=2)
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
print("good matches for homography:", len(good))
if len(good) >= 4:
    pts2 = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hr, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    auto = cv2.warpPerspective(tilted, Hr, (w, h))
    save("auto_rectified", auto)
    print("RANSAC inliers:", int(mask.sum()), "of", len(good))
