"""Week-10 figures: feature detection & matching on real images."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-10/assets"
os.makedirs(OUT, exist_ok=True)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

img = cv2.imread(f"{DATA}/Vehicle_classes/car_images_/65cf04a8-121b-4175-a105-b3beba6453d2.jpg")
img = cv2.resize(img, (300, 300))
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
save("gray", g)

# Harris corners
harris = cv2.cornerHarris(np.float32(g), 2, 3, 0.04)
canvas = img.copy(); canvas[harris > 0.01*harris.max()] = [0, 0, 255]
save("harris", canvas)

# ORB keypoints
orb = cv2.ORB_create(400)
kp = orb.detect(g, None)
save("orb_keypoints", cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
     flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS))

# matching: original vs a rotated+scaled version (shows invariance)
M = cv2.getRotationMatrix2D((150, 150), 30, 0.8)
img2 = cv2.warpAffine(img, M, (300, 300))
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(g, None)
kp2, des2 = orb.detectAndCompute(g2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:30]
match_img = cv2.drawMatches(img, kp1, img2, kp2, matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
save("matches", match_img)
print("total ORB keypoints:", len(kp), "good matches drawn:", len(matches))
