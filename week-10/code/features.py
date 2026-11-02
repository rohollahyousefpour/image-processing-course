"""
Week 10 - Feature detection & matching (runnable).
Detect Harris corners and ORB keypoints, then match ORB features between an
image and a rotated/scaled copy - and show how the ratio test cleans up matches.

Usage:
    python features.py [path-to-image]

Defaults to a real vehicle image from E:/data.
"""
import sys, cv2, numpy as np

DEFAULT = "E:/data/Vehicle_classes/car_images_/65cf04a8-121b-4175-a105-b3beba6453d2.jpg"
OUT = "."


def save(name, im):
    cv2.imwrite(f"{OUT}/{name}.png", im)
    print("wrote", name)


def main(path):
    img = cv2.imread(path)
    if img is None:
        print("could not read", path); return
    img = cv2.resize(img, (300, 300))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Harris corners ---
    harris = cv2.cornerHarris(np.float32(g), 2, 3, 0.04)
    canvas = img.copy()
    canvas[harris > 0.01 * harris.max()] = [0, 0, 255]
    save("out_harris", canvas)

    # --- ORB keypoints ---
    orb = cv2.ORB_create(500)
    kp = orb.detect(g, None)
    save("out_orb", cv2.drawKeypoints(
        img, kp, None, color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS))
    print("ORB keypoints:", len(kp))

    # --- matching to a rotated + scaled copy ---
    M = cv2.getRotationMatrix2D((150, 150), 30, 0.8)
    img2 = cv2.warpAffine(img, M, (300, 300))
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(g, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    # cross-check matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cross = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    save("out_matches_crosscheck", cv2.drawMatches(
        img, kp1, img2, kp2, cross[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    # ratio-test matching (Lowe)
    bf2 = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf2.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    save("out_matches_ratio", cv2.drawMatches(
        img, kp1, img2, kp2, good[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    print(f"cross-check matches: {len(cross)}")
    print(f"raw knn pairs: {len(knn)}  ->  ratio-test kept: {len(good)}")
    print("Lesson: the ratio test discards ambiguous matches, "
          "leaving a cleaner set to estimate a transform (Week 11).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
