"""
Week 11 - Perspective rectification (the DocScan core), runnable.

Two ways to turn an angled photo into a flat scan:
  (A) manual 4-corner rectification with getPerspectiveTransform
  (B) automatic alignment via ORB feature matching + findHomography(RANSAC)

Usage:
    python rectify.py [path-to-image]

Defaults to a real water-meter numerator from E:/data. Since we don't have a
真 angled photo, we SIMULATE one by warping the template, then recover it - so
the pipeline is fully reproducible offline.
"""
import sys, cv2, numpy as np

DEFAULT = "E:/data/abfa_image/numerators/num_1332.jpg"


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL using the x+/-y trick."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)],   # TL smallest x+y
                       pts[np.argmin(d)],   # TR smallest x-y
                       pts[np.argmax(s)],   # BR largest  x+y
                       pts[np.argmax(d)]])  # BL largest  x-y


def main(path):
    img = cv2.imread(path)
    if img is None:
        print("could not read", path); return
    img = cv2.resize(img, (360, 160))
    h, w = img.shape[:2]

    # --- simulate an angled photo ---
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 25], [w - 15, 5], [w - 5, h - 10], [25, h - 20]])
    Hmat = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(img, Hmat, (w, h), borderValue=(30, 30, 30))
    cv2.imwrite("out_photo.png", photo)

    # === (A) manual: we "detected" the 4 corners = dst; rectify them ===
    corners = order_corners(dst)
    rect_dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    Hrect = cv2.getPerspectiveTransform(corners, rect_dst)
    manual = cv2.warpPerspective(photo, Hrect, (w, h))
    cv2.imwrite("out_manual.png", manual)

    # === (B) automatic: ORB + findHomography(RANSAC) ===
    orb = cv2.ORB_create(1000)
    g_tpl = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_pho = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g_pho, None)
    k2, d2 = orb.detectAndCompute(g_tpl, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    print(f"good matches: {len(good)}")
    if len(good) >= 4:
        p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hr, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        auto = cv2.warpPerspective(photo, Hr, (w, h))
        cv2.imwrite("out_auto.png", auto)
        inl = int(mask.sum())
        # reprojection error on inliers
        proj = cv2.perspectiveTransform(p1, Hr)
        err = np.linalg.norm((proj - p2).reshape(-1, 2), axis=1)
        rmse = float(np.sqrt((err[mask.ravel() == 1] ** 2).mean()))
        print(f"RANSAC inliers: {inl}/{len(good)}   reprojection RMSE: {rmse:.2f}px")
        if inl < 15:
            print("WARNING: few inliers - homography may be unreliable.")
    print("Wrote out_photo.png (angled), out_manual.png, out_auto.png (both rectified).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
