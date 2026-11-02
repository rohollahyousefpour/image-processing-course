# Week 11 Deliverable — Homography & Perspective Rectification
# تحویلیِ هفتهٔ ۱۱ — هوموگرافی و تصحیحِ پرسپکتیو

**Due:** before Week 12. **Weight:** part of the 25% weekly grade + the **key DocScan milestone**.

## Tasks
1. **Simulate/obtain an angled photo.** Either photograph a meter/document at an angle, or
   warp a clean template into a trapezoid to get a reproducible "angled photo"
   (`../code/rectify.py` shows both).
2. **Manual rectification.** Detect (or take as given) the 4 corners, **order them TL,TR,BR,BL**,
   build the homography with `getPerspectiveTransform`, and `warpPerspective` to a flat scan.
3. **Automatic rectification.** Match ORB features to the template, estimate the homography with
   `findHomography(..., cv2.RANSAC)`, and warp. **Report the inlier count and reprojection RMSE.**
4. **Compare & reflect.** Put the angled photo, the manual scan, and the auto scan side by side.
   In 3–4 sentences: why does rectification make digit reading (Weeks 12–14) much easier and
   more accurate?

## Bonus
- Add a confidence gate: if RANSAC inliers < 15, reject and ask for a new photo.
- Feed your Week 9 `approxPolyDP` corners directly into the manual rectifier (close the loop).

## Deliverable
The three images (angled / manual / auto), the reported inlier count + RMSE, and your reflection.

See `RUBRIC.md`.
