# Week 10 Deliverable — Feature Detection & Matching
# تحویلیِ هفتهٔ ۱۰ — تشخیص و تطبیقِ ویژگی‌ها

**Due:** before Week 11. **Weight:** part of the 25% weekly grade + a DocScan milestone.

## Tasks (use a real image — a meter, plate, or the provided vehicle image)
1. **Detect.** Detect Harris corners and ORB keypoints on the image; visualize both
   (`../code/features.py`). Report the ORB keypoint count.
2. **Match under transformation.** Make a rotated + scaled copy of the image, detect ORB
   features on both, and match them. Draw the matches. Explain in one sentence why the
   features still match (invariance).
3. **Clean up.** Apply BOTH the cross-check and the Lowe ratio test. Report how many matches
   each keeps, and show that the ratio test removes ambiguous/wrong matches.
4. **DocScan note.** In 3–4 sentences: how will feature matching between a photographed page
   and a clean template let you align (rectify) the page in Week 11? Which OpenCV function
   turns matched points into a transform?

## Bonus
- Try SIFT (`cv2.SIFT_create()`, `NORM_L2`) and compare keypoint count + match quality to ORB.
- Match the meter image to a *cropped region* of itself (template search) instead of a rotation.

## Deliverable
The detection visualizations, the two match images (cross-check vs ratio-test), the reported
counts, and your DocScan note.

See `RUBRIC.md`.
