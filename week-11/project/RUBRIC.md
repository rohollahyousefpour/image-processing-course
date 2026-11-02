# Week 11 Rubric — Homography & Perspective Rectification (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Angled input | 12 | A reproducible angled photo obtained (real or simulated by warping). |
| Manual rectify | 26 | 4 corners **ordered correctly**; homography built; flat scan produced. |
| Automatic rectify | 28 | ORB + `findHomography(RANSAC)`; warp correct; **inliers + RMSE reported**. |
| Comparison & reasoning | 22 | Side-by-side shown; explains why rectification helps downstream digit reading. |
| Code quality | 12 | Runs on a real image; grayscale for features; sensible RANSAC threshold. |

## Automatic deductions
- **−12** corners not ordered → mirrored/rotated scan presented as correct.
- **−8** used `warpAffine` (affine) and claims it fixes perspective.
- **−6** reports "it worked" with no inlier count or reprojection error (no quantitative check).
- **−5** no `borderValue`/size handling → scan clipped or full of black holes without comment.

## Instructor note
Ask: "If the meter cover is slightly curved, will one homography give a perfect scan?" The
correct answer is no — a homography assumes a **flat plane**; curvature needs nonlinear dewarping.
A student who separates the planar assumption from the algorithm understands the model's limits.
