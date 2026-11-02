# Week 10 Rubric — Feature Detection & Matching (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Detection | 22 | Harris corners and ORB keypoints detected and visualized; ORB count reported. |
| Matching under transform | 25 | Rotated/scaled copy matched; matches drawn; invariance explained correctly. |
| Match cleanup | 26 | Cross-check AND ratio test applied; counts reported; shows ratio test removes bad matches. |
| DocScan reasoning | 17 | Correctly explains features→correspondence→transform; names `findHomography`. |
| Code quality | 10 | Runs on a real image; grayscale conversion; sensible parameters. |

## Automatic deductions
- **−10** matches drawn without any filtering (raw noisy matches presented as final).
- **−8** used Hamming distance with a float descriptor (SIFT) or L2 with a binary one (ORB).
- **−6** claims features match "because ORB is good" without naming scale/rotation invariance.

## Instructor note
Ask the student what happens to matching if the page is photographed at 45° and half-lit.
A strong answer separates *detection repeatability* (fewer keypoints survive) from *descriptor
invariance* (surviving ones still match), and points to RANSAC for the geometric outliers.
