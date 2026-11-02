# Week 8 Rubric — Segmentation (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Otsu vs adaptive | 20 | Both applied; winner per lighting justified. |
| k-means colour | 20 | k=2/4/8 shown; over/under-segmentation + role of k discussed. |
| Watershed separation | 25 | Touching digits separated with marker-controlled watershed. |
| Connected components | 15 | Each digit labelled; count correct. |
| DocScan milestone | 20 | One ordered cropped image per digit. |

## Automatic deductions
- **−10** watershed without markers (over-segmentation).
- **−8** global Otsu on a clearly shadowed image.
- **−6** digits cropped but not sorted left-to-right (wrong number order).

## Instructor note
Ask which method separates two touching digits. If the student says watershed (with a distance
transform), they understand why thresholding alone can't do it.
