# Week 9 Rubric — Contours & Shape (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Find & measure | 22 | Contours found; area + bounding box per digit reported and drawn. |
| Shape filtering | 25 | Noise contours dropped by area + aspect ratio; count verified. |
| Crop & order | 28 | One cropped image per digit, correctly ordered left-to-right. |
| Descriptors | 15 | At least two descriptors (aspect ratio, extent, solidity, circularity) used sensibly. |
| Reasoning | 10 | Explains why the filter rule separates digits from noise/scratches. |

## Automatic deductions
- **−10** contours run on a poor binary (garbage in, garbage out) without noting it.
- **−8** digits cropped but not ordered (wrong number sequence).
- **−6** no shape filter (noise contours counted as digits).

## Instructor note
Ask which descriptor best separates a digit from a long scratch. If the student says aspect
ratio (the scratch is extreme), they understand shape filtering.
