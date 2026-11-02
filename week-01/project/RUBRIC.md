# Week 1 Rubric — Images & the Pipeline (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Inspection | 25 | 3 real images loaded; shape/dtype/min/max/mean printed; grayscale + HSV shown. |
| Correct brighten | 25 | `brighten` avoids uint8 overflow; before/after shown; overflow explained correctly. |
| DocScan test image | 20 | A real photo captured, saved as PNG, properties reported, hard stages identified. |
| Color reasoning | 15 | Explains when grayscale suffices vs when color is needed, with a concrete example. |
| Clarity & correctness | 15 | Correct use of BGR/RGB, [y,x] indexing; no obvious traps; readable report. |

## Automatic deductions
- **−10** if intermediate results are saved as JPEG (use PNG).
- **−8** if BGR/RGB is confused (colors visibly wrong) or x/y indices swapped.
- **−8** if pixels are looped in Python instead of vectorized.

## Instructor note
Ask each student to show the overflow image and say what value `200 + 100` becomes in uint8.
If they can explain the wrap-around, they understand what a pixel really is.
