# Week 5 Rubric — Edge Detection (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Sobel gradients | 20 | x/y/magnitude computed and visualized; each explained. |
| Canny + steps | 25 | Canny run; smoothing and threshold effects shown; 4 steps explained. |
| auto-Canny | 20 | Median-based thresholds implemented and compared to manual. |
| DocScan milestone | 25 | Clean connected boundary edges on 3 real photos. |
| Reasoning | 10 | Blur-before-differentiate justified; gradient vs Canny distinguished. |

## Automatic deductions
- **−10** running Canny with no smoothing (noise floods the edge map) without noting it.
- **−8** applying a negative-valued edge kernel on uint8 (half the response clipped).
- **−6** confusing the (thick) gradient magnitude with (thin, connected) Canny edges.

## Instructor note
Ask a student to compute Sobel-x on a small patch by hand and say whether the sign indicates a
dark-to-light or light-to-dark edge. Understanding "gradient = edge" is the goal.
