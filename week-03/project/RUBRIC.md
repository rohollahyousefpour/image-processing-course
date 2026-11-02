# Week 3 Rubric — Spatial Filtering (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Convolution from scratch | 30 | Correct 3x3 conv; matches cv2.filter2D; border handling explained. |
| Blur comparison | 25 | Box vs Gaussian at multiple sizes; the readability trade-off quantified. |
| Sharpening | 20 | Unsharp mask applied; halo limit found and shown. |
| DocScan reasoning | 15 | Right smoother chosen for a noisy photo, with justification. |
| Correctness | 10 | float used for edge-type kernels; kernels normalized; no in-place bug. |

## Automatic deductions
- **−10** blur kernel not normalized (output darker/brighter as a side effect).
- **−8** writing convolution output in place (corrupts later pixels).
- **−6** applying a negative-valued kernel on uint8 (clipping loses half the response).

## Instructor note
Have a student convolve a 3x3 patch by the sharpen kernel by hand and state whether it is a
low-pass or high-pass filter. Understanding "the kernel is the algorithm" is the goal.
