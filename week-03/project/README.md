# Week 3 Deliverable — Spatial Filtering
# تحویلیِ هفتهٔ ۳ — فیلترِ مکانی

**Due:** before Week 4. **Weight:** part of the 25% weekly grade.

## Tasks (real meter/plate images from `E:/data`)
1. **Convolution from scratch.** Implement a 3x3 convolution in NumPy; verify it matches
   `cv2.filter2D` (max difference ~0). Explain your border handling.
2. **Blur comparison.** Box vs Gaussian at sizes 3, 7, 15. At what size do the digits/plate
   characters become unreadable? Discuss noise removal vs detail loss.
3. **Sharpen.** Unsharp-mask a plate; increase the strength until halos appear; report the limit.
4. **DocScan note.** Which smoother best prepares a noisy meter photo for thresholding
   (Gaussian / median / bilateral), and why?

Reference: `../code/convolve_scratch.py`, `../code/filters.py`.

See `RUBRIC.md`.
