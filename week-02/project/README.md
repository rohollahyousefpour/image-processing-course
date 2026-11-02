# Week 2 Deliverable — Histograms & Enhancement
# تحویلیِ هفتهٔ ۲ — هیستوگرام و بهبود

**Due:** before Week 3. **Weight:** part of the 25% weekly grade + a DocScan milestone.

## Tasks (on real images from `E:/data`)
1. **Diagnose.** For 5 meter/plate images: plot the histogram and classify each
   (under/over-exposed, low-contrast, or fine). Use `../code/histogram.py` as a start.
2. **Enhance & compare.** Apply stretch, equalization, gamma, and CLAHE to a hard (dark or
   low-contrast) meter image. Show before/after and say which is best **for reading digits**
   (not just which looks nicest).
3. **Threshold.** Produce a clean binary of the digits with Otsu **and** adaptive thresholding.
   Pick foreground = white. Say which wins on a shadowed image and why.
4. **DocScan milestone.** For 3 hard meter photos, deliver a readable binary of the digits and
   document the enhancement + threshold combo that works.

## Deliverable
A short report with histograms, before/after images, the chosen pipeline, and your reasoning.

See `RUBRIC.md`.
