# Week 5 Deliverable — Edge Detection
# تحویلیِ هفتهٔ ۵ — تشخیصِ لبه

**Due:** before Week 6. **Weight:** part of the 25% weekly grade + a DocScan milestone.

## Tasks (real meter/plate images from `E:/data`)
1. **Gradients.** Compute Sobel-x, Sobel-y, and gradient magnitude; visualize each. Explain
   what each highlights.
2. **Canny + steps.** Run Canny; show the effect of (a) smoothing on vs off, and (b) the two
   thresholds. Explain non-maximum suppression and hysteresis in your own words.
3. **auto-Canny.** Implement median-based thresholds (`../code/auto_canny.py`); compare to
   hand-tuned values across a few images.
4. **DocScan milestone.** Produce clean, connected edges of the page/meter boundary on 3 real
   photos. (Bonus: use HoughLinesP to find the straight boundary lines.)

See `RUBRIC.md`.
