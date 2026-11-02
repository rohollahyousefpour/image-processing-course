# Week 8 Deliverable — Segmentation
# تحویلیِ هفتهٔ ۸ — قطعه‌بندی

**Due:** before Week 9. **Weight:** part of the 25% weekly grade + a DocScan milestone.

## Tasks (real images from `E:/data`)
1. **Threshold comparison.** Otsu vs adaptive on several meter images; pick the winner per
   lighting condition and explain.
2. **k-means colour.** Segment a car/plate photo at k = 2, 4, 8 (`../code/kmeans_color.py`);
   discuss over- vs under-segmentation and the role of k.
3. **Watershed.** Separate touching meter digits with distance transform + marker-controlled
   watershed (`../code/watershed_digits.py`); label each with connected components.
4. **DocScan milestone.** Output one cropped image per digit, ordered left-to-right.

See `RUBRIC.md`.
