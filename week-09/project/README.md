# Week 9 Deliverable — Contours & Shape
# تحویلیِ هفتهٔ ۹ — کانتور و شکل

**Due:** before Week 10. **Weight:** part of the 25% weekly grade + a DocScan milestone.

## Tasks (on a cleaned meter binary from Week 7/8)
1. **Find & measure.** Find contours; report each digit's area and bounding box; draw them.
2. **Filter.** Drop noise contours with an area + aspect-ratio rule; verify the digit count
   matches expectation (`../code/contours.py`).
3. **Crop & order.** Output one cropped image per digit, sorted left-to-right.
4. **Bonus.** Approximate a document's outer contour with `approxPolyDP` to its 4 corners
   (feeds Week 11's perspective correction).

## Deliverable
The measured table, the filtered/annotated image, and the ordered digit crops.

See `RUBRIC.md`.
