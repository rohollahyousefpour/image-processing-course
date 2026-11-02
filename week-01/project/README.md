# Week 1 Deliverable — Images & the Pipeline
# تحویلیِ هفتهٔ ۱ — تصویر و خطِ لوله

**Due:** before Week 2. **Weight:** part of the 25% weekly grade.

## Tasks
Work with **real images** from the course datasets (`E:/data/...`).

1. **Inspect.** Load 3 images (at least one water-meter numerator from
   `E:/data/abfa_image/numerators/`, one plate from `E:/data/plate/unique_maxsize/`, and one
   of your own). For each: print `shape`, `dtype`, `min/max/mean`. Convert to grayscale and HSV.
2. **Brighten correctly.** Write `brighten(img, k)` with no uint8 overflow; show a meter image
   `+60` done right vs the naive `img + k`, and explain the difference you see.
3. **Your DocScan test image.** Take a phone photo of a document *or* a meter. Save it as PNG.
   Report its properties and the hardest pipeline stages you anticipate.
4. **Short note.** For the meter crop, why is grayscale enough? When would you keep color?

Reference: `../code/`. Generate figures with the pattern in `assets/make_week01.py`.

## Deliverable
A short report (md/PDF) with the printed properties, before/after images, and your answers.

## Rubric — see `RUBRIC.md`.
