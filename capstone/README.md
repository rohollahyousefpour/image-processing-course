# Capstone — DocScan: a smart document scanner & reader
# پروژهٔ نهایی — DocScan: اسکنر و خوانندهٔ هوشمندِ سند

The single project built across the course. It takes a **photo of a document** (a page, form,
or receipt, shot at an angle in uneven light) and turns it into a **clean, deskewed, machine-
readable scan** — then reads some content with a small neural network.

Chosen because it exercises almost every technique in the syllabus, and ends with a genuinely
useful artifact.

## The pipeline (and where each week contributes)

```
photo ─► grayscale/denoise ─► detect page ─► perspective-correct ─► binarize ─► clean ─► segment ─► read
        (W1–4)               (W5,9)          (W10–11)               (W2,8)     (W7)    (W9)      (W12–14)
```

| Stage | Weeks | Techniques |
|-------|-------|-----------|
| Load & prepare | 1–4 | color spaces, histogram, denoising, restoration |
| Find the page | 5, 9 | edges (Canny), contours, the largest quadrilateral |
| Flatten it | 10–11 | corner features, homography, perspective warp |
| Make it crisp | 2, 6, 8 | contrast, periodic-noise removal, adaptive/Otsu threshold |
| Clean up | 7 | morphology (remove specks, close gaps) |
| Segment content | 9 | connected components → text lines / fields / digits |
| Read it | 12–14 | features + classifier, then a CNN (MNIST/EMNIST) for digits/chars |

## Deliverable (Week 14)
- A runnable `docscan` pipeline: input photo → output clean scan (+ extracted digits/text).
- A short report: the pipeline, the parameters you chose and **why**, and **failure cases**.
- A defense: explain the math behind two stages, and where a classical step beat (or lost to)
  the neural one.

## Milestones (checked at these weeks)
- **W5**: detect and draw the page boundary on a test photo.
- **W8**: produce a clean binary image of the page.
- **W11**: perspective-correct the page to a flat rectangle.
- **W14**: read the digits/characters and assemble the end-to-end tool.

## Data
Use your own phone photos of documents + a public set (e.g. scanned forms) + MNIST/EMNIST for
the reading stage. Keep a handful of *hard* images (blur, shadow, skew) to test robustness.
For this course we used the real `E:/data` meter (`abfa_image/numerators`), plate
(`plate/unique_maxsize`), and vehicle (`Vehicle_classes`) sets throughout.

## Reference code you can build on
Every stage already exists as a small, runnable script from the weekly lessons:

| Stage | Script | Produces |
|-------|--------|----------|
| Rectify (perspective) | `week-11/code/rectify.py` | manual + auto (ORB+RANSAC) flat scan |
| Clean & segment digits | `week-09/code/contours.py` | ordered per-digit crops |
| Read digits (CNN) | `week-13/code/digit_cnn.py` | trained 0–9 classifier (~0.97) |
| Read via transfer learning | `week-14/code/transfer_learn.py` | frozen ImageNet features + SVM |

Wire these together, add the robustness features (confidence gate, digit-count check, fallback),
and you have the end-to-end DocScan tool.

## Verified results from the reference pipeline
- **Rectification** on a real meter: 66 ORB matches → 57 RANSAC inliers → reprojection RMSE 1.17 px.
- **Digit reading** (small CNN, 1898 params) on 0–9: **0.97** test accuracy.
- **Transfer learning** on look-alike car models: **0.48 → 0.87** vs the classical HOG+SVM baseline.
