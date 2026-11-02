# Week 2 Rubric — Histograms & Enhancement (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Histogram diagnosis | 20 | 5 images correctly classified from their histograms, with reasoning. |
| Enhancement comparison | 25 | stretch/equalize/gamma/CLAHE applied; best-for-digits chosen with justification. |
| Otsu vs adaptive | 25 | Both produced; correct foreground; the shadow case analyzed correctly. |
| DocScan milestone | 20 | 3 hard photos → readable binary of digits; pipeline documented. |
| Reasoning quality | 10 | Chooses "best for the goal" not "prettiest"; parameters justified. |

## Automatic deductions
- **−10** per-channel RGB equalization (color cast) instead of value-channel.
- **−8** global Otsu used on a clearly shadowed image where adaptive was needed.
- **−6** foreground/background convention wrong (digits black instead of white).

## Instructor note
Ask a student to predict, from a histogram alone, whether an image needs enhancement — then
reveal the image. Reading the histogram first is the habit this week builds.
