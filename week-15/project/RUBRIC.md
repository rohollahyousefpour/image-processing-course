# Week 15 Rubric — Sequence Recognition & OCR (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Train & report | 22 | CRNN+CTC trained; final sequence accuracy **and** CER shown with a per-epoch log. |
| Decoded readings | 18 | 5 readings shown vs. truth; ≥1 error correctly labeled insertion/deletion/substitution. |
| End-to-end vs. segment | 24 | Segment-then-classify baseline built and beaten; gap explained via `a^L` + touching/rolled digits. |
| Length study | 16 | CER reported for short vs. long strips; trend tied back to the multiplication argument. |
| DocScan reflection | 20 | Connects reader to rectify→detect→read; names what real meter photos need. |

## Automatic deductions
- **−12** reports only per-**digit** accuracy (not sequence accuracy or CER) for a *string* reader.
- **−10** no CER at all — only exact-match accuracy, so near-misses are invisible.
- **−8** claims the CRNN "segments the digits internally" as a hard cut (it uses overlapping
  feature columns + CTC alignment, not explicit cuts).
- **−6** compares readings position-by-position instead of by edit distance (breaks on length
  changes).
- **−5** no training curve / per-epoch numbers (can't see the loss and CER fall).

## Instructor note
Ask: "Your Week 13 CNN reads a digit at 97% — why not just cut the strip and reuse it?" A strong
student answers with the `a^L` multiplication (97% per digit → ~83% at six digits) **and** the
physical reasons cuts fail (touching digits, odometer roll-over), then explains that the CRNN
never cuts: it keeps the width axis, emits overlapping feature columns, and lets **CTC** sum over
all alignments so the network learns to align itself. Probe the metrics too: "You got 99% CER but
94% sequence accuracy — which is which, and why report both?" The point of the week is the
*end-to-end sequence mindset* and *honest sequence-level evaluation*, closing the DocScan
rectify → detect → **read** pipeline.
