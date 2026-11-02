# Week 14 Rubric — Object Detection & Localization (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| IoU & NMS correctness | 24 | `iou` verified (0.167 / 1.0 / 0.0) and NMS keeps the right boxes; clamp + per-class noted. |
| Digit-strip localization | 22 | Sliding-window demo run; window counts + IoU-to-ground-truth reported; knobs explained. |
| Read the strip (Week 13 tie-in) | 16 | Localized crop fed to the digit reader (or precise plan); read value + crop size shown. |
| Detector evaluation | 22 | Precision/recall computed correctly; PR curve sketched; AP/mAP intuition stated. |
| DocScan reflection | 16 | Connects find→read to rectify→detect→segment→read; names a real-photo failure + the right knob. |

## Automatic deductions
- **−12** judges the detector with plain classification accuracy instead of IoU-aware mAP.
- **−8** runs NMS across classes (a meter suppresses a nearby dial) or omits NMS entirely.
- **−6** IoU forgets to clamp overlap width/height at 0 (phantom negative area).
- **−5** confuses the two thresholds — confidence (pre-NMS) vs NMS IoU (`tau`).
- **−4** reports box quality without any IoU number.

## Instructor note
Ask: "Your detector reports the same meter three times — is that an IoU problem or a confidence
problem?" A strong student separates the two knobs: too many *duplicates* of one object → lower the
**NMS** threshold; too many *spurious* boxes → raise the **confidence** threshold. Then push on the
metric: "Why not just report accuracy?" — because a box can carry the right label yet sit in the
wrong place, so detection must be scored with IoU-aware precision/recall and mAP. The week's point
is the reusable machinery (IoU + NMS) and honest evaluation, not beating a number on a toy image —
and that machinery is exactly what lets DocScan *find* the digit strip before Week 13 *reads* it.
