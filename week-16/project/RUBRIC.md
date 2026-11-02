# Week 16 Rubric — Transfer Learning & DocScan Capstone (100 pts)

## Part A — Transfer learning (30 pts)
| Criterion | Pts | Full marks when… |
|---|---:|---|
| Frozen feature extraction | 12 | Pretrained net used correctly (its own preprocessing); features extracted. |
| Honest comparison | 12 | Test accuracy vs Week-12 HOG reported; explains that only features changed. |
| Metrics | 6 | Confusion matrix + per-class precision/recall shown. |

## Part B — DocScan capstone (70 pts)
| Criterion | Pts | Full marks when… |
|---|---:|---|
| Rectification | 14 | Tilted photo warped flat via homography (Week 11). |
| Clean & segment | 16 | Digits isolated with threshold/morphology + contours; ordered left-to-right. |
| Digit reading | 16 | Each digit classified (CNN or transfer); full reading produced. |
| Robustness (≥2) | 14 | At least two of: confidence gate, digit-count check, fallback path. |
| Evaluation & failure analysis | 10 | End-to-end accuracy on real images; a failure case + one concrete fix. |

## Automatic deductions
- **−10** transfer learning used the wrong preprocessing (not the model's `weights.transforms()`).
- **−10** capstone "works" on one image but has no robustness handling at all.
- **−8** reports only a best-case demo with no failure case or honest error rate.
- **−6** pipeline stages not attributable to the weeks that taught them (no traceability).

## Instructor note
The capstone is graded on **honesty and engineering**, not on hitting a magic accuracy. A student
who ships a modest reader that *gates its own low-confidence outputs* and clearly explains where it
fails has understood the course better than one who claims a perfect demo with no failure analysis.
