# Week 7 Rubric — Morphology (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Four basic ops | 25 | erode/dilate/open/close shown; effects + SE-size role explained. |
| Speck removal (opening) | 20 | Opening cleans salt noise without shrinking digits. |
| Gap repair (closing) | 20 | Closing repairs broken digits into solid strokes. |
| Top-hat lighting fix | 20 | Uneven lighting corrected; thresholding improves after top-hat. |
| DocScan milestone | 15 | Clean binary; each digit a solid, separable blob. |

## Automatic deductions
- **−10** SE so large the digits become unrecognizable blobs.
- **−8** wrong order (closing before opening) fusing a speck into a digit.
- **−6** foreground/background convention wrong (operations act inverted).

## Instructor note
Ask: opening vs closing — which removes bright specks, which fills dark gaps? If a student can
map the defect (specks/gaps) to the operation, they've understood morphology.
