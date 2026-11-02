# Week 6 Rubric — The Frequency Domain (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Spectra + interpretation | 25 | 3 spectra computed (with log + fftshift); each correctly interpreted. |
| Low/high-pass filtering | 25 | Masking done correctly; results match spatial equivalents; ringing noted. |
| Periodic noise removal | 30 | Peaks located; notch filter removes the stripes cleanly; before/after shown. |
| Convolution theorem | 10 | Speed/equivalence of frequency vs spatial explained. |
| DocScan reasoning | 10 | Frequency-vs-spatial choice justified for a concrete case. |

## Automatic deductions
- **−10** forgetting fftshift or the log (spectrum unreadable).
- **−8** an ideal hard cutoff used without noting ringing.
- **−6** notch discs so large that useful detail is lost.

## Instructor note
Ask a student to predict the spectrum of vertical stripes. If they say "two symmetric points on
the horizontal axis," they understand the spectrum as a map of waves.
