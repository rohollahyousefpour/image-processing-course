# Week 4 Rubric — Noise & Restoration (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Controlled noise experiment | 20 | Known Gaussian + salt-pepper added; ground truth kept. |
| PSNR comparison | 25 | ≥3 filters measured vs clean; table correct; winners identified. |
| Wrong-filter failure | 20 | Gaussian-on-salt-pepper failure shown + explained (mean vs median). |
| DocScan milestone | 25 | Real noisy photo denoised → clean readable binary of digits. |
| Reasoning | 10 | Filter matched to noise; trade-off (noise vs detail) discussed. |

## Automatic deductions
- **−10** eyeballing quality with no PSNR/SSIM (or downstream) measurement.
- **−8** median used where averaging was needed, or vice versa, without justification.
- **−6** over-denoising that erases the digits.

## Instructor note
Ask: mean vs median of {100,100,100,255}? If the student explains why the median (100) ignores
the outlier, they understand why median crushes salt-and-pepper noise.
