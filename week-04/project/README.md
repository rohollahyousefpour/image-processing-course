# Week 4 Deliverable — Noise & Restoration
# تحویلیِ هفتهٔ ۴ — نویز و بازسازی

**Due:** before Week 5. **Weight:** part of the 25% weekly grade + a DocScan milestone.

## Tasks
1. **Controlled experiment.** Take a clean meter image; add Gaussian and salt-pepper noise
   (`../code/noise.py`). You now have ground truth.
2. **Compare + measure.** Denoise each with ≥3 filters; compute **PSNR vs the clean image**
   (`../code/denoise_compare.py`). Tabulate; identify the winner per noise type.
3. **Wrong-filter failure.** Show that Gaussian blur fails on salt-pepper (spreads the specks),
   and explain with the mean-vs-median argument.
4. **DocScan milestone.** Denoise a real noisy meter photo so its adaptive/Otsu binary (Week 2)
   is clean and the digits are readable. Report before/after binaries.

## Deliverable
A short report with the PSNR table, before/after images, and your reasoning.

See `RUBRIC.md`.
