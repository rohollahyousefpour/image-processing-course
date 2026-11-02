# Week 16 Deliverable — Transfer Learning & DocScan Capstone
# تحویلیِ هفتهٔ ۱۶ — یادگیریِ انتقالی و پروژهٔ نهاییِ DocScan

**Due:** end of course. **Weight:** the capstone (largest single component).

## Part A — Transfer learning (warm-up)
1. Use a pretrained net (`../code/transfer_learn.py`) as a **frozen** feature extractor on the
   same 2–3 classes you used in Week 12. Report the test accuracy.
2. Compare **directly** to your Week 12 HOG+SVM number. Explain what changed (only the features)
   and why deep features separate the classes better.
3. Show the confusion matrix and per-class precision/recall.

## Part B — The DocScan capstone (the whole course)
Assemble the end-to-end pipeline on real `E:/data` meter/plate images:
1. **Rectify** a tilted photo to a flat scan (Week 11 homography).
2. **Clean & segment** the digits (Weeks 2/7/8 threshold+morphology, Week 9 contours).
3. **Read** each digit (Week 13 CNN, or transfer-learned features).
4. **Robustness (≥2 required):** a confidence gate (reject low-inlier rectification), a
   digit-count sanity check, and/or a feature↔contour fallback.
5. **Evaluate:** report end-to-end accuracy on a handful of real images, show a failure case,
   and propose one concrete improvement.

## Deliverable
Part A numbers + confusion matrix; Part B: the running pipeline, a side-by-side of the 5 stages,
your robustness features, and the honest evaluation + failure analysis.

See `RUBRIC.md` and the top-level `capstone/README.md`.
