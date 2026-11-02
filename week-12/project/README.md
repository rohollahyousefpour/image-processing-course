# Week 12 Deliverable — Features → Machine Learning
# تحویلیِ هفتهٔ ۱۲ — از ویژگی تا یادگیریِ ماشین

**Due:** before Week 13. **Weight:** part of the 25% weekly grade.

## Tasks (use 2–3 classes from `E:/data`, e.g. car models, or plate vs non-plate)
1. **Build a dataset.** Extract HOG features for each image; assemble `X` (features) and
   `y` (labels). Report the shape and per-class counts (`../code/classify.py`).
2. **Train & evaluate honestly.** Stratified 70/30 train/test split. Train **k-NN and SVM**;
   report **both train and test accuracy** for each.
3. **Confusion matrix + per-class metrics.** Print the confusion matrix and precision/recall/F1
   per class. State which classes get confused and why you think so.
4. **Diagnose & cure overfitting.** If train ≫ test, name the problem and try **one** cure
   (more data, lower `C`, PCA, or add a colour histogram). Report whether test accuracy improved.
5. **Reflect.** In 3–4 sentences: what is the accuracy ceiling of hand-designed features here,
   and why does that motivate deep learning (Weeks 13–14)?

## Bonus
- Replace HOG with HOG **+ colour histogram** (feature concatenation) and compare.
- Run 5-fold cross-validation for a more stable accuracy estimate.

## Deliverable
The reported accuracies (train & test), the confusion matrix, your one cure + its effect, and
the reflection.

See `RUBRIC.md`.
