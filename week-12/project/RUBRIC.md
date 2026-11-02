# Week 12 Rubric — Features → Machine Learning (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Dataset & features | 20 | HOG features extracted; `X`,`y` built; shape + per-class counts reported. |
| Correct evaluation | 24 | Stratified train/test split; BOTH train and test accuracy reported per model. |
| Confusion & metrics | 22 | Confusion matrix + per-class precision/recall/F1; confused classes identified. |
| Overfitting diagnosis | 22 | Names train≫test as overfitting; applies one cure; reports its effect on test. |
| Reflection | 12 | Explains the hand-feature ceiling and why it motivates CNNs. |

## Automatic deductions
- **−15** reports only training accuracy (or no split at all) — the score is meaningless.
- **−8** claims "high accuracy" from an imbalanced set without precision/recall.
- **−6** changes several things at once when "curing" overfitting (can't attribute the effect).
- **−5** data leakage: scaling/PCA fit on the whole set before the split.

## Instructor note
Ask: "Your SVM got 100% on train. Is that good?" A strong student immediately says no —
it's the signature of overfitting — and reaches for a test-set number and a confusion matrix
before trusting any model.
