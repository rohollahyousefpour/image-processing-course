# Week 13 Rubric — Convolutional Neural Networks (100 pts)

| Criterion | Pts | Full marks when… |
|---|---:|---|
| Train & report | 24 | CNN trained; final test accuracy + per-epoch training log shown. |
| Learned filters | 20 | First-layer filters visualized and sensibly described (edges/strokes). |
| Confusion matrix | 18 | Confused digit pairs identified with a plausible reason. |
| Classical comparison | 22 | SVM/k-NN baseline reported; honest account of when CNNs win. |
| DocScan reflection | 16 | Connects the reader to rectify→segment→read; notes what real digits need. |

## Automatic deductions
- **−12** reports accuracy on the training set (no held-out test).
- **−8** claims "CNNs are always better" — contradicted by the SVM's 0.99 here.
- **−6** describes conv layers without connecting them to convolution/kernels from Week 3.
- **−5** no training curve or per-epoch numbers (can't see under/over-fitting).

## Instructor note
Ask: "Your CNN got 97% and the SVM 99% — why bother with the CNN?" A strong student explains
that on this easy, clean, tiny dataset classical ML is already near-perfect, and the CNN's edge
appears on large, varied, high-resolution problems — and through transfer learning (Week 14).
The point of the week is *understanding the mechanism*, not beating the baseline on toy digits.
