# Week 13 Deliverable — Convolutional Neural Networks
# تحویلیِ هفتهٔ ۱۳ — شبکه‌های عصبیِ کانولوشنی

**Due:** before Week 14. **Weight:** part of the 25% weekly grade + the DocScan digit-reader milestone.

## Tasks
1. **Train the digit CNN.** Run `../code/digit_cnn.py`. Report the final test accuracy and paste
   the training log (loss ↓, accuracy ↑ per epoch).
2. **Visualize learned filters.** Extract and display the first-layer `3×3` filters. In 2–3
   sentences, describe what they appear to detect and relate them to Week 3 kernels.
3. **Read the confusion matrix.** Which digit pairs get confused (e.g. 8↔1)? Why might they look
   similar at `8×8`?
4. **Compare to classical.** Train an SVM (or k-NN) on the same digits; report its accuracy.
   Explain why both are high here and when you'd expect the CNN to pull ahead.
5. **Reflect.** How does this digit reader complete the DocScan pipeline (rectify → segment →
   **read**)? What would you change to read real meter digits instead of `8×8` toy digits?

## Bonus
- Add **dropout** or **data augmentation** and see if test accuracy improves.
- Change the learning rate by 10× up and down; describe what happens to the loss curve.
- Count the network's parameters and compare to a fully-connected net on the same input.

## Deliverable
The test accuracy + training log, the learned-filter image, the confusion matrix reading, the
classical comparison, and your DocScan reflection.

See `RUBRIC.md`.
