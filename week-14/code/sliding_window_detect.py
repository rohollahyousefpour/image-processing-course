"""
Week 14 - Sliding-window localization of the DocScan digit strip, end to end,
using only NumPy. No downloads, no GPU: builds a synthetic "meter" image with a
bright horizontal digit strip, sweeps a window, scores each crop, and runs NMS
to collapse the cluster into one clean box -- the crop Week 13's reader expects.

Usage:
    python sliding_window_detect.py

Pipeline (identical to the lecture's diagram):
    sweep windows -> score each -> NMS -> final box -> (compare to ground truth)
"""
import numpy as np
from iou_nms import iou, nms


def make_meter(H=120, W=200, seed=0):
    """A gray 'meter' photo with one bright, high-contrast horizontal strip."""
    rng = np.random.default_rng(seed)
    img = 40 + 20 * rng.standard_normal((H, W))          # noisy dark background
    # the digit strip: bright band with vertical 'digit' edges inside it
    y1, y2, x1, x2 = 55, 80, 60, 160
    img[y1:y2, x1:x2] = 210
    for x in range(x1 + 6, x2, 14):                      # dark digit separators
        img[y1:y2, x:x + 3] = 60
    img = np.clip(img, 0, 255)
    gt = (x1, y1, x2, y2)                                # ground-truth box
    return img, gt


def strip_score(crop):
    """
    'Does this crop look like a digit strip?' Two honest cues:
      * high mean brightness (the strip is bright), and
      * strong vertical edges (digits create vertical light/dark transitions).
    Returned as a 0..1-ish score.
    """
    if crop.size == 0:
        return 0.0
    brightness = crop.mean() / 255.0
    vert_edges = np.abs(np.diff(crop, axis=1)).mean() / 255.0
    return float(0.5 * brightness + 4.0 * vert_edges)


def slide(img, win, stride):
    """Yield (x, y, w, h) windows across the image."""
    H, W = img.shape
    w, h = win
    for y in range(0, H - h + 1, stride):
        for x in range(0, W - w + 1, stride):
            yield x, y, w, h


def detect(img, win=(104, 28), stride=8, score_thr=0.55, tau=0.4):
    boxes, scores = [], []
    for (x, y, w, h) in slide(img, win, stride):
        s = strip_score(img[y:y + h, x:x + w])
        if s > score_thr:
            boxes.append((x, y, x + w, y + h))
            scores.append(s)
    if not boxes:
        return [], [], []
    keep = nms(boxes, scores, tau=tau)
    return boxes, scores, keep


def main():
    img, gt = make_meter()
    boxes, scores, keep = detect(img)
    print(f"windows above threshold : {len(boxes)}")
    print(f"boxes after NMS         : {len(keep)}")

    if not keep:
        print("no strip found -- lower score_thr")
        return

    best = max(keep, key=lambda i: scores[i])
    pred = boxes[best]
    print(f"ground-truth strip box  : {gt}")
    print(f"predicted strip box     : {pred}")
    print(f"IoU(pred, gt)           : {iou(pred, gt):.3f}")
    x1, y1, x2, y2 = pred
    print(f"-> hand this {x2-x1}x{y2-y1} crop to Week 13's digit reader.")
    print("\nWhole pipeline ran on CPU with NumPy only: "
          "sweep -> score -> NMS -> one clean box.")


if __name__ == "__main__":
    main()
