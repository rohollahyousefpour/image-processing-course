"""
Week 14 - IoU, NMS, and precision/recall from scratch (the two tools every
object detector needs). Pure Python + a tiny bit of NumPy; runs in a blink.

Usage:
    python iou_nms.py

Verifies the numbers we worked by hand in the lecture:
    * iou((1,1,5,4), (3,2,7,6)) == 0.167
    * NMS on {0.9, 0.8, 0.75} with tau=0.5 keeps boxes 0 and 2
    * a 5-detection precision/recall table
"""
import numpy as np


def iou(a, b):
    """Intersection over Union of two boxes given as (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)   # clamp at 0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, tau=0.5):
    """Greedy non-maximum suppression. Returns indices of surviving boxes."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)                                    # take the best
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < tau]
    return keep


def precision_recall(preds, gts, iou_thr=0.5):
    """
    preds : list of (box, score) sorted or unsorted.
    gts   : list of ground-truth boxes.
    Returns per-rank (label, precision, recall) after sorting by score.
    A prediction is a TP if IoU >= iou_thr with an unclaimed ground truth.
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    claimed = [False] * len(gts)
    tp = fp = 0
    rows = []
    for i in order:
        box = preds[i][0]
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j >= 0 and not claimed[best_j]:
            claimed[best_j] = True
            tp += 1
            label = "TP"
        else:
            fp += 1
            label = "FP"
        precision = tp / (tp + fp)
        recall = tp / len(gts)
        rows.append((label, precision, recall))
    return rows


def _demo():
    # 1) IoU matches the by-hand lecture example.
    a, b = (1, 1, 5, 4), (3, 2, 7, 6)
    print(f"iou({a}, {b}) = {iou(a, b):.3f}   (lecture: 0.167)")
    print(f"iou identical boxes = {iou(a, a):.3f}   (must be 1.000)")
    print(f"iou disjoint boxes  = {iou((0,0,2,2), (5,5,7,7)):.3f}   (must be 0.000)")

    # 2) NMS matches the by-hand lecture example.
    boxes = [(10, 10, 50, 40),      # B1  (score 0.90)  -> keep
             (12, 12, 52, 42),      # B2  (score 0.80)  -> suppressed (overlaps B1)
             (90, 10, 130, 40)]     # B3  (score 0.75)  -> keep (separate object)
    scores = [0.90, 0.80, 0.75]
    keep = nms(boxes, scores, tau=0.5)
    print(f"\nNMS(tau=0.5) keeps indices {keep}   (lecture: [0, 2])")
    print("  IoU(B1,B2) =", round(iou(boxes[0], boxes[1]), 2),
          " IoU(B1,B3) =", round(iou(boxes[0], boxes[2]), 2))

    # 3) precision / recall table (3 ground-truth objects).
    gts = [(0, 0, 20, 20), (40, 0, 60, 20), (80, 0, 100, 20)]
    preds = [((0, 0, 19, 19), 0.95),      # TP
             ((40, 0, 59, 19), 0.90),     # TP
             ((10, 40, 30, 60), 0.80),    # FP (no gt near here)
             ((80, 0, 100, 20), 0.70),    # TP
             ((45, 45, 65, 65), 0.60)]    # FP
    print("\nPrecision/recall as we accept lower-scored boxes:")
    print(" rank  result  precision  recall")
    for r, (label, p, rec) in enumerate(precision_recall(preds, gts), 1):
        print(f"  {r:>3}   {label:>4}     {p:5.2f}    {rec:5.2f}")


if __name__ == "__main__":
    _demo()
