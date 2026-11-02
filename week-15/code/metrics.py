"""
Week 15 - Sequence metrics for OCR: edit distance, CER, sequence accuracy.

Everything here is pure Python (no dependencies) so the ideas are transparent:
- edit_distance : Levenshtein distance via the O(len(a)*len(b)) DP recurrence
                  shown on the slides (insert / delete / substitute).
- cer           : character error rate = total edits / total reference chars.
- sequence_accuracy : fraction of predictions that match exactly.

Run directly for a small self-test:
    python metrics.py
"""
from typing import Sequence, List


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein edit distance between two sequences (strings or lists)."""
    # D[j] holds the distance for a[:i] vs b[:j]; we keep only one row.
    D = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, D[0] = D[0], i           # prev = D[i-1][j-1]
        for j, cb in enumerate(b, 1):
            cur = min(
                D[j] + 1,               # delete ca
                D[j - 1] + 1,           # insert cb
                prev + (0 if ca == cb else 1),  # match or substitute
            )
            prev, D[j] = D[j], cur
    return D[-1]


def cer(refs: List[Sequence], hyps: List[Sequence]) -> float:
    """Character error rate over a corpus: sum(edits) / sum(reference lengths)."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    chars = sum(len(r) for r in refs) or 1
    return edits / chars


def sequence_accuracy(refs: List[Sequence], hyps: List[Sequence]) -> float:
    """Fraction of predictions that exactly equal their reference."""
    if not refs:
        return 0.0
    correct = sum(1 for r, h in zip(refs, hyps) if list(r) == list(h))
    return correct / len(refs)


def _demo() -> None:
    cases = [
        ("018426", "018426", "exact match"),
        ("018426", "018326", "one substitution 4->3"),
        ("018426", "01826", "one deletion (dropped 4)"),
        ("018426", "0184426", "one insertion (extra 4)"),
        ("7013", "713", "dropped a 0"),
    ]
    print(f"{'reference':>10} {'prediction':>11} {'edits':>6} {'CER':>6}   note")
    for ref, hyp, note in cases:
        d = edit_distance(ref, hyp)
        print(f"{ref:>10} {hyp:>11} {d:>6} {d/len(ref):>6.3f}   {note}")

    refs = [c[0] for c in cases]
    hyps = [c[1] for c in cases]
    print(f"\ncorpus CER            : {cer(refs, hyps):.3f}")
    print(f"corpus sequence accuracy: {sequence_accuracy(refs, hyps):.3f}")


if __name__ == "__main__":
    _demo()
