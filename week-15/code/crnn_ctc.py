"""
Week 15 - Reading a whole string with a CRNN + CTC (the DocScan meter reader).

A small, honest, CPU-friendly demo of end-to-end sequence OCR:

  1. Build synthetic "meter strips" by concatenating the 8x8 sklearn digits
     into random 3-6 digit numbers (like a meter numerator).
  2. Define a CRNN: CNN (keeps the width axis) -> map-to-sequence -> BiLSTM
     -> per-column class scores over {0..9, blank}.
  3. Train it with nn.CTCLoss (sums over all alignments -- no per-digit cuts).
  4. Greedy-decode the test strips and report sequence accuracy and CER.

This is a teaching-scale model: it shows CER falling and the collapse rule
working in a few minutes on CPU, not a production meter reader.

Usage:
    python crnn_ctc.py

Requires: torch, scikit-learn, numpy. Uses GPU automatically if available.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

from metrics import cer, sequence_accuracy

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BLANK = 10          # class index of the CTC blank (10 digits are 0..9)
DIGIT_W = 8         # each digit image is 8x8
MIN_LEN, MAX_LEN = 3, 6
MAX_W = DIGIT_W * MAX_LEN     # all strips padded to this width (48)


# ---------------------------------------------------------------- data
def build_dataset(n_samples, images_by_class, rng):
    """Return (X, targets, target_lengths) of synthetic digit strips.

    X               : (n, 1, 8, MAX_W) float32, zero-padded on the right.
    targets         : flat 1-D LongTensor of all labels concatenated (CTC form).
    target_lengths  : length of each sample's label.
    """
    X = np.zeros((n_samples, 1, DIGIT_W, MAX_W), dtype=np.float32)
    all_targets, tgt_lens = [], []
    for k in range(n_samples):
        length = rng.integers(MIN_LEN, MAX_LEN + 1)
        digits = rng.integers(0, 10, size=length)
        strip = np.concatenate(
            [images_by_class[d][rng.integers(len(images_by_class[d]))]
             for d in digits], axis=1)               # 8 x (8*length)
        X[k, 0, :, : strip.shape[1]] = strip
        all_targets.extend(int(d) for d in digits)
        tgt_lens.append(length)
    return (torch.tensor(X),
            torch.tensor(all_targets, dtype=torch.long),
            torch.tensor(tgt_lens, dtype=torch.long))


# ---------------------------------------------------------------- model
class CRNN(nn.Module):
    def __init__(self, n_classes=11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 8x48 -> 4x24
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # -> 2x12
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),  # -> 1x12
        )
        self.rnn = nn.LSTM(32, 64, bidirectional=True)   # (T, B, 32) -> (T, B, 128)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):                 # x: (B, 1, 8, W)
        f = self.cnn(x)                   # (B, 32, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)  # map-to-sequence: (T, B, 32)
        y, _ = self.rnn(f)                # (T, B, 128)
        return self.fc(y).log_softmax(2)  # (T, B, n_classes)

    def seq_len(self, width):
        """How many CNN columns T for an input of the given pixel width."""
        return width // 4                 # two width-halving pools of size 2


# ---------------------------------------------------------------- decode
def greedy_decode(logp_tc, blank=BLANK):
    """logp_tc: (T, n_classes) log-probs for ONE sample -> list of digit ints."""
    path = logp_tc.argmax(1).tolist()
    out, prev = [], None
    for c in path:
        if c != prev and c != blank:      # merge repeats, then drop blanks
            out.append(c)
        prev = c
    return out


# ---------------------------------------------------------------- train / eval
def main():
    d = load_digits()
    imgs = d.images.astype(np.float32) / 16.0        # 1797 of 8x8 in [0,1]
    labels = d.target
    images_by_class = {c: imgs[labels == c] for c in range(10)}

    rng = np.random.default_rng(0)
    Xtr, ytr, ltr = build_dataset(4000, images_by_class, rng)
    Xte, yte, lte = build_dataset(800, images_by_class, rng)
    Xtr, Xte = Xtr.to(DEV), Xte.to(DEV)
    print(f"train {len(ltr)} strips  test {len(lte)} strips  device {DEV}")

    net = CRNN().to(DEV)
    print(f"CRNN parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    T = net.seq_len(MAX_W)                            # every strip padded to MAX_W
    in_lens = torch.full((0,), T)                     # placeholder, set per-batch
    N, BS, EPOCHS = len(ltr), 128, 25
    tgt_offsets = torch.cumsum(torch.cat([torch.tensor([0]), ltr]), 0)

    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(N)
        running = 0.0
        for i in range(0, N, BS):
            idx = perm[i:i + BS]
            xb = Xtr[idx]
            # gather this batch's flat targets and lengths
            tl = ltr[idx]
            tgt = torch.cat([ytr[tgt_offsets[j]:tgt_offsets[j + 1]] for j in idx])
            logp = net(xb)                            # (T, B, 11)
            il = torch.full((len(idx),), logp.size(0), dtype=torch.long)
            loss = ctc(logp, tgt.to(DEV), il, tl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0:
            acc, c = evaluate(net, Xte, yte, lte)
            print(f"epoch {epoch+1:3d}  ctc_loss {running/N:.3f}  "
                  f"seq_acc {acc:.3f}  CER {c:.3f}")

    acc, c = evaluate(net, Xte, yte, lte, show=8)
    print(f"\nFINAL  sequence accuracy {acc:.3f}   CER {c:.3f}")
    print("End-to-end CRNN+CTC: one net reads the whole meter string, no digit cutting.")


def evaluate(net, X, y_flat, lens, show=0):
    net.eval()
    refs, hyps = [], []
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), lens]), 0)
    with torch.no_grad():
        logp = net(X).cpu()                           # (T, N, 11)
    for k in range(len(lens)):
        ref = y_flat[offsets[k]:offsets[k + 1]].tolist()
        hyp = greedy_decode(logp[:, k, :])
        refs.append(ref)
        hyps.append(hyp)
    if show:
        print("\n  reference        prediction")
        for k in range(show):
            r = "".join(map(str, refs[k]))
            h = "".join(map(str, hyps[k]))
            mark = "ok " if r == h else "x  "
            print(f"  {mark} {r:<12} {h}")
    return sequence_accuracy(refs, hyps), cer(refs, hyps)


if __name__ == "__main__":
    main()
