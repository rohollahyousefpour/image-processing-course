"""Week-13 figures: a Convolutional Neural Network that reads digits (DocScan!).
Trains a small CNN on 0-9 digit images, compares to a classical HOG+SVM baseline,
and visualizes the learned first-layer filters. Fully reproducible offline.
"""
import os, cv2, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

OUT = "E:/programs/image-processing-course/week-13/assets"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n)


# ---------- data ----------
d = load_digits()
X = d.images.astype(np.float32) / 16.0          # (N,8,8) in [0,1]
y = d.target.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# ---------- 1. sample digits grid ----------
grid = np.zeros((8 * 5, 8 * 10), np.uint8)
for k in range(10):
    idx = np.where(ytr == k)[0][:5]
    for r, i in enumerate(idx):
        grid[r * 8:(r + 1) * 8, k * 8:(k + 1) * 8] = (Xtr[i] * 255).astype(np.uint8)
save("digit_samples", cv2.resize(grid, (400, 200), interpolation=cv2.INTER_NEAREST))


# ---------- 2. the CNN ----------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))    # 8x8 -> 4x4
        x = self.pool(torch.relu(self.c2(x)))    # 4x4 -> 2x2
        return self.fc(x.flatten(1))


net = DigitCNN().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
Xtr_t = torch.tensor(Xtr).unsqueeze(1).to(dev)
ytr_t = torch.tensor(ytr).to(dev)
Xte_t = torch.tensor(Xte).unsqueeze(1).to(dev)

losses, accs = [], []
N, BS = len(Xtr_t), 128
for epoch in range(60):
    net.train()
    perm = torch.randperm(N, device=dev)
    ep_loss = 0.0
    for i in range(0, N, BS):
        idx = perm[i:i + BS]
        opt.zero_grad()
        out = net(Xtr_t[idx]); loss = lossf(out, ytr_t[idx])
        loss.backward(); opt.step()
        ep_loss += loss.item() * len(idx)
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).cpu().numpy()
    acc = (pred == yte).mean()
    losses.append(ep_loss / N); accs.append(acc)
cnn_acc = accs[-1]
print(f"CNN test accuracy: {cnn_acc:.3f}")

# ---------- 3. training curve (loss + acc) drawn with cv2 ----------
cur = np.full((260, 420, 3), 255, np.uint8)
for arr, col, sc in [(losses, (231, 76, 60), max(losses)), (accs, (46, 204, 113), 1.0)]:
    pts = [(20 + int(i / len(arr) * 380), 240 - int(v / sc * 210))
           for i, v in enumerate(arr)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(cur, a, b, col, 2)
cv2.putText(cur, "loss (red), test acc (green) vs epoch", (20, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
save("training_curve", cur)

# ---------- 4. learned first-layer filters ----------
w = net.c1.weight.detach().cpu().numpy()          # (8,1,3,3)
fg = np.zeros((3, 3 * 8 + 7), np.float32)
for k in range(8):
    f = w[k, 0]; f = (f - f.min()) / (np.ptp(f) + 1e-9)
    fg[:, k * 4:k * 4 + 3] = f
save("learned_filters", cv2.resize((fg * 255).astype(np.uint8), (400, 50),
     interpolation=cv2.INTER_NEAREST))

# ---------- 5. classical baseline on the SAME digits ----------
Xf_tr = Xtr.reshape(len(Xtr), -1); Xf_te = Xte.reshape(len(Xte), -1)
svm = SVC(kernel="rbf", C=10).fit(Xf_tr, ytr)
svm_acc = (svm.predict(Xf_te) == yte).mean()
print(f"classical SVM (raw pixels) test accuracy: {svm_acc:.3f}")

# ---------- 6. a few predictions ----------
vis = np.zeros((8, 8 * 8), np.uint8)
net.eval()
with torch.no_grad():
    idxs = np.arange(8)
    preds = net(Xte_t[:8]).argmax(1).cpu().numpy()
for r, i in enumerate(idxs):
    vis[:, r * 8:(r + 1) * 8] = (Xte[i] * 255).astype(np.uint8)
big = cv2.resize(vis, (8 * 40, 40), interpolation=cv2.INTER_NEAREST)
big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
for r in range(8):
    ok = preds[r] == yte[r]
    cv2.putText(big, str(preds[r]), (r * 40 + 12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 200, 0) if ok else (0, 0, 255), 2)
save("predictions", big)
print(f"SUMMARY  CNN {cnn_acc:.3f}  vs  SVM {svm_acc:.3f}  (digits are easy; both high)")
