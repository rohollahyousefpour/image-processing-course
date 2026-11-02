"""
Week 13 - A small CNN that reads digits (the DocScan digit reader), runnable.
Trains on the 0-9 digit images, reports test accuracy, and compares to a
classical SVM baseline on the same data.

Usage:
    python digit_cnn.py

Requires: torch, scikit-learn. Runs on CPU in seconds; uses GPU if available.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        return self.fc(x.flatten(1))


def main():
    d = load_digits()
    X = d.images.astype(np.float32) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    print(f"train {len(Xtr)}  test {len(Xte)}  device {dev}")

    net = DigitCNN().to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"CNN parameters: {n_params}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr).unsqueeze(1).to(dev)
    ytr_t = torch.tensor(ytr).to(dev)
    Xte_t = torch.tensor(Xte).unsqueeze(1).to(dev)

    N, BS = len(Xtr_t), 128
    for epoch in range(60):
        net.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        if (epoch + 1) % 15 == 0:
            net.eval()
            with torch.no_grad():
                acc = (net(Xte_t).argmax(1).cpu().numpy() == yte).mean()
            print(f"epoch {epoch+1:3d}  loss {loss.item():.3f}  test acc {acc:.3f}")

    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).cpu().numpy()
    cnn_acc = (pred == yte).mean()
    print(f"\nCNN test accuracy: {cnn_acc:.3f}")
    print("Confusion matrix (CNN):")
    print(confusion_matrix(yte, pred))

    # classical baseline on the same digits
    svm = SVC(kernel="rbf", C=10).fit(Xtr.reshape(len(Xtr), -1), ytr)
    svm_acc = (svm.predict(Xte.reshape(len(Xte), -1)) == yte).mean()
    print(f"\nClassical SVM (raw pixels) test accuracy: {svm_acc:.3f}")
    print("Note: on easy 8x8 digits both are excellent. The CNN's advantage "
          "shows on harder data and via transfer learning (Week 14).")


if __name__ == "__main__":
    main()
