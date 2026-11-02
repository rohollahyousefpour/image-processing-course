"""Week-14 figures: transfer learning beats the Week-12 ceiling, + DocScan pipeline.
Uses a cached ImageNet ConvNeXt as a frozen feature extractor on the SAME
206/405/Dena car problem where HOG+SVM plateaued at 0.48.
"""
import os, cv2, numpy as np, torch, torch.nn as nn, torchvision
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DATA = "E:/data/Vehicle_classes/train"
OUT = "E:/programs/image-processing-course/week-14/assets"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["206", "405", "Dena"]
PER_CLASS = 60
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n)


# ---------- pretrained feature extractor (frozen) ----------
weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
model = convnext_base(weights=weights).to(dev).eval()
feat = nn.Sequential(model.features, model.avgpool, nn.Flatten()).to(dev).eval()
pre = weights.transforms()          # official ImageNet preprocessing


def embed(path):
    im = cv2.imread(path)
    if im is None:
        return None
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb).permute(2, 0, 1)          # HWC->CHW uint8
    t = pre(t).unsqueeze(0).to(dev)
    with torch.no_grad():
        return feat(t).cpu().numpy().ravel()


# ---------- build deep-feature dataset ----------
X, y = [], []
for label, c in enumerate(CLASSES):
    d = os.path.join(DATA, c)
    for f in sorted(os.listdir(d))[:PER_CLASS]:
        v = embed(os.path.join(d, f))
        if v is not None:
            X.append(v); y.append(label)
X, y = np.array(X), np.array(y)
print("deep features:", X.shape)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = SVC(kernel="rbf", C=10).fit(Xtr, ytr)
pred = clf.predict(Xte)
tl_acc = accuracy_score(yte, pred)
print(f"TRANSFER-LEARNING test accuracy: {tl_acc:.3f}  (Week-12 HOG+SVM was 0.48)")

# ---------- bar chart: HOG vs transfer ----------
bar = np.full((300, 400, 3), 255, np.uint8)
for i, (name, acc, col) in enumerate([("HOG+SVM", 0.48, (60, 76, 231)),
                                      ("Transfer", tl_acc, (46, 204, 113))]):
    x0 = 70 + i * 160
    h = int(acc * 220)
    cv2.rectangle(bar, (x0, 260 - h), (x0 + 90, 260), col, -1)
    cv2.putText(bar, f"{acc:.2f}", (x0 + 15, 255 - h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(bar, name, (x0, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.line(bar, (40, int(260 - 0.33 * 220)), (360, int(260 - 0.33 * 220)), (150, 150, 150), 1)
cv2.putText(bar, "chance 0.33", (250, int(258 - 0.33 * 220)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
save("accuracy_bars", bar)

# ---------- confusion matrix (transfer) ----------
cm = confusion_matrix(yte, pred); cell = 90
cimg = np.full((cell * 3 + 40, cell * 3 + 40, 3), 255, np.uint8)
mx = cm.max()
for i in range(3):
    for j in range(3):
        v = cm[i, j]; sh = int(255 - 180 * v / (mx + 1e-9))
        y0, x0 = 40 + i * cell, 40 + j * cell
        col = (sh, sh, 255) if i == j else (255, sh, sh)
        cv2.rectangle(cimg, (x0, y0), (x0 + cell, y0 + cell), col, -1)
        cv2.rectangle(cimg, (x0, y0), (x0 + cell, y0 + cell), (120, 120, 120), 1)
        cv2.putText(cimg, str(v), (x0 + 35, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
for k, c in enumerate(CLASSES):
    cv2.putText(cimg, c[:4], (45 + k * cell, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(cimg, c[:4], (2, 95 + k * cell), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
save("confusion_transfer", cimg)
print("done. transfer learning lifted the ceiling with the SAME data + a frozen ImageNet net.")
