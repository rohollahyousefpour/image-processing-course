"""
Week 16 - Transfer learning beats the Week-12 ceiling (runnable).
Uses a pretrained ImageNet ConvNeXt as a FROZEN feature extractor on the SAME
206/405/Dena car problem where HOG+SVM plateaued at 0.48.

Usage:
    python transfer_learn.py [class1 class2 class3 ...]

Requires: torch, torchvision (pretrained weights must be cached), scikit-learn.
Runs on GPU if available, else CPU.
"""
import os, sys, cv2, numpy as np, torch, torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA = "E:/data/Vehicle_classes/train"
PER_CLASS = 60
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def main(classes):
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights).to(dev).eval()
    feat = nn.Sequential(model.features, model.avgpool, nn.Flatten()).to(dev).eval()
    pre = weights.transforms()          # the model's OWN preprocessing

    def embed(path):
        im = cv2.imread(path)
        if im is None:
            return None
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        t = pre(torch.tensor(rgb).permute(2, 0, 1)).unsqueeze(0).to(dev)
        with torch.no_grad():
            return feat(t).cpu().numpy().ravel()

    X, y = [], []
    for label, c in enumerate(classes):
        d = os.path.join(DATA, c)
        for f in sorted(os.listdir(d))[:PER_CLASS]:
            v = embed(os.path.join(d, f))
            if v is not None:
                X.append(v); y.append(label)
    X, y = np.array(X), np.array(y)
    print(f"deep features: {X.shape}  (device {dev})")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    clf = SVC(kernel="rbf", C=10).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)

    print(f"\nTRANSFER-LEARNING test accuracy: {acc:.3f}")
    print("(Week 12 HOG+SVM on the SAME data was 0.48; chance is 0.33)")
    print("\nConfusion matrix:")
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=classes, zero_division=0))
    print("Only the FEATURES changed (HOG -> frozen ImageNet). Same SVM, same split.")


if __name__ == "__main__":
    cls = sys.argv[1:] if len(sys.argv) > 1 else ["206", "405", "Dena"]
    main(cls)
