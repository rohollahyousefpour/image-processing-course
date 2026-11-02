"""
Week 12 - Features -> ML classifier (runnable).
Classify Iranian car models from HOG features with k-NN and SVM, honestly
evaluate on a held-out test set, print a confusion matrix, and demonstrate
overfitting.

Usage:
    python classify.py [class1 class2 class3 ...]

Defaults to 206 / 405 / Dena from E:/data/Vehicle_classes/train.
Requires: scikit-image, scikit-learn.
"""
import os, sys, cv2, numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DATA = "E:/data/Vehicle_classes/train"
PER_CLASS = 60
SIZE = (96, 96)


def hog_feat(path):
    im = cv2.imread(path)
    if im is None:
        return None
    g = cv2.cvtColor(cv2.resize(im, SIZE), cv2.COLOR_BGR2GRAY)
    return hog(g, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True)


def build(classes):
    X, y = [], []
    for label, c in enumerate(classes):
        d = os.path.join(DATA, c)
        for f in sorted(os.listdir(d))[:PER_CLASS]:
            v = hog_feat(os.path.join(d, f))
            if v is not None:
                X.append(v); y.append(label)
    return np.array(X), np.array(y)


def main(classes):
    X, y = build(classes)
    print(f"dataset: {X.shape}, per-class counts: {np.bincount(y)}")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    for name, clf in [("kNN(k=5)", KNeighborsClassifier(5)),
                      ("SVM(RBF,C=10)", SVC(kernel="rbf", C=10))]:
        clf.fit(Xtr, ytr)
        atr = accuracy_score(ytr, clf.predict(Xtr))
        ate = accuracy_score(yte, clf.predict(Xte))
        flag = "  <-- OVERFIT" if atr - ate > 0.3 else ""
        print(f"{name:16s} train {atr:.2f}  test {ate:.2f}{flag}")

    # detailed report + confusion matrix for the SVM
    svm = SVC(kernel="rbf", C=10).fit(Xtr, ytr)
    pred = svm.predict(Xte)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=classes, zero_division=0))

    # a cure attempt: lower C + cross-validation
    svm2 = SVC(kernel="rbf", C=1)
    cv = cross_val_score(svm2, X, y, cv=5)
    print(f"5-fold CV accuracy (C=1): {cv.mean():.2f} +/- {cv.std():.2f}")
    print("Lesson: hand features + SVM beat chance (0.33) but plateau on "
          "look-alike classes -> motivates CNNs (Week 13).")


if __name__ == "__main__":
    cls = sys.argv[1:] if len(sys.argv) > 1 else ["206", "405", "Dena"]
    main(cls)
