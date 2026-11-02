"""Week-12 figures: from image features to a machine-learning classifier.
Real task: classify Iranian car models (206 vs 405 vs Dena) from HOG features.
"""
import os, cv2, numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

DATA = "E:/data/Vehicle_classes/train"
OUT = "E:/programs/image-processing-course/week-12/assets"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["206", "405", "Dena"]
PER_CLASS = 60          # keep it small & fast
SIZE = (96, 96)
rng = np.random.RandomState(0)


def save(n, im):
    cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n)


def load_gray(path):
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.cvtColor(cv2.resize(im, SIZE), cv2.COLOR_BGR2GRAY)
    return im


def hog_feat(g):
    return hog(g, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True)


# --- 1. HOG visualization on one real car ---
sample_dir = os.path.join(DATA, "206")
sample = os.path.join(sample_dir, sorted(os.listdir(sample_dir))[0])
g = load_gray(sample)
_, hog_img = hog(g, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=True)
hog_img = (255 * (hog_img / (hog_img.max() + 1e-9))).astype(np.uint8)
save("sample_gray", g)
save("hog_vis", cv2.applyColorMap(hog_img, cv2.COLORMAP_INFERNO))

# --- 2. build the dataset (HOG features + labels) ---
X, y = [], []
for label, c in enumerate(CLASSES):
    d = os.path.join(DATA, c)
    files = sorted(os.listdir(d))[:PER_CLASS]
    for f in files:
        g = load_gray(os.path.join(d, f))
        if g is not None:
            X.append(hog_feat(g)); y.append(label)
X, y = np.array(X), np.array(y)
print("dataset:", X.shape, "labels:", np.bincount(y))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                      random_state=0, stratify=y)

# --- 3. train two classifiers ---
knn = KNeighborsClassifier(5).fit(Xtr, ytr)
svm = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
for name, clf in [("kNN", knn), ("SVM", svm)]:
    acc_tr = accuracy_score(ytr, clf.predict(Xtr))
    acc_te = accuracy_score(yte, clf.predict(Xte))
    print(f"{name}: train acc {acc_tr:.2f}  test acc {acc_te:.2f}")

# --- 4. confusion matrix (SVM) rendered with cv2 ---
cm = confusion_matrix(yte, svm.predict(Xte))
cell = 90
img = np.full((cell * 3 + 40, cell * 3 + 40, 3), 255, np.uint8)
mx = cm.max()
for i in range(3):
    for j in range(3):
        v = cm[i, j]
        shade = int(255 - 180 * v / (mx + 1e-9))
        y0, x0 = 40 + i * cell, 40 + j * cell
        col = (shade, shade, 255) if i == j else (255, shade, shade)
        cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), col, -1)
        cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (120, 120, 120), 1)
        cv2.putText(img, str(v), (x0 + 35, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
for k, c in enumerate(CLASSES):
    cv2.putText(img, c[:4], (45 + k * cell, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, c[:4], (2, 95 + k * cell),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
save("confusion", img)

# --- 5. 2D feature space via PCA (shows separability) ---
p2 = PCA(2).fit_transform(X)
canvas = np.full((400, 500, 3), 255, np.uint8)
cols = [(60, 76, 231), (113, 204, 46), (219, 152, 52)]  # BGR
p2n = p2 - p2.min(0)
p2n = (p2n / p2n.max(0) * [440, 340] + [30, 30]).astype(int)
for (px, py), lab in zip(p2n, y):
    cv2.circle(canvas, (px, py), 5, cols[lab], -1)
for k, c in enumerate(CLASSES):
    cv2.circle(canvas, (40, 30 + k * 20), 5, cols[k], -1)
    cv2.putText(canvas, c, (52, 35 + k * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
save("feature_space", canvas)
print("done.")
