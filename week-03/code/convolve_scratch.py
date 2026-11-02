"""A from-scratch convolution vs cv2.filter2D (Week 3)."""
import sys, cv2, numpy as np

def convolve3x3(img, k):
    pad = np.pad(img.astype(np.float32), 1, mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = np.sum(pad[y:y+3, x:x+3] * k)
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else r"E:/data/abfa_image/numerators/num_1332.jpg"
    g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # sharpen
    mine = convolve3x3(g, k)
    theirs = cv2.filter2D(g, -1, k)
    print("max abs difference vs cv2:", int(np.abs(mine.astype(int) - theirs.astype(int)).max()))
    cv2.imwrite("out_sharp_scratch.png", mine)
