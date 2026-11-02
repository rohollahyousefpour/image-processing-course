"""Week-7 figures: morphology on a real binarized meter/plate."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-07/assets"
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)

g = cv2.cvtColor(cv2.imread(f"{DATA}/abfa_image/numerators/num_3494.jpg"), cv2.COLOR_BGR2GRAY)
g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
# binarize (digits white on black) so morphology operates on the digits
_, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# add small speckle noise to show opening removing it
noise = (np.random.rand(*b.shape) < 0.01).astype(np.uint8)*255
b_noisy = cv2.bitwise_or(b, noise)

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

def main():
    save("binary", b)
    save("noisy", b_noisy)
    save("erode", cv2.erode(b, k, iterations=1))
    save("dilate", cv2.dilate(b, k, iterations=1))
    save("open", cv2.morphologyEx(b_noisy, cv2.MORPH_OPEN, k))   # removes specks
    save("close", cv2.morphologyEx(b, cv2.MORPH_CLOSE, k))       # fills gaps
    grad = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k)            # outline
    save("gradient", grad)

if __name__=="__main__":
    main()
