"""Week-4 figures: noise models & restoration, on real images."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-04/assets"
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)

g = cv2.cvtColor(cv2.imread(f"{DATA}/plate/unique_maxsize/11147L.jpg"), cv2.COLOR_BGR2GRAY)
g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

def add_gaussian(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)

def add_sp(img, p=0.06):
    out = img.copy()
    out[np.random.rand(*img.shape) < p/2] = 255
    out[np.random.rand(*img.shape) < p/2] = 0
    return out

def main():
    save("clean", g)
    gn = add_gaussian(g); save("gaussian_noise", gn)
    sp = add_sp(g);       save("sp_noise", sp)
    # restorations of gaussian noise
    save("gauss_denoise_gauss", cv2.GaussianBlur(gn, (5,5), 0))
    save("gauss_denoise_bilateral", cv2.bilateralFilter(gn, 9, 60, 60))
    save("gauss_denoise_nlm", cv2.fastNlMeansDenoising(gn, None, 15, 7, 21))
    # restorations of salt-pepper
    save("sp_denoise_gauss", cv2.GaussianBlur(sp, (5,5), 0))   # smears, bad
    save("sp_denoise_median", cv2.medianBlur(sp, 3))            # great

if __name__=="__main__":
    main()
