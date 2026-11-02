"""Week-6 figures: Fourier / frequency domain, on real images."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-06/assets"
os.makedirs(OUT, exist_ok=True)

g = cv2.cvtColor(cv2.imread(f"{DATA}/plate/unique_maxsize/11147L.jpg"), cv2.COLOR_BGR2GRAY)
g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_CUBIC)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

def spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = 20*np.log(np.abs(F)+1)         # log for dynamic range
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), F

def ideal_mask(shape, r, low=True):
    rows, cols = shape; cy, cx = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    d = np.sqrt((Y-cy)**2 + (X-cx)**2)
    m = (d <= r) if low else (d > r)
    return m.astype(np.float32)

def apply_filter(F, mask):
    G = F * mask
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def main():
    save("gray", g)
    spec, F = spectrum(g)
    save("spectrum", spec)
    save("lowpass", apply_filter(F, ideal_mask(g.shape, 30, low=True)))   # blur
    save("highpass", apply_filter(F, ideal_mask(g.shape, 20, low=False))) # edges

    # periodic noise demo + its removal in frequency
    rows, cols = g.shape
    x = np.arange(cols)
    pattern = (20*np.sin(2*np.pi*x/8)).astype(np.float32)   # vertical stripes
    noisy = np.clip(g.astype(np.float32)+pattern[None,:], 0, 255).astype(np.uint8)
    save("periodic_noisy", noisy)
    specn, Fn = spectrum(noisy); save("periodic_spectrum", specn)
    # notch: kill the two bright peaks off-center (approx) via a band mask
    mask = np.ones(g.shape, np.float32); cy, cx = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    for peak in [cx-32, cx+32]:
        d = np.sqrt((Y-cy)**2 + (X-peak)**2); mask[d < 6] = 0
    save("periodic_fixed", apply_filter(Fn, mask))

if __name__=="__main__":
    main()
