"""Frequency-domain filtering on a real image (Week 6)."""
import sys, cv2, numpy as np

def spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = 20*np.log(np.abs(F)+1)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), F

def disc(shape, r, low=True):
    rows, cols = shape; cy, cx = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    d = np.sqrt((Y-cy)**2 + (X-cx)**2)
    return ((d <= r) if low else (d > r)).astype(np.float32)

def apply(F, mask):
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(F*mask)))
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def main(path):
    g = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (256,256))
    spec, F = spectrum(g)
    cv2.imwrite("out_spectrum.png", spec)
    cv2.imwrite("out_lowpass.png", apply(F, disc(g.shape, 30, True)))
    cv2.imwrite("out_highpass.png", apply(F, disc(g.shape, 20, False)))
    print("wrote spectrum, lowpass, highpass")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/plate/unique_maxsize/11147L.jpg")
