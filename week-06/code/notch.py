"""Add periodic noise and remove it with a notch filter (Week 6)."""
import sys, cv2, numpy as np

def main(path):
    g = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (256,256))
    x = np.arange(256)
    g_noisy = np.clip(g + (20*np.sin(2*np.pi*x/8))[None,:], 0, 255).astype(np.uint8)
    F = np.fft.fftshift(np.fft.fft2(g_noisy))
    cy, cx = 128, 128
    mask = np.ones(F.shape, np.float32)
    Y, X = np.ogrid[:256, :256]
    for peak in [cx-32, cx+32]:
        mask[np.sqrt((Y-cy)**2 + (X-peak)**2) < 6] = 0
    clean = np.abs(np.fft.ifft2(np.fft.ifftshift(F*mask)))
    cv2.imwrite("out_periodic_noisy.png", g_noisy)
    cv2.imwrite("out_periodic_fixed.png",
                cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    print("wrote noisy + fixed")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:/data/plate/unique_maxsize/11147L.jpg")
