"""Week-2 figures: point operations & histograms, on a real meter image."""
import os
import cv2
import numpy as np

DATA = "E:/data"
OUT = "E:/programs/image-processing-course/week-02/assets"
os.makedirs(OUT, exist_ok=True)

gray = cv2.cvtColor(cv2.imread(f"{DATA}/abfa_image/numerators/num_3494.jpg"),
                    cv2.COLOR_BGR2GRAY)


def save(name, img):
    cv2.imwrite(f"{OUT}/{name}.png", img)
    print("wrote", name, img.shape)


def hist_image(g, color=(60, 60, 60), w=256, h=120):
    """Render a grayscale histogram as a small bar-chart image."""
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.max() + 1e-9) * (h - 8)
    canvas = np.full((h, w, 3), 245, np.uint8)
    for x in range(256):
        cv2.line(canvas, (x, h - 1), (x, h - 1 - int(hist[x])), color, 1)
    return canvas


def gamma(g, gm):
    lut = np.array([((i / 255.0) ** (1.0 / gm)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(g, lut)


def main():
    save("gray", gray)
    save("hist_before", hist_image(gray))

    # contrast stretch (min-max normalize)
    stretch = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    save("stretch", stretch)

    # histogram equalization
    eq = cv2.equalizeHist(gray)
    save("equalized", eq)
    save("hist_after", hist_image(eq))

    # gamma
    save("gamma_dark", gamma(gray, 0.5))   # darker
    save("gamma_bright", gamma(gray, 2.0))  # brighter

    # Otsu threshold (digits vs background)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save("otsu", otsu)


if __name__ == "__main__":
    main()
