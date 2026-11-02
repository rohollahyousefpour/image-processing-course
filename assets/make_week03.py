"""Week-3 figures: spatial filtering & convolution, on real images."""
import os, cv2, numpy as np
DATA = "E:/data"; OUT = "E:/programs/image-processing-course/week-03/assets"
os.makedirs(OUT, exist_ok=True)

# Use a plate (has fine detail: good for blur/sharpen visibility) + the meter.
plate = cv2.cvtColor(cv2.imread(f"{DATA}/plate/unique_maxsize/11147L.jpg"), cv2.COLOR_BGR2GRAY)
plate = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)  # upscale for print
meter = cv2.cvtColor(cv2.imread(f"{DATA}/abfa_image/numerators/num_1332.jpg"), cv2.COLOR_BGR2GRAY)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

def main():
    save("plate_gray", plate)
    save("plate_box", cv2.blur(plate, (5, 5)))                    # box/mean blur
    save("plate_gauss", cv2.GaussianBlur(plate, (7, 7), 1.5))     # gaussian blur
    # unsharp mask sharpening: img + (img - blur)
    blur = cv2.GaussianBlur(plate, (0, 0), 2.0)
    sharp = cv2.addWeighted(plate, 1.5, blur, -0.5, 0)
    save("plate_sharp", sharp)

    # add salt-pepper noise, then median denoise (preview of week 4)
    noisy = plate.copy()
    n = (np.random.rand(*plate.shape) < 0.05); noisy[n] = 255
    n2 = (np.random.rand(*plate.shape) < 0.05); noisy[n2] = 0
    save("plate_noisy", noisy)
    save("plate_median", cv2.medianBlur(noisy, 3))

    save("meter_gray", meter)
    save("meter_gauss", cv2.GaussianBlur(meter, (5, 5), 1.0))

if __name__ == "__main__":
    np.random.seed(0)
    main()
