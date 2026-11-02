"""Week-14: the full DocScan pipeline as one strip, on a real meter digit-strip."""
import os, cv2, numpy as np
OUT = "E:/programs/image-processing-course/week-14/assets"
os.makedirs(OUT, exist_ok=True)


def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n)


img = cv2.imread("E:/data/abfa_image/numerators/num_1332.jpg")
img = cv2.resize(img, (360, 160))
h, w = img.shape[:2]

# stage 1: angled photo
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst = np.float32([[40, 25], [w - 15, 5], [w - 5, h - 10], [25, h - 20]])
photo = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                            borderValue=(30, 30, 30))
# stage 2: rectify back
rect = cv2.warpPerspective(photo, cv2.getPerspectiveTransform(dst, src), (w, h))
# stage 3: binarize
g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
g = cv2.GaussianBlur(g, (3, 3), 0)
bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY_INV, 21, 8)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
bw3 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
# stage 4: segment digit boxes
cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
seg = rect.copy()
boxes = []
for c in cnts:
    x, yy, ww, hh = cv2.boundingRect(c)
    if hh > 30 and 8 < ww < 90 and ww * hh > 300:
        boxes.append((x, yy, ww, hh))
for (x, yy, ww, hh) in sorted(boxes):
    cv2.rectangle(seg, (x, yy), (x + ww, yy + hh), (0, 0, 255), 2)

# assemble a labelled vertical strip of the four stages
stages = [("1. angled photo", photo), ("2. rectified (W11)", rect),
          ("3. binarized (W7-8)", bw3), ("4. digits found (W9)", seg)]
pad = 30
strip = np.full((len(stages) * (h + pad) + 10, w + 20, 3), 255, np.uint8)
for i, (label, im) in enumerate(stages):
    y0 = i * (h + pad) + pad
    strip[y0:y0 + h, 10:10 + w] = im
    cv2.putText(strip, label, (10, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)
save("docscan_pipeline", strip)
print(f"digit boxes found: {len(boxes)}")
