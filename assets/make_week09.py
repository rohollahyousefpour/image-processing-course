"""Week-9 figures: contours, bounding boxes, shape analysis on real images."""
import os, cv2, numpy as np
DATA="E:/data"; OUT="E:/programs/image-processing-course/week-09/assets"
os.makedirs(OUT, exist_ok=True)

def save(n, im): cv2.imwrite(f"{OUT}/{n}.png", im); print("wrote", n, im.shape)

g = cv2.cvtColor(cv2.imread(f"{DATA}/abfa_image/numerators/num_3494.jpg"), cv2.COLOR_BGR2GRAY)
g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
_, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
save("gray", g)
save("binary", b)

cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# keep reasonably sized contours (digits)
cnts = [c for c in cnts if cv2.contourArea(c) > 200]

# draw all contours
canvas = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas, cnts, -1, (0, 0, 255), 2)
save("contours", canvas)

# bounding boxes, sorted left-to-right
canvas2 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
boxes = sorted([cv2.boundingRect(c) for c in cnts], key=lambda r: r[0])
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(canvas2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(canvas2, str(i+1), (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
save("boxes", canvas2)

# convex hull of the largest contour
if cnts:
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    canvas3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas3, [big], -1, (0, 0, 255), 2)
    cv2.drawContours(canvas3, [hull], -1, (0, 255, 0), 2)
    save("hull", canvas3)

print("num digit contours:", len(cnts))
