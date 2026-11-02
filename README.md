# Practical Image Processing — Course Materials
## پردازش تصویرِ کاربردی

Undergraduate, hands-on. Slides in English (Beamer/metropolis), lecture notes in Persian
(XeLaTeX + xepersian). Language/tools: **Python + OpenCV + NumPy + scikit-image + Matplotlib**;
deep-learning weeks use **PyTorch** (Keras alternative noted).

Scope: **classical image processing (weeks 1–11) + an introduction to deep learning for
vision (weeks 12–16).**

> Guiding principle: *understand the pixels and the math, then reach for the library — and
> for a neural net only when classical methods run out.*

Capstone (built across the whole course): **DocScan — a smart document scanner & reader.**

---

## Repository layout

```
image-processing-course/
├── README.md                     ← this file (master plan)
├── common/
│   ├── beamer-preamble.tex       ← shared slide theme + code + plots (English, pdfLaTeX)
│   └── notes-preamble.tex        ← shared Persian lecture-notes setup (XeLaTeX)
├── capstone/README.md            ← the DocScan project spec
└── week-01/  … week-16/
    ├── slides/slides.tex         ← original single-lecture deck (kept for reference)
    ├── slides/slides-a.tex       ← Session 1 deck (the week meets twice) ~26–34 frames
    ├── slides/slides-b.tex       ← Session 2 deck ~26–34 frames
    ├── notes/notes.tex           ← detailed Persian lecture notes (≥12 pp)
    ├── code/                     ← runnable Python for the session
    └── project/                  ← weekly deliverable brief + rubric
```

## How to build

**Two sessions per week.** Each week meets twice, so its lecture is delivered as two decks:
`slides/slides-a.tex` (Session 1) and `slides/slides-b.tex` (Session 2). The original combined
`slides/slides.tex` is kept for reference.

Slides (English): `cd week-01/slides && pdflatex slides-a.tex` (twice), then the same for
`slides-b.tex`.
Persian notes (RTL): `cd week-01/notes && xelatex notes.tex` (twice).

Build all session decks at once (from the repo root):
```bash
for w in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  for s in a b; do
    ( cd week-$w/slides && pdflatex -interaction=nonstopmode slides-$s.tex && pdflatex -interaction=nonstopmode slides-$s.tex )
  done
done
```

## Python environment (for the code)
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy opencv-python scikit-image matplotlib
# deep-learning weeks:
pip install torch torchvision
```

---

## 16-week plan (two sessions per week)

| Wk | Topic (EN) | موضوع (FA) | Weekly deliverable |
|----|-----------|-----------|--------------------|
| 1 | Digital images & the pipeline | تصویرِ دیجیتال و خطِ لوله | Load/inspect/convert images |
| 2 | Point ops & histograms | عملیاتِ نقطه‌ای و هیستوگرام | Contrast/equalization tool |
| 3 | Spatial filtering & convolution | فیلترِ مکانی و کانولوشن | Blur/sharpen from scratch |
| 4 | Noise & restoration | نویز و بازسازی | Denoising comparison |
| 5 | Edge detection | تشخیصِ لبه | Canny edge pipeline |
| 6 | Frequency domain (Fourier) | حوزهٔ فرکانس (فوریه) | Periodic-noise removal |
| 7 | Morphological processing | پردازشِ مورفولوژیک | Clean up a binary mask |
| 8 | Thresholding & segmentation | آستانه‌گذاری و قطعه‌بندی | Otsu + watershed segmenter |
| 9 | Contours, components & shape | کانتور، مؤلفه‌ها و شکل | Count & measure objects |
| 10 | Feature detection & description | تشخیص و توصیفِ ویژگی | Keypoints + descriptors |
| 11 | Matching, homography, warping | تطبیق، هوموگرافی، تبدیل | Perspective correction (DocScan core) |
| 12 | From features to learning | از ویژگی تا یادگیری | Classical classifier on image features |
| 13 | CNNs for vision | شبکه‌های کانولوشنی | Train a small CNN (MNIST/EMNIST) |
| 14 | Object detection & localization | تشخیص و مکان‌یابیِ اشیا | Detect the document/digit region |
| 15 | Sequence recognition & OCR (CRNN/CTC) | تشخیصِ دنباله و OCR | Read the full meter numerator |
| 16 | Transfer learning + capstone | یادگیریِ انتقالی + پروژهٔ نهایی | DocScan end-to-end + defense |

## Assessment
- 40% — Capstone (DocScan pipeline, end-to-end)
- 25% — Weekly deliverables
- 20% — Understanding & defense (explain the math and the choices, not just call a library)
- 15% — Quality of analysis (comparisons, failure cases, parameter reasoning)

## Prerequisites
Programming (Python basics), basic linear algebra (vectors/matrices) and calculus
(derivatives). No prior image-processing or ML background assumed.
