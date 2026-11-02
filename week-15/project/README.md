# Week 15 Deliverable — Sequence Recognition & OCR
# تحویلیِ هفتهٔ ۱۵ — بازشناسیِ دنباله و OCR

**Due:** before the finale week. **Weight:** part of the 25% weekly grade + the DocScan
*read-the-whole-number* milestone.

## Tasks
1. **Train the CRNN+CTC reader.** Run `../code/crnn_ctc.py`. Report the final **sequence accuracy**
   and **CER**, and paste the training log (CTC loss ↓, CER ↓ per epoch).
2. **Show decoded readings.** Print 5 test strips beside their ground truth. Identify at least one
   error and classify it as an **insertion**, **deletion**, or **substitution** (use `metrics.py`).
3. **End-to-end vs. segment-then-classify.** Build a baseline that cuts each strip into fixed
   `8×8` windows and runs your Week 13 digit CNN on each; report its sequence accuracy on the same
   test strips and explain the gap (touching/rolled digits, error multiplication `a^L`).
4. **Length study.** Retrain (or evaluate) on strips of length 3 vs. 6 digits and report how CER
   changes with length. Relate the trend to the `a^L` argument from Session 1.
5. **Reflect.** How does this reader complete DocScan's **rectify → detect → read** pipeline
   (weeks 11, 14, 15)? What would real meter photos need that the synthetic strips do not
   (glare, tilt, colored decimal digit, variable digit width)?

## Bonus
- Replace greedy decoding with a small **beam search** and report any CER improvement.
- Add the **blank-collapse rule** to a unit test: feed hand-made paths (e.g. `4 4 - 2 2 - 6`) and
  assert the decoded string.
- Corrupt the strips (add noise / random shifts) and measure how CER degrades — an honest
  robustness curve.

## Deliverable
The sequence accuracy + CER + training log, the 5 decoded readings with one labeled error, the
segment-then-classify comparison, the length study, and your DocScan reflection.

---

## فارسی — خلاصهٔ تحویلی

۱. **آموزشِ خوانندهٔ CRNN+CTC:** فایلِ `../code/crnn_ctc.py` را اجرا کنید و دقتِ دنباله و CER نهایی را
   به‌همراهِ لاگِ آموزش گزارش دهید.
۲. **نمایشِ خواندن‌ها:** ۵ نوارِ آزمون را کنارِ برچسبِ واقعی چاپ کنید و دستِ‌کم یک خطا را به‌صورتِ
   درج/حذف/جایگزینی دسته‌بندی کنید (با کمکِ `metrics.py`).
۳. **سرتاسری در برابرِ برش‌سپس‌طبقه‌بندی:** یک خطِ پایه بسازید که نوار را به پنجره‌های `۸×۸` می‌برد و
   CNNِ هفتهٔ ۱۳ را روی هر کدام اجرا می‌کند؛ دقتِ دنباله‌اش را گزارش کنید و شکاف را با استدلالِ ضربِ
   خطا (`a^L`) و ارقامِ چسبیده/چرخان توضیح دهید.
۴. **مطالعهٔ طول:** روی نوارهای ۳ رقمی و ۶ رقمی ارزیابی کنید و ببینید CER چگونه با طول تغییر می‌کند.
۵. **تأمل:** این خواننده چگونه خطِ لولهٔ «صاف‌کردن ← تشخیص ← خواندن» را کامل می‌کند و عکس‌های واقعیِ
   کنتور چه چیزی لازم دارند که نوارهای مصنوعی ندارند؟

بخشِ امتیازی: جستجوی پرتو، آزمونِ واحد برای قاعدهٔ فروکاست، و منحنیِ مقاومت در برابرِ نویز.

See `RUBRIC.md`.
