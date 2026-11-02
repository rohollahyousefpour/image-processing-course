# Week 14 Deliverable — Object Detection & Localization
# تحویلیِ هفتهٔ ۱۴ — تشخیص و مکان‌یابیِ اشیا

**Due:** before the capstone finale. **Weight:** part of the 25% weekly grade + the DocScan
"find the strip, then read it" milestone.

## Tasks
1. **Implement `iou` and `nms`.** Complete (or re-derive) `../code/iou_nms.py`. Verify
   `iou((1,1,5,4),(3,2,7,6)) == 0.167`, identical boxes `== 1.0`, disjoint boxes `== 0.0`.
   Show that NMS on scores `{0.9, 0.8, 0.75}` with `tau=0.5` keeps boxes `0` and `2`.
2. **Localize the digit strip.** Run `../code/sliding_window_detect.py`. Report how many windows
   pass the score threshold, how many survive NMS, and the **IoU** of your predicted strip box
   against the ground truth. Explain what raising/lowering `stride`, `score_thr`, and `tau` does.
3. **Read the strip (reuse Week 13).** Take the localized crop and feed it to your Week 13 digit
   reader (or describe precisely how you would). Report the read value and confirm the crop is the
   size the reader expects.
4. **Score a detector.** Using `precision_recall`, compute precision & recall for a small set of
   detections vs. ground truth and **sketch the PR curve**. State the AP intuition (area under it).
5. **Reflect.** How does "find the strip, then read it" complete the DocScan pipeline
   (rectify → **detect** → segment → read)? What breaks on a real phone photo (glare, angle,
   multiple meters) and which knob (confidence vs NMS threshold) would you turn?

## Bonus
- Add a second object class (e.g. a round **dial**) to `make_meter` and run **per-class** NMS.
  Show that cross-class NMS wrongly suppresses one object.
- Add an **image pyramid** to `sliding_window_detect.py` so a resized strip is still found.
- Compute the sliding-window count for a 640×480 image at stride 8, 5 scales, and compare it to
  the single forward pass of a YOLO-style grid (`S×S×(B·5+C)`).

## Deliverable
Your `iou`/`nms` verification output, the localization result (window counts + IoU), the read
value from the reused reader, the precision/recall numbers + PR sketch, and your DocScan reflection.

---

## شرحِ فارسی

**موعد:** پیش از فینالِ پروژهٔ نهایی. **وزن:** بخشی از نمرهٔ هفتگیِ ۲۵٪ + نقطهٔ‌عطفِ
«اول پیدا کن، بعد بخوان» در DocScan.

### کارها
۱. **`iou` و `nms` را پیاده‌سازی کنید.** فایلِ `../code/iou_nms.py` را کامل (یا بازاستخراج) کنید.
   بررسی کنید که `iou((1,1,5,4),(3,2,7,6))` برابرِ `0.167`، جعبه‌های یکسان `1.0` و جعبه‌های جدا
   `0.0` می‌شود. نشان دهید NMS روی امتیازهای `{0.9, 0.8, 0.75}` با `tau=0.5` جعبه‌های `0` و `2`
   را نگه می‌دارد.
۲. **نوارِ ارقام را مکان‌یابی کنید.** `../code/sliding_window_detect.py` را اجرا کنید. گزارش دهید
   چند پنجره از آستانهٔ امتیاز عبور کرد، چند تا پس از NMS ماند، و **IoU** جعبهٔ پیش‌بینی‌شده نسبت به
   واقعیت چقدر است. اثرِ تغییرِ `stride`، `score_thr` و `tau` را توضیح دهید.
۳. **نوار را بخوانید (بازاستفاده از هفتهٔ ۱۳).** برشِ مکان‌یابی‌شده را به خوانندهٔ ارقامِ هفتهٔ ۱۳
   بدهید (یا دقیقاً شرح دهید چگونه این کار را می‌کنید). مقدارِ خوانده‌شده را گزارش کنید و مطمئن شوید
   اندازهٔ برش همانی است که خواننده انتظار دارد.
۴. **آشکارساز را ارزیابی کنید.** با `precision_recall`، precision و recall را برای مجموعه‌ای کوچک از
   تشخیص‌ها نسبت به واقعیت حساب کنید و **منحنیِ PR** را رسم کنید. شهودِ AP (مساحتِ زیرِ منحنی) را بیان کنید.
۵. **بازتاب.** چگونه «اول پیدا کن، بعد بخوان» خطِ‌لولهٔ DocScan را (صاف‌سازی ← **تشخیص** ← قطعه‌بندی ←
   خواندن) کامل می‌کند؟ روی عکسِ واقعیِ گوشی چه چیزی خراب می‌شود (انعکاسِ نور، زاویه، چند کنتور) و
   کدام دکمه (آستانهٔ اطمینان در برابرِ آستانهٔ NMS) را می‌چرخانید؟

### امتیازِ اضافی
- یک کلاسِ دومِ شیء (مثلاً یک **عقربهٔ** گرد) به `make_meter` اضافه کنید و NMS را **per-class** اجرا کنید.
  نشان دهید NMSِ بین‌کلاسی به‌اشتباه یکی از اشیا را سرکوب می‌کند.
- یک **هرمِ تصویر** به `sliding_window_detect.py` بیفزایید تا نوارِ تغییراندازه‌داده‌شده هم پیدا شود.
- تعدادِ پنجره‌های لغزان را برای تصویرِ ۶۴۰×۴۸۰ با گامِ ۸ و ۵ مقیاس حساب کنید و آن را با \emph{یک}
  گذرِ رو‌به‌جلوی شبکهٔ YOLO مقایسه کنید.

### تحویلی
خروجیِ بررسیِ `iou`/`nms`، نتیجهٔ مکان‌یابی (تعدادِ پنجره‌ها + IoU)، مقدارِ خوانده‌شده از خوانندهٔ
بازاستفاده‌شده، اعدادِ precision/recall + طرحِ منحنیِ PR، و بازتابِ DocScan.

See `RUBRIC.md`.
