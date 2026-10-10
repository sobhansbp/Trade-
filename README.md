# aitrader: ربات معاملاتی هوش مصنوعی برای EURUSD و طلا (XAUUSD)

ترکیب یک **لایه راهبردی روزانه** (آزموده روی ۲۰+ سال داده)، یک **میز ۱۲ تحلیلگر ساعتی** که تقریباً همه روش‌های تحلیل نمودار را پوشش می‌دهد، **یادگیری ماشین walk-forward**، **مدیریت ریسک سخت‌گیرانه** و یک **میز تحلیلگر Claude** با مناظره خریدار/فروشنده.

> ⚠️ این نرم‌افزار ابزار تحلیل و پژوهش است، نه توصیه مالی. سود ۱۰٪ ماهانه با هیچ تنظیمی پایدار نیست؛ جدول مونت‌کارلو در داشبورد هزینه هر هدف سود را نشان می‌دهد. لبه آماری اثبات‌شده کوچک است (جزئیات در [docs/RESEARCH.md](docs/RESEARCH.md)). پیش از هر استفاده واقعی، حداقل چند ماه روی حساب دمو اجرا کنید.

## نصب

```bash
pip install -r requirements.txt
```

## استفاده

```bash
# تحلیل لحظه‌ای هر دو نماد + داشبورد HTML در reports/dashboard.html
python -m aitrader analyze EURUSD XAUUSD

# هوش مصنوعی: تحلیل اخبار + مناظره خریدار/فروشنده با Groq (یا Claude)
export GROQ_API_KEY=gsk_...            # یا ANTHROPIC_API_KEY=sk-ant-...
python -m aitrader analyze EURUSD XAUUSD --ai
# انتخاب ریسک هر معامله (پیش‌فرض 0.005 = ۰٫۵٪؛ بیش از ۱٪ توصیه نمی‌شود)
python -m aitrader analyze EURUSD XAUUSD --risk 0.01

# بک‌تست کامل خارج از نمونه + پرتفوی ترکیبی
python -m aitrader backtest EURUSD XAUUSD

# اجرای زنده: بعد از بسته شدن هر کندل ساعتی تحلیل، ژورنال، معامله کاغذی و اعلان تلگرام
export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
python -m aitrader live EURUSD XAUUSD --ai
# ارسال سفارش واقعی به MetaTrader 5 (فقط ویندوز، pip install MetaTrader5)
python -m aitrader live EURUSD XAUUSD --execute

# استفاده از داده بروکر خودتان (MT4/MT5/TradingView، تایم‌فریم H1 یا کمتر)
python -m aitrader analyze XAUUSD --csv XAUUSD=path/to/XAUUSD_H1.csv
```

کلید API را فقط به‌صورت متغیر محیطی بدهید و هرگز در کد یا گیت قرار ندهید. ارائه‌دهنده با `AITRADER_LLM=groq|claude` و مدل Groq با `AITRADER_GROQ_MODEL` (پیش‌فرض `openai/gpt-oss-120b`) قابل تغییر است.

اگر نام نماد در بروکر شما متفاوت است (مثلاً `XAUUSDm`)، متغیر `MT5_SYMBOL_XAUUSD=XAUUSDm` را تنظیم کنید. مدل Claude با `AITRADER_MODEL` قابل تغییر است (پیش‌فرض `claude-opus-5-5`).

## اجرای زنده روی حساب دمو

### روش ۱: خودکار با GitHub Actions (بدون نیاز به کامپیوتر روشن)
فایل `.github/workflows/paper-trading.yml` هر ساعت (یکشنبه تا جمعه، دقیقه ۷ UTC) یک دور کامل اجرا می‌کند: تحلیل، اخبار با Groq هر ۳ ساعت، میز هوش مصنوعی فقط وقتی سیگنال معامله هست، و به‌روزرسانی دو حساب کاغذی (تاکتیکی و سوئینگ، هر کدام ۱۰ هزار دلار). نتیجه در پوشه `journal/` کامیت می‌شود:
- `journal/STATUS.md` وضعیت حساب، پوزیشن‌ها، آخرین تحلیل و معاملات بسته‌شده
- `journal/paper_trades.csv` و `journal/paper_equity.csv` تاریخچه
- `journal/news_scores.csv` امتیاز اخبار برای سنجش رو به جلو
- `journal/forward_tests.csv` آزمون رو به جلو فرضیه‌های دور سوم (فقط ثبت، بدون معامله؛ `aitrader/live/forward.py`)

راه‌اندازی: در گیت‌هاب به Settings → Secrets and variables → Actions بروید و `GROQ_API_KEY` (و در صورت تمایل `TELEGRAM_BOT_TOKEN` و `TELEGRAM_CHAT_ID`) را اضافه کنید. برای اجرای دستی از تب Actions → paper-trading → Run workflow استفاده کنید.

### داشبورد زنده و ربات تلگرام روی Cloudflare
یک Cloudflare Worker (`cloudflare/aitrader-desk`) ژورنال ربات را از همین مخزن می‌خواند و سه کار انجام می‌دهد. خود معامله و یادگیری ماشین همچنان روی GitHub Actions اجرا می‌شود، چون Workers پایتون و LightGBM را اجرا نمی‌کند.
- **داشبورد زنده:** https://aitrader-desk.sobhan13851121.workers.dev (موجودی دو دفتر، تصمیم آخر هر نماد، سوگیری روزانه، پلن، منحنی سرمایه، آزمون رو به جلو و اخبار). داده خام JSON در `/api/status` است.
- **ربات تلگرام (فقط برای صاحب حساب):** `/status`، `/eurusd`، `/gold`، `/positions`، `/forward`، `/news`، `/dashboard`. هر پیام دیگر به Groq فرستاده می‌شود و با داده لحظه‌ای ربات جواب می‌گیرد (سقف ۲۰ سؤال در ساعت).
- **نگهبان:** هر ساعت (دقیقه ۳۷) بررسی می‌کند. اگر ربات در ساعات کاری بیش از ۱۵۰ دقیقه ژورنال را به‌روز نکرده باشد، در تلگرام هشدار می‌دهد و پس از برگشت هم خبر می‌دهد.

استقرار دوباره:
```bash
cd cloudflare/aitrader-desk
npx wrangler@4.139.0 deploy
npx wrangler@4.139.0 secret put TELEGRAM_BOT_TOKEN   # و TELEGRAM_CHAT_ID، GROQ_API_KEY، TG_WEBHOOK_SECRET
```
وب‌هوک تلگرام با `setWebhook` و پارامتر `secret_token` (همان TG_WEBHOOK_SECRET) روی آدرس `/tg` تنظیم می‌شود. برای عوض کردن کلید Groq از همان دستور `secret put` یا داشبورد Cloudflare استفاده کنید (Workers → aitrader-desk → Settings → Variables and Secrets).

### روش ۲: متاتریدر ۵ روی ویندوز (سفارش واقعی روی حساب دمو)
`scripts/run_live_windows.bat` را باز کنید، کلیدها را وارد کنید، در MT5 وارد یک **حساب دمو** شوید و فایل را اجرا کنید. با `--execute` سفارش‌ها با حد ضرر و تارگت به MT5 ارسال می‌شوند.

## قانون معامله

> **نتیجه اعتبارسنجی ۱۶ ساله (دور سوم):** بخش ساعتی (دفتر تاکتیکی) در ۲۰۱۴ تا ۲۰۲۶ پس از هزینه سود نداد (ضریب سود ۰٫۹۵ تا ۱٫۰۰). تنها لایه با شواهد مثبت بلندمدت **دفتر سوئینگ روزانه** است (۲۰۰۴ تا ۲۰۲۶، شارپ حدود ۰٫۶۷). دفتر تاکتیکی برای جمع‌کردن شواهد رو به جلو در حساب کاغذی فعال مانده، ولی برای پول واقعی توصیه نمی‌شود. جزئیات: `docs/RESEARCH.md` بخش ۷.

1. **سوگیری روزانه** (میانگین وزن برابر اجزا، بین ‎-1 و ‎+1):
   - EURUSD: مومنتوم ۲۰/۶۰/۱۲۰/۲۵۰ روزه، EMA20/100، روند بازده ۲ساله آمریکا (معکوس)، روند ۲۰روزه DXY (معکوس)
   - طلا: مومنتوم، EMA50/200، روند بازده واقعی ۱۰ساله (معکوس)، تمایل ساختاری خرید
2. فقط وقتی **|سوگیری| ≥ ۰٫۵** است معامله می‌شود و فقط **در جهت آن**.
3. ورود وقتی که **برآیند میز ۱۲ تحلیلگر ساعتی** هم‌جهت شود (> ۰٫۰۵). ساعت‌های ۲۱ تا ۲۴ UTC و ۳۰ دقیقه قبل تا ۱ ساعت بعد از اخبار مهم ممنوع است.
4. **حد ضرر** پشت نزدیک‌ترین ابطال ساختاری (سوئینگ، کف/سقف روز قبل) بین ۰٫۸ تا ۳ ATR. **۵۰٪ در +1R** و انتقال حد ضرر به نقطه سر به سر، باقی تا **+2.5R** یا ۷۲ ساعت.
5. **ریسک ۰٫۵٪** سرمایه در هر معامله. نصف شدن ریسک در افت ۶٪ و توقف در افت ۱۲٪. معامله دوم هم‌جهت با دلار (مثلاً خرید یورو + خرید طلا) با نصف ریسک.
6. **تحلیلگر اخبار (هوش مصنوعی):** اخبار مخالف قوی معامله را متوقف می‌کند و «ریسک رویداد» حجم را نصف می‌کند؛ هرگز معامله باز نمی‌کند.
7. **حساب ترکیبی:** استراتژی ساعتی + سوئینگ روزانه با ریسک برابر (همبستگی ۰٫۲۴، شارپ ترکیبی خارج از نمونه حدود ۱٫۳).
8. **میز هوش مصنوعی** فقط می‌تواند وتو کند (تبدیل به «صبر»). باز کردن معامله، برگرداندن جهت یا تغییر ریسک در کد ممنوع است.

## میز ۱۲ تحلیلگر ساعتی

| تحلیلگر | روش‌ها |
|---|---|
| روند چندزمانی | EMA 20/50/200، Supertrend، ایچیموکو، Parabolic SAR، ADX/DMI، ساختار H4 و D1 |
| مومنتوم | RSI، MACD، استوکاستیک، ROC، مومنتوم سری زمانی ۵ و ۲۰ روزه |
| بازگشت به میانگین | z-score بولینگر، RSI(3)، فاصله از VWAP، فیلتر Hurst |
| اسمارت‌مانی | BOS، CHoCH، اوردر بلاک، FVG، premium/discount، OTE فیبوناچی |
| نقدینگی | sweep سوئینگ، equal highs/lows، PDH/PDL، PWH/PWL، رنج آسیا، Wyckoff spring/upthrust، واگرایی SMT |
| الگوها | کندل‌ها در محل مناسب، واگرایی معمولی/مخفی RSI، سقف/کف دوقلو، سر و شانه، هارمونیک (گارتلی، بت، پروانه، خرچنگ، سایفر)، قواعد موج الیوت، مثلث/کنج |
| شکست | فشردگی بولینگر-کلتنر، دانچیان، شکست رنج آسیا در لندن، کندل displacement |
| سطوح | S/R خوشه‌ای، پروفایل حجم (طلا) / TPO (یورو) روز قبل، پیوت‌های کلاسیک |
| کلان | بازده واقعی، DXY، بازده ۲ساله، نقره، VIX، S&P |
| فصلی | t-stat خودآموز ساعت روز و روز هفته |
| موقعیت‌گیری | شاخص COT سفته‌بازان (مخالف در حدهای افراطی) |
| یادگیری ماشین | LightGBM با برچسب سه‌مانعی و اعتبارسنجی purged walk-forward |

## ساختار کد

```
aitrader/
  config.py            مشخصات نمادها و پارامترها
  data/loader.py       Yahoo، FRED، CFTC COT، تقویم اقتصادی، CSV بروکر
  indicators/          core (اندیکاتورها)، structure (SMC/سطوح/پروفایل)، patterns
  features.py          ماتریس ویژگی علّی + چندزمانی + بین‌بازاری
  strategic.py         لایه راهبردی روزانه
  analysts/ensemble.py ۱۲ تحلیلگر + وزن‌دهی رژیمی
  ml/models.py         LightGBM walk-forward + متا-برچسب
  risk/manager.py      حد ضرر، شبیه‌ساز معامله، اندازه پوزیشن
  backtest/engine.py   بک‌تست رویدادمحور، آمار، پرتفوی
  ai/llm.py            لایه مشترک Groq / Claude
  ai/claude_desk.py    مناظره خریدار/فروشنده/مدیر معامله + نرده‌های ریسک
  news/                جمع‌آوری اخبار و تحلیلگر اخبار
  portfolio.py         حساب ترکیبی، جدول ماهانه، مونت‌کارلو
  risk/montecarlo.py   رابطه ریسک و سود و احتمال نابودی
  report/              نمودار PNG/Plotly و داشبورد فارسی
  live/runner.py       حلقه زنده، ژورنال، معامله کاغذی، تلگرام، MT5
  live/forward.py      آزمون رو به جلو از پیش ثبت‌شده برای فرضیه‌هایی که هنوز اثبات نشده‌اند
  data/histdata.py     داده یک‌دقیقه‌ای رایگان ۲۰۱۰ به بعد (HistData، تبدیل ساعت به UTC)
cloudflare/aitrader-desk/  Worker: داشبورد زنده، دستورهای تلگرام، نگهبان (فقط خواندن ژورنال)
research/             آزمون‌های دور سوم: ۶۰ قاعده روی ۱۶ سال، Chronos، اعتبارسنجی بلندمدت، چارت چندزمانه
tests/                 تست نبود نگاه به آینده، شبیه‌ساز، نرده‌ها
```

## پژوهش دور سوم (بازتولید)

```bash
python -m aitrader.data.histdata EURUSD 2010 2026 && python -m aitrader.data.histdata XAUUSD 2010 2026
PYTHONPATH=. python3 research/prep_data.py          # M5/H1 از داده یک‌دقیقه‌ای
PYTHONPATH=. python3 research/round3_tests.py       # ۶۰ آزمون + کنترل آزمون چندگانه
PYTHONPATH=. python3 research/round3_charts.py      # نمودارها و خروجی فشرده برای داشبورد
PYTHONPATH=. python3 research/long_history_backtest.py EURUSD   # و XAUUSD، سپس combine
PYTHONPATH=. python3 research/foundation_models.py  # Chronos-Bolt و Chronos-2 (نیاز به torch و chronos-forecasting)
```

خلاصه نتایج در `docs/RESEARCH.md` بخش ۷ و بخش «آزمایشگاه پژوهش» داشبورد است.

## تست

```bash
python -m pytest -q
```
