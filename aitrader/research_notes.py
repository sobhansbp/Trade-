"""Key research findings shown on the dashboard (details in docs/RESEARCH.md)."""

FINDINGS = [
    ("سیگنال‌های ساعتی به‌تنهایی لبه ندارند",
     "روی H1 تقریباً همه ۱۲ تحلیلگر ضریب اطلاعات نزدیک صفر داشتند (|IC|<0.05). نسخه ۱ که فقط برآیند میز ساعتی را معامله "
     "می‌کرد در هر دو نماد خارج از نمونه زیان داد؛ این با پژوهش‌هایی که می‌گویند قواعد نموداری درون‌روزی پس از هزینه لبه را از دست می‌دهند سازگار است."),
    ("روند میان‌مدت و نرخ بهره، ستون اصلی",
     "روی داده روزانه ۲۰۰۵ تا ۲۰۲۳، مومنتوم ۱۲۰روزه و EMA20/100 برای یورو در هر سه زیردوره مثبت بودند و سیگنال اختلاف نرخ بهره "
     "(روند بازده ۲ساله آمریکا) در دوره ۲۰۲۴ به بعد هم مثبت ماند. ترکیب با وزن برابر در حالت سوئینگ: شارپ ۰٫۳۷ برای یورو و ۰٫۵۵ برای طلا (۲۰۰۵ تا ۲۰۲۳)."),
    ("طلا: روند + بازده واقعی + تمایل ساختاری خرید",
     "مومنتوم و EMA50/200 روی طلا کار کردند. رابطه معکوس با بازده واقعی تا ۲۰۲۳ برقرار بود ولی در ۲۰۲۴ تا ۲۰۲۶ با خرید بانک‌های مرکزی شکست."),
    ("بازگشت به میانگین کوتاه‌مدت زیان‌ده است",
     "RSI(2) و z-score کوتاه‌مدت روی هر دو نماد در دوره توسعه شارپ منفی داشتند، پس فقط به‌عنوان ابزار زمان‌بندی در میز ساعتی باقی ماندند."),
    ("یک نشت داده پیدا و حذف شد",
     "قیمت بسته‌شدن روزانه EURUSD در یاهو در واقع قیمت ساعت ۰۰:۰۰ UTC همان روز است؛ برای همین سیگنال DXY حدود ۲۱ ساعت از آینده را می‌دید "
     "(شارپ کاذب ۰٫۹۶ تا ۱٫۰۷). با داده رسمی FRED و یک روز تأخیر اضافه برای همه داده‌های بین‌بازاری، این مزیت ناپدید شد."),
    ("طراحی نهایی پیش از دیدن Holdout قفل شد",
     "قانون نهایی: فقط در جهت سوگیری روزانه قوی (|bias|≥۰٫۵) و فقط وقتی میز ساعتی تأیید کند. این قانون بین ۱۳ گزینه روی دوره توسعه "
     "و بر پایه منطق از پیش تعیین‌شده انتخاب شد، سپس یک بار روی ۲۵٪ پایانی داده‌ها آزمون شد."),
    ("نتیجه صادقانه Holdout و عدم قطعیت",
     "یورو در Holdout (فوریه تا اکتبر ۲۰۲۶) مثبت بود در حالی که خود جفت‌ارز حدود ۷٪ افت کرد؛ طلا در اصلاح ۱۷ درصدی از سقف حدود ۶٪ زیان داد. "
     "چون یاهو فقط ۷۳۰ روز داده ساعتی می‌دهد و پنجره هر روز جلو می‌رود، نتیجه پرتفوی ساعتی بین اجراها بین حدود ۷+٪ تا ۱۵+٪ در ۲۰ ماه "
     "(شارپ ۰٫۵ تا ۰٫۹) جابه‌جا شد. این دامنه، عدم قطعیت واقعی تخمین است. پس از اولین نگاه به Holdout دو تغییر داده شد، پس کاملاً بکر نیست."),
    ("یادگیری ماشین: لبه ضعیف و فیلتر متا خاموش",
     "AUC خارج از نمونه مدل جهت حدود ۰٫۵۲ تا ۰٫۵۵ بود و فیلتر متا-برچسب‌گذاری نتیجه را بهتر نکرد، پس به‌صورت پیش‌فرض خاموش است "
     "ولی خروجی ML به‌عنوان یکی از ۱۲ تحلیلگر نمایش داده می‌شود."),
    ("ترکیب دو استراتژی، اهرم واقعی بهبود",
     "همبستگی استراتژی ساعتی و سوئینگ روزانه فقط ۰٫۲۴ است. ترکیب با ریسک برابر در دوره خارج از نمونه شارپ را به حدود ۱٫۳ رساند و "
     "افت حداکثر را به حدود ۵٪ کم کرد. شارپ بلندمدت سوئینگ ۰٫۶۳ است، پس انتظار واقع‌بینانه حدود ۰٫۹ است."),
    ("نمادهای بیشتر کمکی نکرد",
     "همان دستور روزانه روی ۷ جفت‌ارز دیگر، نقره، نفت، مس، اوراق و شاخص‌ها آزمون شد (بدون تنظیم). جفت‌ارزها همه تابع دلارند و "
     "بقیه بازارها به‌تنهایی شارپ ۰٫۱ تا ۰٫۴ داشتند، پس سبد بزرگ‌تر شارپ را از ۰٫۶۸ به حدود ۰٫۴ پایین آورد؛ فقط افت ۲۰۲۴ تا ۲۰۲۶ را کم کرد."),
    ("هوش مصنوعی Groq برای اخبار، نه برای خواندن چارت",
     "مدل gpt-oss-120b روی Groq اخبار را درست تفسیر کرد، ولی مدل بینایی در خواندن تصویر چارت قیمت و حد ضرر را اشتباه گفت. "
     "برای همین هوش مصنوعی فقط اخبار را امتیاز می‌دهد، می‌تواند معامله را وتو یا حجم را نصف کند و هر امتیاز برای سنجش رو به جلو ثبت می‌شود."),
    ("۱۰٪ ماهانه از نظر ریاضی در دسترس نیست",
     "سقف رشد مرکب با اهرم بهینه حدود SR²/2 در سال است: با شارپ ۰٫۹ حدود ۳٫۴٪ در ماه. برای ۱۰٪ ماهانه شارپ پایدار ۱٫۵ لازم است. "
     "در مونت‌کارلو، میانه ماهانه در هیچ اهرمی به ۱۰٪ نرسید و بالاتر از نوسان ۸۰٪ سالانه هم سود کم شد و هم احتمال نصف شدن حساب بالا رفت."),
]

# ---------------------------------------------------------------- round 3 (research lab section)

FINDINGS += [
    ("۶۰ قاعده معروف روی ۱۶ سال: هیچ‌کدام قبول نشد",
     "قواعد مقالات (الگوی W فیکس‌ها، بریدن-رانالدو، FOMC)، استراتژی‌های محبوب (شکست لندن، ORB، ICT جوداس، ویلیامز) و اثرهای "
     "تقویمی با پارامترهای منبع خودشان روی داده یک‌دقیقه‌ای ۲۰۱۰ تا ۲۰۲۶ آزمون شدند. پس از هزینه خرد و کنترل آزمون چندگانه هیچ‌کدام "
     "معنادار نماند و رتبه درون‌نمونه، برون‌نمونه را پیش‌بینی نکرد (همبستگی رتبه‌ای ۰٫۲۱، p=0.11)."),
    ("الگوی فیکس‌ها پس از انتشار محو شد",
     "پنجره W3 (۰۸:۰۰ لندن تا فیکس ECB) در ۲۰۱۰ تا ۲۰۱۸ روی یورو روزانه ۱٫۹+ واحد پایه خالص با t=3.0 داشت و در ۲۰۱۹ تا ۲۰۲۶ "
     "به ۰٫۶- رسید؛ همان چیزی که خود مقاله گفته بود: سود فقط با اسپرد بین‌بانکی ممکن است."),
    ("سود طلا در ۲۰۱۹ تا ۲۰۲۶ شبانه بود",
     "تقریباً تمام بازده طلا بین ۱۳:۳۰ تا ۰۸:۲۰ نیویورک به دست آمد و جلسه روزانه کومکس منفی بود. شکست لندن طلا هم فقط در سمت "
     "فروش سود داد. هیچ‌کدام معنادار نیست، پس فقط به‌عنوان فرضیه رو به جلو و بدون معامله ثبت می‌شوند."),
    ("یک لبه ظاهری، ساخته داده بود",
     "خلاف گپ آخر هفته یورو در بک‌تست ۴٫۳+ واحد پایه خالص در برون‌نمونه داشت، ولی فقط با ورود دقیقاً در قیمت بازگشایی یکشنبه. "
     "با ۱۵ دقیقه تأخیر به ۰٫۱- رسید، چون این سود همان اسپرد باز لحظه بازگشایی بود."),
]

LAB_SURVEY = [
    ("حساب‌های خرد: اکثریت زیان می‌دهند",
     "طبق ESMA، ۷۴ تا ۸۹٪ حساب‌های CFD خرد زیان می‌دهند. بنا بر میانگین صنعت به نقل از FPFX، فقط حدود ۷٪ شرکت‌کنندگان "
     "چالش پراپ‌فرم به برداشت سود می‌رسند. FTMO در ۱۰ سال بیش از ۴۵۰ میلیون دلار پرداخت کرده، ولی درآمد اصلی پراپ‌ها از هزینه چالش است.",
     [("ESMA", "https://www.esma.europa.eu/node/84933"),
      ("Tradeinformer / FPFX", "https://tradeinformer.com/prop-weekly/how-does-ftmo-make-money"),
      ("Finance Magnates", "https://www.financemagnates.com/forex/ftmo-announces-over-450-million-paid-out-as-prop-trading-firm-turns-10/")]),
    ("اکسپرت‌ها، کانال‌های سیگنال و کپی‌ترید",
     "بیشتر اکسپرت‌های طلا در بازار MQL5 شبکه‌ای یا مارتینگل‌اند و بازده تبلیغ‌شده یا بک‌تست است یا سابقه زنده کوتاه دارد. "
     "کانال‌های تلگرامی نرخ برد ۷۰ تا ۹۲٪ ادعا می‌کنند، ولی بک‌تست مستقل سیگنال‌ها نرخ برد ۲۶ تا ۳۴٪ نشان داد. "
     "فهرست «بهترین تریدرها» در کپی‌ترید هم دچار سوگیری بقاست.",
     [("Myfxbook data (MQL5)", "https://www.mql5.com/en/blogs/post/772467"),
      ("Telegram signal reviews", "https://www.telegramsignalsreviews.com/alltelegramsignalsreviews"),
      ("Martingale risk (MQL5)", "https://www.mql5.com/en/forum/502291/page8")]),
    ("ربات‌های هوش مصنوعی با پول واقعی",
     "در فصل اول Alpha Arena (پول واقعی) فقط Qwen (حدود ۲۲+٪) و DeepSeek (حدود ۵+٪) سود کردند. Claude، Gemini، Grok و GPT-5 "
     "بین ۴۲ تا ۵۹ درصد زیان دادند. چارچوب‌های چندعاملی مثل TradingAgents فقط بک‌تست کوتاه دارند. پس مدل زبانی بدون یک لبه آماری "
     "پشت سرش معامله‌گر خوبی نیست؛ برای همین در این ربات هوش مصنوعی فقط حق وتو دارد.",
     [("Alpha Arena S1", "https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/"),
      ("TradingAgents", "https://arxiv.org/abs/2412.20138")]),
    ("مدل‌های پایه سری زمانی",
     "مقاله Kronos (آموزش روی ۱۲ میلیارد کندل) فقط معیارهای رتبه‌ای گزارش کرده، نه سود پس از هزینه. یک ارزیابی دقیق TimesFM "
     "نشان داد «دقت جهت» بالا اغلب همان نرخ پایه است. ما Chronos را روی داده بعد از انتشار مدل آزمودیم (جدول پایین).",
     [("Kronos", "https://arxiv.org/abs/2508.02739"), ("TimesFM base-rate benchmark", "https://arxiv.org/abs/2607.12248")]),
    ("مقالات ساعت‌های روز و فیکس",
     "الگوی W کرون، مولر و ویلان (برگشت قیمت حول فیکس‌ها، ۱۹۹۹ تا ۲۰۱۸) فقط با اسپرد بین‌بانکی شارپ ۰٫۵ تا ۰٫۸ دارد. "
     "بریدن و رانالدو نشان دادند هر ارز در ساعات کاری محلی خودش ضعیف می‌شود. اوانز برگشت پس از فیکس را در آخر ماه قوی‌تر یافت "
     "و کامینسکی و هینی نشت اطلاعات پیش از فیکس طلا را مستند کردند.",
     [("Krohn-Mueller-Whelan", "https://www.bis.org/events/221213_bis_bdi_ecb_exchange_rates/mueller.pdf"),
      ("Breedon-Ranaldo", "https://www.snb.ch/en/publications/research/working-papers/2011/working_paper_2011_04"),
      ("Evans", "https://pages.stern.nyu.edu/~jh4/SternMicroMtg/Old/SternMicroMtg2015/Papers/4ExEvans.pdf"),
      ("Gold fix leakage", "https://www.mining.com/web/a-leaky-fix/")]),
    ("اثرهای تقویمی و رویدادی",
     "به گفته کارنائوخ، دلار پیش از جلسه FOMC در جهت انتظار بازار حرکت می‌کند (شارپ ۰٫۹۳)، ولی این سیگنال قیمت آتی فدفاند لازم دارد. "
     "فصلی‌بودن طلا (قدرت در سپتامبر و نوامبر) پس از انتشار ضعیف شد. اثر جریان‌های آخر ماه هم محل مناقشه است.",
     [("Karnaukh", "https://acfr.aut.ac.nz/__data/assets/pdf_file/0019/190315/DollarAheadFOMC_NinaKarnaukh.pdf"),
      ("Gold seasonality (CXO)", "https://cxoadvisory.com/calendar-effects/gold-seasonality-drivers"),
      ("Month-end is dead?", "https://www.poundsterlinglive.com/eurusd/11424-month-end-is-dead-hsbc-s-donnelly")]),
    ("استراتژی‌های محبوب روزمعاملاتی",
     "تکرار مستقل مقاله ORB زاراتینی-عزیز سود ناخالص را بازتولید کرد ولی سود خالص صفر شد. بک‌تست ۱۰ ساله شکست لندن روی EURUSD "
     "(۲۰۱۶ تا ۲۰۲۶) ۲۸٪ زیان داد. برای ICT (سیلور بولت و جوداس) هیچ آمار ممیزی‌شده‌ای پیدا نشد.",
     [("ORB replication", "https://www.mql5.com/en/blogs/post/776235"),
      ("London breakout EURUSD", "https://backtrex.com/en/backtests/london-breakout-eur-usd"),
      ("ICT Silver Bullet", "https://backtrex.com/en/blog/ict-silver-bullet-strategy-trading-guide")]),
    ("معیار حرفه‌ای‌ها",
     "شاخص SG Trend (میانگین بزرگ‌ترین صندوق‌های روندگیر) در سال خوب ۲۰۲۶ تا آوریل ۱۰٫۳+٪ و در ۱۲ ماه منتهی به فوریه ۱۴٫۹+٪ بازده داشت؛ "
     "یعنی حتی بهترین مدیران سیستماتیک در سال خوب حدود ۱ تا ۱٫۲٪ در ماه می‌گیرند، نه ۱۰٪.",
     [("Top Traders Unplugged", "https://www.toptradersunplugged.com/trend-following-performance-report-february-2026/")]),
]

LAB_HYPOTHESES = [
    ("فروش طلا در شکست کف رنج آسیا:",
     "سمت فروشِ شکست لندن طلا در درون‌نمونه ۳٫۶+ واحد پایه خالص (t=1.8) و در برون‌نمونه ۲٫۸+ (t=1.05) داشت. چون این تفکیک پس از "
     "دیدن نتایج پیدا شد، فقط از ۱۲ اکتبر ۲۰۲۶ رو به جلو ثبت می‌شود."),
    ("تمرکز بازده طلا در ساعات شبانه:",
     "در ۲۰۱۹ تا ۲۰۲۶ خرید شبانه (۱۳:۳۰ تا ۰۸:۲۰ نیویورک) ۴٫۹+ واحد پایه خالص در هر شب داشت (t=2.4) و خرید در جلسه روزانه ۳٫۲- "
     "بود. در ۲۰۱۰ تا ۲۰۱۸ چنین الگویی نبود. اگر ادامه پیدا کند، می‌تواند زمان ورود دفتر سوئینگ طلا را بهتر کند. جلسه روزانه به‌عنوان "
     "گروه کنترل ثبت می‌شود."),
    ("ردشده:",
     "خلاف گپ آخر هفته یورو. سودش فقط با ورود دقیقاً در قیمت بازگشایی یکشنبه وجود داشت و با ۱۵ دقیقه تأخیر از ۴٫۳+ به ۰٫۱- رسید."),
]

LAB_CHART_NOTES = {
    "asof": "جمعه ۲۰۲۶-۱۰-۰۹",
    "EURUSD": [
        "روزانه نزولی است: سقف پایین‌تر 1.1654 (۹ سپتامبر) و کف پایین‌تر 1.1162 (۵ اکتبر). قیمت 1.1206 از سقف ژانویه (1.2024) "
        "حدود ۶٫۸٪ پایین آمده و زیر EMA20، EMA50 و EMA200 است. RSI روزانه حدود ۲۴ است، پس اشباع فروش و احتمال اصلاح صعودی "
        "کوتاه وجود دارد، ولی روند همچنان نزولی است.",
        "گپ‌های ارزش منصفانه (FVG) نزولی روزانه: 1.1285 تا 1.1329، 1.1400 تا 1.1431 و 1.1451 تا 1.1531. اوردربلاک‌های نزولی: "
        "1.1608 تا 1.1635 و 1.1713 تا 1.1797.",
        "چهارساعته هم نزولی است با سقف‌های پایین‌تر 1.1288، 1.1279 و 1.1246. نزدیک‌ترین مقاومت FVG نزولی 1.1241 تا 1.1255 است. "
        "یک‌ساعته و ۱۵ دقیقه در برگشت کوتاه صعودی داخل رنج 1.1176 تا 1.1246 هستند. اوردربلاک صعودی H1 در 1.1176 تا 1.1186 حمایت نزدیک است.",
        "سوگیری روزانه ربات ‎-1.00 است و هر چهار جزء آن نزولی‌اند: مومنتوم، EMA20/100، روند بازده ۲ساله آمریکا و روند شاخص دلار. "
        "دفتر سوئینگ در فروش است.",
        "سناریوی هم‌جهت با روند: فروش در اصلاح به 1.1240 تا 1.1290، پس از برگشت ساختار H1 به نزولی. حد ضرر بالای 1.1335 و تارگت‌ها "
        "1.1162 و 1.1100. بسته‌شدن کندل چهارساعته بالای 1.1330 این سناریو را باطل می‌کند و اصلاح عمیق‌تر تا 1.1400 تا 1.1430 را محتمل می‌کند.",
    ],
    "XAUUSD": [
        "روزانه: طلا پس از سقف تاریخی 5586 (۲۹ ژانویه ۲۰۲۶) حدود ۸ ماه است که در اصلاح است و تا 4091 (۷ اکتبر)، یعنی حدود ۲۷٪، پایین آمد. "
        "ساختار روزانه هنوز نزولی است (سقف پایین‌تر 4259 در ۲ اکتبر). قیمت 4216 روی حمایت مهمی است: اصلاح ۰٫۷۸۶ فیبوناچی موج 3786 تا 5586 "
        "در 4171 و اوردربلاک صعودی روزانه 4011 تا 4086.",
        "مقاومت‌های روزانه: اوردربلاک نزولی 4169 تا 4223 (قیمت الان داخل آن است) و FVG نزولی 4218 تا 4289. بالاتر از آن، اوردربلاک "
        "نزولی 4372 تا 4440.",
        "چهارساعته و یک‌ساعته صعودی شده‌اند: سقف بالاتر 4234 و کف بالاتر 4129. حمایت‌ها: اوردربلاک صعودی H1 در 4193 تا 4216، "
        "FVG صعودی H4 در 4171 تا 4201 و اوردربلاک صعودی H4 در 4091 تا 4150.",
        "سوگیری روزانه ربات ‎-0.25 است: مومنتوم خنثی، EMA50/200 نزولی، بازده واقعی نزولی و تمایل ساختاری خرید صعودی. چون کمتر از "
        "آستانه ۰٫۵ است، ربات وارد معامله نمی‌شود.",
        "سناریوها: بسته‌شدن کندل چهارساعته بالای 4292 مسیر را به 4310 تا 4350 و سپس 4372 تا 4440 باز می‌کند. اگر قیمت در 4234 تا 4290 "
        "رد شود، برگشت به 4171 تا 4194 و بعد 4091 تا 4150 محتمل است. بسته‌شدن روزانه زیر 4086 راه 3950 تا 4000 را باز می‌کند.",
    ],
}

LAB_SOURCES = [
    ("Krohn, Mueller & Whelan: FX fixings and returns around the clock (BIS)", "https://www.bis.org/events/221213_bis_bdi_ecb_exchange_rates/mueller.pdf"),
    ("Breedon & Ranaldo: Intraday patterns in FX returns (SNB WP 2011-04)", "https://www.snb.ch/en/publications/research/working-papers/2011/working_paper_2011_04"),
    ("Evans: fixing and month-end (NYU Stern)", "https://pages.stern.nyu.edu/~jh4/SternMicroMtg/Old/SternMicroMtg2015/Papers/4ExEvans.pdf"),
    ("Norges Bank: the WMR fix and its impact", "https://www.norges-bank.no/contentassets/619c8b75e1ed4ba691e8ad6a006855e6/39-panagiotou---the-wmr-fix-and-its-impact-on-currency-markets-.pdf"),
    ("Karnaukh: The dollar ahead of FOMC target rate changes", "https://acfr.aut.ac.nz/__data/assets/pdf_file/0019/190315/DollarAheadFOMC_NinaKarnaukh.pdf"),
    ("Macrosynergy: USD before FOMC decisions", "https://macrosynergy.com/?p=41075"),
    ("Mining.com: A leaky fix (gold fix)", "https://www.mining.com/web/a-leaky-fix/"),
    ("CXO Advisory: gold seasonality", "https://cxoadvisory.com/calendar-effects/gold-seasonality-drivers"),
    ("HSBC: month-end is dead (Pound Sterling Live)", "https://www.poundsterlinglive.com/eurusd/11424-month-end-is-dead-hsbc-s-donnelly"),
    ("ORB paper replicated: gross reproduced, net zero (MQL5)", "https://www.mql5.com/en/blogs/post/776235"),
    ("Backtrex: London breakout EUR/USD 10 years", "https://backtrex.com/en/backtests/london-breakout-eur-usd"),
    ("Backtrex: ICT Silver Bullet", "https://backtrex.com/en/blog/ict-silver-bullet-strategy-trading-guide"),
    ("Alpha Architect: intraday momentum", "https://alphaarchitect.com/2014/08/attention-prop-traders-the-first-half-hour-of-trading-predicts-the-last-half-hour/"),
    ("WH SelfInvest: Larry Williams volatility breakout", "https://www.whselfinvest.de/en-de/trading-platform/free-trading-strategies/tradingsystem/56-volatility-break-out-larry-williams-free"),
    ("Money Metals: gold up in Asia, down in the West (2026)", "https://www.moneymetals.com/news/2026/07/12/a-strange-dichotomy-gold-up-on-the-year-in-asian-markets-down-big-in-the-west-005059"),
    ("Kronos: foundation model for financial K-lines", "https://arxiv.org/abs/2508.02739"),
    ("When directional accuracy lies: TimesFM benchmark", "https://arxiv.org/abs/2607.12248"),
    ("TradingAgents: multi-agent LLM trading", "https://arxiv.org/abs/2412.20138"),
    ("Alpha Arena season 1 results", "https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/"),
    ("Do EAs actually work? Myfxbook data (MQL5)", "https://www.mql5.com/en/blogs/post/772467"),
    ("Telegram signal channel reviews", "https://www.telegramsignalsreviews.com/alltelegramsignalsreviews"),
    ("ESMA: CFD restrictions and retail loss rates", "https://www.esma.europa.eu/node/84933"),
    ("Tradeinformer: how FTMO makes money (FPFX 7%)", "https://tradeinformer.com/prop-weekly/how-does-ftmo-make-money"),
    ("Finance Magnates: FTMO $450M paid out", "https://www.financemagnates.com/forex/ftmo-announces-over-450-million-paid-out-as-prop-trading-firm-turns-10/"),
    ("Top Traders Unplugged: SG Trend Index 2026", "https://www.toptradersunplugged.com/trend-following-performance-report-february-2026/"),
    ("HistData.com free 1-minute data", "https://www.histdata.com/"),
    ("Amazon Chronos (GitHub)", "https://github.com/amazon-science/chronos-forecasting"),
]
