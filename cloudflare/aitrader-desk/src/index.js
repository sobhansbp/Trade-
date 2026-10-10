// aitrader-desk: live status page, owner-only Telegram commands and a watchdog for the
// hourly GitHub Actions bot. Nothing here trades: the Worker only reads the journal files
// the bot commits to the public repository and presents them.

const FILES = {
  state: "journal/paper_state.json",
  equity: "journal/paper_equity.csv",
  signals: "journal/signals.csv",
  trades: "journal/paper_trades.csv",
  forward: "journal/forward_tests.csv",
  news: "journal/news_view.json",
};
const SYMBOLS = ["EURUSD", "XAUUSD"];
const FA = { LONG: "خرید", SHORT: "فروش", WAIT: "صبر", trending: "روند‌دار", ranging: "رنج", transition: "گذار" };
const SYM_FA = { EURUSD: "یورو/دلار", XAUUSD: "طلا" };
const HYP_FA = {
  gold_london_breakout_short: "فروش طلا در شکست کف رنج آسیا",
  gold_overnight_long: "خرید شبانه طلا",
  gold_day_long: "کنترل: خرید طلا در جلسه روزانه",
};

// ------------------------------------------------------------------ data

async function getText(env, path) {
  try {
    const r = await fetch(`${env.REPO_RAW}/${path}`, { cf: { cacheTtl: 60, cacheEverything: true } });
    return r.ok ? await r.text() : null;
  } catch {
    return null;
  }
}

function tail(text, n) {
  // header + last n lines; the journal CSVs never contain embedded newlines
  if (!text) return text;
  const lines = text.trimEnd().split("\n");
  return lines.length <= n + 1 ? text : [lines[0], ...lines.slice(-n)].join("\n");
}

function parseCSV(text) {
  if (!text) return [];
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const head = rows.shift() || [];
  return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ""])));
}

function simpleCSV(text) {
  // fast path for files the bot writes without quoted fields (equity, signals)
  if (!text) return [];
  const lines = text.trimEnd().split("\n");
  const head = lines.shift().split(",");
  return lines.map((l) => { const c = l.split(","); return Object.fromEntries(head.map((h, i) => [h, c[i] ?? ""])); });
}

function equitySeries(text, maxRows = 1200, points = 300) {
  // last maxRows rows of paper_equity.csv, thinned to about `points` rows (always keeps the last one)
  if (!text) return [];
  const lines = text.trimEnd().split("\n");
  const head = lines[0].split(",");
  const col = (k) => head.indexOf(k);
  const [it, ia, is, ic] = ["time", "tactical", "swing", "combined"].map(col);
  const start = Math.max(1, lines.length - maxRows);
  const step = Math.max(1, Math.floor((lines.length - start) / points));
  const out = [];
  for (let i = start; i < lines.length; i += step) {
    const c = lines[i].split(",");
    out.push({ time: c[it], tactical: c[ia], swing: c[is], combined: c[ic] });
  }
  const lastLine = lines[lines.length - 1].split(",");
  if (out.length && out[out.length - 1].time !== lastLine[it]) {
    out.push({ time: lastLine[it], tactical: lastLine[ia], swing: lastLine[is], combined: lastLine[ic] });
  }
  return out;
}

const json = (t) => { try { return t ? JSON.parse(t) : null; } catch { return null; } };

async function loadAll(env) {
  const [state, equity, signals, trades, forward, news] = await Promise.all(
    Object.values(FILES).map((p) => getText(env, p)));
  const sig = simpleCSV(tail(signals, 60));
  const latest = {};
  for (const s of sig) latest[s.symbol] = s;          // chronological file: last row wins
  const eq = equitySeries(equity);                 // ~7 weeks of hourly rows, thinned: keeps CPU time small
  const last = eq.length ? new Date(eq[eq.length - 1].time) : null;
  return { state: json(state), equity: eq, latest, trades: parseCSV(tail(trades, 40)),
           forward: parseCSV(forward), news: json(news), updated: last };
}

// ------------------------------------------------------------------ formatting

const n = (x) => (x === "" || x == null ? NaN : Number(x));
const money = (x) => {   // manual grouping: Intl number formatting is comparatively slow on a cold isolate
  const v = n(x);
  return (v < 0 ? "-$" : "$") + Math.abs(v).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
};
const pct = (x, d = 2) => (n(x) >= 0 ? "+" : "") + (n(x) * 100).toFixed(d) + "%";
const sgn = (x, d = 2) => (Number.isFinite(n(x)) ? (n(x) >= 0 ? "+" : "") + n(x).toFixed(d) : "–");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const utc = (d) => (d ? d.toISOString().slice(0, 16).replace("T", " ") + " UTC" : "–");

function marketOpen(now = new Date()) {
  const d = now.getUTCDay(), h = now.getUTCHours();
  return (d === 0 && h >= 21) || (d >= 1 && d <= 4) || (d === 5 && h < 21);
}

function expectRuns(now = new Date()) {
  // the bot's cron runs hourly Sunday-Friday (UTC); give it time after the weekend gap
  const d = now.getUTCDay(), h = now.getUTCHours();
  return !((d === 6 && h >= 1) || (d === 0 && h < 3));
}

function staleMinutes(d, now = new Date()) {
  return d.updated ? (now - d.updated) / 60000 : Infinity;
}

function books(d) {
  const st = d.state;
  if (!st) return null;
  const init = st.initial || 10000;
  const t = st.tactical, s = st.swing;
  return {
    init, t, s, total: t.equity + s.equity,
    tRet: t.equity / init - 1, sRet: s.equity / init - 1,
    tDD: t.equity / t.peak - 1, sDD: s.equity / s.peak - 1,
    totRet: (t.equity + s.equity) / (2 * init) - 1,
  };
}

function forwardStats(rows) {
  const out = {};
  for (const r of rows) {
    if (Number(r.side) === 0) continue;
    const o = (out[r.hypothesis] ||= { n: 0, sum: 0, wins: 0 });
    o.n += 1; o.sum += n(r.net_bp); o.wins += n(r.net_bp) > 0 ? 1 : 0;
  }
  return out;
}

// ------------------------------------------------------------------ Telegram texts

const HELP = [
  "ربات میز معاملاتی یورو و طلا (حساب دمو)",
  "",
  "/status وضعیت حساب و آخرین تصمیم‌ها",
  "/eurusd تحلیل یورو/دلار",
  "/gold تحلیل طلا",
  "/positions پوزیشن‌های باز",
  "/forward آزمون رو به جلو فرضیه‌ها",
  "/news جمع‌بندی اخبار",
  "/dashboard لینک داشبورد زنده",
  "",
  "هر سؤال دیگری هم بنویسید، با داده‌های لحظه‌ای ربات جواب داده می‌شود. همه‌چیز تحلیل است، نه توصیه سرمایه‌گذاری.",
].join("\n");

function statusText(d, env) {
  const b = books(d);
  const lines = ["وضعیت حساب دمو"];
  if (b) {
    lines.push(`تاکتیکی ساعتی (آزمایشی): ${money(b.t.equity)} (${pct(b.tRet)})`,
               `سوئینگ روزانه: ${money(b.s.equity)} (${pct(b.sRet)})`,
               `جمع: ${money(b.total)} (${pct(b.totRet)})`);
  }
  lines.push("");
  for (const sym of SYMBOLS) {
    const s = d.latest[sym];
    if (!s) continue;
    lines.push(`${sym} ${s.price} → ${FA[s.ai_decision || s.decision] || s.decision} | سوگیری روزانه ${sgn(s.bias)} | میز ساعتی ${sgn(s.composite, 3)}`);
  }
  const stale = staleMinutes(d);
  lines.push("", `آخرین اجرای ربات: ${utc(d.updated)}` + (Number.isFinite(stale) ? ` (${Math.round(stale)} دقیقه پیش)` : ""),
             `بازار: ${marketOpen() ? "باز" : "بسته"}`);
  return lines.join("\n");
}

function symbolText(d, sym) {
  const s = d.latest[sym];
  if (!s) return `داده‌ای برای ${sym} نیست.`;
  const dec = s.ai_decision || s.decision;
  const lines = [`${sym} (${SYM_FA[sym]}) · ${s.time_utc.slice(0, 16)} UTC`,
                 `قیمت: ${s.price}`,
                 `تصمیم: ${FA[dec] || dec}` + (s.ai_decision && s.ai_decision !== s.decision ? ` (موتور کمّی: ${FA[s.decision]})` : ""),
                 `سوگیری روزانه: ${sgn(s.bias)} (معامله فقط وقتی |سوگیری| ≥ 0.5)`,
                 `برآیند میز ۱۲ تحلیلگر ساعتی: ${sgn(s.composite, 3)} · رژیم: ${FA[s.regime] || s.regime}`];
  if (s.entry) {
    lines.push(`${dec === "WAIT" ? "پلن مشروط" : "پلن"}: ورود ${s.entry} · حد ضرر ${s.stop} · تارگت ۱ (۵۰٪) ${s.tp1} · تارگت نهایی ${s.tp2}`);
  }
  const sw = d.state?.swing?.positions?.[sym];
  if (sw) lines.push(`دفتر سوئینگ: ${(sw.frac * 100).toFixed(0)}% سرمایه دفتر`);
  const nv = d.news?.view?.symbols?.[sym];
  if (nv) lines.push("", `اخبار: ${sgn(nv.score)}${nv.event_risk ? " (ریسک رویداد)" : ""}`, nv.rationale_fa || "");
  return lines.join("\n");
}

function positionsText(d) {
  const st = d.state;
  if (!st) return "داده حساب در دسترس نیست.";
  const lines = ["پوزیشن‌های تاکتیکی:"];
  const tp = Object.entries(st.tactical.positions || {});
  if (!tp.length) lines.push("پوزیشن بازی نیست.");
  for (const [sym, p] of tp) {
    lines.push(`${sym} ${FA[p.side] || p.side} · ورود ${p.entry} · حد ضرر ${p.stop} · TP1 ${p.tp1} · TP2 ${p.tp2} · ریسک ${money(p.risk_usd)}`);
  }
  lines.push("", "دفتر سوئینگ (درصد سرمایه دفتر):");
  for (const [sym, p] of Object.entries(st.swing.positions || {})) lines.push(`${sym}: ${(p.frac * 100).toFixed(0)}%`);
  return lines.join("\n");
}

function forwardText(d) {
  const fs = forwardStats(d.forward);
  const lines = ["آزمون رو به جلو (فقط ثبت، بدون معامله؛ از ۱۲ اکتبر ۲۰۲۶):"];
  for (const [k, label] of Object.entries(HYP_FA)) {
    const o = fs[k];
    lines.push(o ? `${label}: ${o.n} معامله · میانگین ${sgn(o.sum / o.n)} واحد پایه · برد ${Math.round((100 * o.wins) / o.n)}%`
                 : `${label}: هنوز معامله‌ای ثبت نشده`);
  }
  return lines.join("\n");
}

function newsText(d) {
  const v = d.news?.view;
  if (!v) return "هنوز جمع‌بندی خبری ثبت نشده.";
  const lines = [`جمع‌بندی اخبار (${v.n_headlines} تیتر، ${d.news.scored_at.slice(0, 16).replace("T", " ")} UTC):`, v.theme_fa || ""];
  for (const sym of SYMBOLS) {
    const s = v.symbols?.[sym];
    if (s) lines.push("", `${sym}: ${sgn(s.score)}${s.event_risk ? " · ریسک رویداد" : ""}`, s.rationale_fa || "");
  }
  return lines.join("\n");
}

// ------------------------------------------------------------------ Telegram plumbing

async function tg(env, method, payload) {
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload),
  });
  return r.ok;
}

async function reply(env, chat, text) {
  for (let i = 0; i < text.length; i += 3900) {
    await tg(env, "sendMessage", { chat_id: chat, text: text.slice(i, i + 3900), disable_web_page_preview: true });
  }
}

async function askGroq(env, question, d, origin) {
  const hour = new Date().toISOString().slice(0, 13);
  const key = `qa:${hour}`;
  const used = Number((await env.KV.get(key)) || 0);
  if (used >= Number(env.QA_PER_HOUR || 20)) return "سقف سؤال در این ساعت پر شده است. کمی بعد دوباره بپرسید.";
  await env.KV.put(key, String(used + 1), { expirationTtl: 3700 });
  const ctx = {
    now_utc: new Date().toISOString(), market_open: marketOpen(), last_bot_run_utc: d.updated?.toISOString(),
    accounts: books(d), swing_positions: d.state?.swing?.positions, tactical_positions: d.state?.tactical?.positions,
    latest_signals: d.latest, news: d.news?.view ? { theme_fa: d.news.view.theme_fa, symbols: d.news.view.symbols } : null,
    forward_tests: forwardStats(d.forward), recent_closed_trades: d.trades.slice(-10),
  };
  const system = [
    "تو دستیار میز معاملاتی aitrader هستی و به صاحب حساب دمو، به فارسی ساده و کوتاه جواب می‌دهی.",
    "فقط از داده‌های JSON زیر استفاده کن و هیچ قیمت، سطح یا عددی از خودت نساز. اگر داده‌ای نیست، صادقانه بگو.",
    "قوانین ربات: فقط در جهت سوگیری روزانه قوی (|bias| ≥ 0.5) و با تأیید میز ۱۲ تحلیلگر ساعتی (composite هم‌جهت > 0.05) معامله می‌کند؛",
    "ریسک هر معامله ۰٫۵٪، ۵۰٪ در +1R بسته و حد ضرر سر به سر، تارگت نهایی 2.5R یا ۷۲ ساعت؛ دفتر سوئینگ موقعیت را متناسب با سوگیری و هدف نوسان ۱۰٪ نگه می‌دارد.",
    "هوش مصنوعی فقط حق وتو دارد. اعتبارسنجی ۱۶ ساله نشان داد بخش ساعتی (دفتر تاکتیکی) در ۲۰۱۴ تا ۲۰۲۶ پس از هزینه سود نداد (ضریب سود ۰٫۹۵ تا ۱٫۰۰) و آزمایشی است؛",
    "دفتر سوئینگ روزانه تنها لایه با شواهد مثبت بلندمدت است (۲۰۰۴ تا ۲۰۲۶، نوسان ۱۰٪: بازده سالانه حدود ۷٪، افت حداکثر حدود ۲۰٪)؛",
    "سود ۱۰٪ ماهانه از نظر آماری واقع‌بینانه نیست. در پایان یک جمله کوتاه بگو که این تحلیل است، نه توصیه سرمایه‌گذاری.",
    "داده‌ها:", JSON.stringify(ctx),
  ].join("\n");
  try {
    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${env.GROQ_API_KEY}` },
      body: JSON.stringify({ model: env.GROQ_MODEL, temperature: 0.2, max_completion_tokens: 1200, reasoning_effort: "low",
                             messages: [{ role: "system", content: system }, { role: "user", content: question.slice(0, 1500) }] }),
    });
    if (!r.ok) return `هوش مصنوعی در دسترس نیست (کد ${r.status}). دستورهای /status و /gold و /eurusd همچنان کار می‌کنند.`;
    const out = await r.json();
    return out?.choices?.[0]?.message?.content?.trim() || "پاسخی دریافت نشد.";
  } catch (e) {
    return "خطا در تماس با هوش مصنوعی. لطفاً دوباره امتحان کنید.";
  }
}

async function handleUpdate(env, update, origin) {
  const msg = update.message || update.edited_message;
  if (!msg?.text) return;
  const chat = String(msg.chat.id);
  if (chat !== String(env.TELEGRAM_CHAT_ID)) {
    await reply(env, chat, "این ربات خصوصی است.");
    return;
  }
  const text = msg.text.trim();
  // slash commands match on the first word; bare Persian keywords only when sent alone
  const first = text.split(/\s+/)[0].toLowerCase().replace(/@\w+$/, "");
  const cmd = first.startsWith("/") ? first : text;
  const d = await loadAll(env);
  const routes = {
    "/start": () => HELP, "/help": () => HELP, "راهنما": () => HELP,
    "/status": () => statusText(d, env), "وضعیت": () => statusText(d, env),
    "/eurusd": () => symbolText(d, "EURUSD"), "/euro": () => symbolText(d, "EURUSD"), "یورو": () => symbolText(d, "EURUSD"),
    "/gold": () => symbolText(d, "XAUUSD"), "/xauusd": () => symbolText(d, "XAUUSD"), "طلا": () => symbolText(d, "XAUUSD"),
    "/positions": () => positionsText(d), "/forward": () => forwardText(d), "/news": () => newsText(d),
    "/dashboard": () => `داشبورد زنده: ${origin}/` + (env.DASHBOARD_URL ? `\nداشبورد کامل پژوهش: ${env.DASHBOARD_URL}` : ""),
  };
  const out = routes[cmd] ? routes[cmd]() : await askGroq(env, text, d, origin);
  await reply(env, chat, out);
}

// ------------------------------------------------------------------ watchdog

async function watchdog(env) {
  const d = await loadAll(env);
  const now = new Date();
  const stale = staleMinutes(d, now);
  const limit = Number(env.STALE_MINUTES || 150);
  const down = await env.KV.get("watchdog:down");
  if (expectRuns(now) && stale > limit) {
    if (!(await env.KV.get("watchdog:alerted"))) {
      await reply(env, env.TELEGRAM_CHAT_ID,
        `هشدار نگهبان: ربات معاملاتی ${Number.isFinite(stale) ? Math.round(stale) + " دقیقه" : "مدتی"} است ژورنالش را به‌روز نکرده ` +
        `(آخرین اجرا ${utc(d.updated)}). تب Actions گیت‌هاب را بررسی کنید: ${env.REPO_URL.replace("/tree/", "/actions?query=branch%3A")}`);
      await env.KV.put("watchdog:alerted", now.toISOString(), { expirationTtl: 6 * 3600 });
    }
    await env.KV.put("watchdog:down", "1");
  } else if (down && stale <= limit) {
    await reply(env, env.TELEGRAM_CHAT_ID, `ربات دوباره فعال است. آخرین اجرا: ${utc(d.updated)}`);
    await env.KV.delete("watchdog:down");
    await env.KV.delete("watchdog:alerted");
  }
  await env.KV.put("watchdog:last_check", JSON.stringify({ at: now.toISOString(), stale_min: Math.round(stale) }));
}

// ------------------------------------------------------------------ page

function equitySvg(rows, init) {
  // percent return of each book and of the combined two-book account
  const pts = rows.map((r) => ({ t: Date.parse(r.time), c: (n(r.combined) / (2 * init) - 1) * 100,
                                 a: (n(r.tactical) / init - 1) * 100, s: (n(r.swing) / init - 1) * 100 }))
                  .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.c));
  if (pts.length < 2) return `<p class="muted">منحنی سرمایه پس از چند اجرای ربات نمایش داده می‌شود.</p>`;
  const p = pts;
  const W = 720, H = 220, L = 52, R = 12, T = 10, B = 24;
  const t0 = p[0].t, t1 = p[p.length - 1].t;
  const vals = p.flatMap((q) => [q.a, q.s, q.c]);
  let lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
  if (hi - lo < 1) { const m = (hi + lo) / 2; lo = m - 0.5; hi = m + 0.5; }
  const X = (t) => L + ((t - t0) / Math.max(1, t1 - t0)) * (W - L - R);
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const line = (f) => p.map((q, i) => `${i ? "L" : "M"}${X(q.t).toFixed(1)},${Y(f(q)).toFixed(1)}`).join("");
  const ticks = [lo, 0, hi].map((v) =>
    `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v >= 0 ? "+" : ""}${v.toFixed(2)}%</text>` +
    `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="${v === 0 ? "zero" : "grid"}"/>`).join("");
  const d0 = new Date(t0).toISOString().slice(0, 10), d1 = new Date(t1).toISOString().slice(0, 10);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="equity curve" class="eq">${ticks}
    <path d="${line((q) => q.a)}" class="a"/><path d="${line((q) => q.s)}" class="s"/><path d="${line((q) => q.c)}" class="c"/>
    <text x="${L}" y="${H - 6}">${d0}</text><text x="${W - R}" y="${H - 6}" text-anchor="end">${d1}</text></svg>
    <div class="legend"><span class="lc">حساب ترکیبی</span><span class="la">دفتر تاکتیکی</span><span class="ls">دفتر سوئینگ</span></div>`;
}

function symbolCard(d, sym) {
  const s = d.latest[sym];
  if (!s) return "";
  const dec = s.ai_decision || s.decision;
  const bias = n(s.bias);
  const pos = ((1 - bias) / 2) * 100;
  const nv = d.news?.view?.symbols?.[sym];
  const sw = d.state?.swing?.positions?.[sym];
  return `<article class="card">
    <div class="row"><span class="sym">${sym}</span><span class="pill ${esc(dec)}">${FA[dec] || esc(dec)}</span></div>
    <div class="kv">
      <div><span>قیمت</span><b>${esc(s.price)}</b></div>
      <div><span>میز ساعتی</span><b>${sgn(s.composite, 3)}</b></div>
      <div><span>رژیم</span><b class="fa">${FA[s.regime] || esc(s.regime)}</b></div>
    </div>
    <div class="meter-l"><span>سوگیری روزانه <b class="num">${sgn(bias)}</b></span><span>${esc(s.time_utc.slice(0, 16))} UTC</span></div>
    <div class="meter"><i style="left:calc(${pos.toFixed(0)}% - 2px)"></i></div>
    <div class="meter-l ltr"><span>bullish +1</span><span>0</span><span>-1 bearish</span></div>
    ${s.entry ? `<h3>${dec === "WAIT" ? "پلن مشروط" : "پلن معامله"}</h3>
    <div class="plan"><div><span>ورود</span><b>${esc(s.entry)}</b></div><div><span>حد ضرر</span><b>${esc(s.stop)}</b></div>
      <div><span>تارگت ۱</span><b>${esc(s.tp1)}</b></div><div><span>تارگت نهایی</span><b>${esc(s.tp2)}</b></div></div>` : ""}
    <p class="muted small">${sw ? `دفتر سوئینگ: <b class="num">${(sw.frac * 100).toFixed(0)}%</b> سرمایه دفتر. ` : ""}
      ${nv ? `اخبار: <b class="num">${sgn(nv.score)}</b>${nv.event_risk ? " · ریسک رویداد" : ""}. ${esc(nv.rationale_fa || "")}` : ""}</p>
  </article>`;
}

function page(d, env, origin) {
  const b = books(d);
  const stale = staleMinutes(d);
  const alive = !expectRuns() || stale <= Number(env.STALE_MINUTES || 150);
  const kpi = (label, eq, ret, dd) => `<div class="kpi"><span>${label}</span><b>${money(eq)}</b>
      <small class="${ret >= 0 ? "good" : "bad"}"><span class="num">${pct(ret)}</span> بازده</small>${dd !== undefined ? `<small class="muted">افت از سقف <span class="num">${pct(dd)}</span></small>` : ""}</div>`;
  const tpos = Object.entries(d.state?.tactical?.positions || {});
  const fs = forwardStats(d.forward);
  const trades = d.trades.slice(-10).reverse();
  const v = d.news?.view;
  return `<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="300">
<title>میز معاملاتی زنده</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{--bg:#f3f5f8;--surface:#fff;--fg:#18212c;--muted:#5f6b7a;--line:#dde2ea;--brass:#a8741f;--buy:#1d7f55;--sell:#b83a33;--wait:#6d6a86;
--buy-bg:#e3f3ea;--sell-bg:#f8e5e3;--wait-bg:#ecebf3;--brass-bg:#f6eedf}
@media (prefers-color-scheme:dark){:root{--bg:#0f141a;--surface:#171e27;--fg:#e5e9ef;--muted:#97a2b1;--line:#2a3440;--brass:#d6a64f;
--buy:#4cc28d;--sell:#ef7a70;--wait:#a9a6c6;--buy-bg:#14302a;--sell-bg:#3a1e1d;--wait-bg:#252438;--brass-bg:#2d2618;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.75 Vazirmatn,Tahoma,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 16px 56px;display:flex;flex-direction:column;gap:20px}
h1{font-size:1.5rem;margin:0}h2{font-size:1.15rem;margin:0 0 8px}h3{font-size:.95rem;color:var(--muted);margin:10px 0 6px}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:10px;border-bottom:2px solid var(--fg);padding-bottom:12px}
.eyebrow{font-size:.75rem;letter-spacing:.06em;color:var(--brass);font-weight:700}
.muted{color:var(--muted)}.small{font-size:.85rem}.good{color:var(--buy)}.bad{color:var(--sell)}
.num,.kv b,.plan b,.kpi b,td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;direction:ltr;unicode-bidi:isolate}
.status{display:inline-flex;gap:8px;flex-wrap:wrap}.badge{border-radius:999px;padding:2px 10px;font-size:.8rem;font-weight:700;background:var(--wait-bg);color:var(--wait)}
.badge.on{background:var(--buy-bg);color:var(--buy)}.badge.off{background:var(--sell-bg);color:var(--sell)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card,.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}
.kpi{display:flex;flex-direction:column;gap:2px}.kpi span{font-size:.8rem;color:var(--muted)}.kpi b{font-size:1.45rem;text-align:right}
.row{display:flex;justify-content:space-between;align-items:center}.sym{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:1.2rem}
.pill{border-radius:999px;padding:2px 12px;font-weight:800}.pill.LONG{background:var(--buy-bg);color:var(--buy)}
.pill.SHORT{background:var(--sell-bg);color:var(--sell)}.pill.WAIT{background:var(--wait-bg);color:var(--wait)}
.kv{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:10px 0}.kv div,.plan div{display:flex;flex-direction:column}
.kv span,.plan span{font-size:.75rem;color:var(--muted)}.kv b.fa{font-family:inherit;direction:rtl}
.meter{position:relative;height:10px;border-radius:5px;background:linear-gradient(90deg,var(--buy),var(--line) 50%,var(--sell));direction:ltr}
.meter i{position:absolute;top:-4px;width:4px;height:18px;border-radius:2px;background:var(--fg)}
.meter-l{display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted)}.ltr{direction:ltr}
.plan{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;background:var(--brass-bg);border-radius:8px;padding:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
th{font-size:.75rem;color:var(--muted)}.scroll{overflow-x:auto}
svg.eq{width:100%;height:auto;font:11px "IBM Plex Mono",monospace;fill:var(--muted);direction:ltr}svg.eq path{fill:none;stroke-width:2}
svg.eq .c{stroke:var(--brass)}svg.eq .a{stroke:#3498db;stroke-width:1.3}svg.eq .s{stroke:#9b59b6;stroke-width:1.3}svg.eq .grid{stroke:var(--line)}svg.eq .zero{stroke:var(--muted);stroke-dasharray:3 3}
.legend{display:flex;gap:14px;font-size:.8rem;color:var(--muted)}.legend span:before{content:"";display:inline-block;width:12px;height:3px;margin-inline-end:6px;vertical-align:middle}
.lc:before{background:var(--brass)}.la:before{background:#3498db}.ls:before{background:#9b59b6}
a{color:var(--brass)}footer{font-size:.8rem;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
@media (max-width:520px){.plan{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><div class="wrap">
<header class="top"><div><div class="eyebrow">AI TRADING DESK · LIVE · PAPER ACCOUNT</div><h1>میز معاملاتی یورو و طلا</h1>
<div class="status"><span class="badge ${alive ? "on" : "off"}">ربات ${alive ? "فعال" : "متوقف"}</span>
<span class="badge ${marketOpen() ? "on" : ""}">بازار ${marketOpen() ? "باز" : "بسته"}</span></div></div>
<div class="muted small">آخرین اجرای ربات: <span class="num">${utc(d.updated)}</span><br>بازخوانی خودکار هر ۵ دقیقه</div></header>
${b ? `<div class="kpis">${kpi("جمع دو دفتر", b.total, b.totRet)}${kpi("دفتر تاکتیکی ساعتی (آزمایشی)", b.t.equity, b.tRet, b.tDD)}${kpi("دفتر سوئینگ روزانه", b.s.equity, b.sRet, b.sDD)}</div>` : ""}
<div class="grid">${SYMBOLS.map((s) => symbolCard(d, s)).join("")}</div>
<section class="card"><h2>منحنی سرمایه (بازده درصدی)</h2>${equitySvg(d.equity, d.state?.initial || 10000)}</section>
<div class="grid">
<section class="card"><h2>پوزیشن‌های تاکتیکی</h2>${tpos.length ? `<div class="scroll"><table><thead><tr><th>نماد</th><th>جهت</th><th>ورود</th><th>حد ضرر</th><th>TP1</th><th>TP2</th><th>ریسک</th></tr></thead><tbody>
${tpos.map(([sym, p]) => `<tr><td class="num">${sym}</td><td>${FA[p.side] || esc(p.side)}</td><td class="num">${esc(p.entry)}</td><td class="num">${esc(p.stop)}</td><td class="num">${esc(p.tp1)}</td><td class="num">${esc(p.tp2)}</td><td class="num">${money(p.risk_usd)}</td></tr>`).join("")}
</tbody></table></div>` : `<p class="muted">پوزیشن بازی نیست. ربات فقط وقتی وارد می‌شود که سوگیری روزانه قوی باشد و میز ساعتی تأیید کند.</p>`}</section>
<section class="card"><h2>آزمون رو به جلو فرضیه‌ها</h2><p class="muted small">فقط ثبت می‌شوند، بدون معامله؛ از ۱۲ اکتبر ۲۰۲۶.</p>
<div class="scroll"><table><thead><tr><th>فرضیه</th><th>معامله</th><th>میانگین خالص (bp)</th><th>برد</th></tr></thead><tbody>
${Object.entries(HYP_FA).map(([k, label]) => { const o = fs[k]; return `<tr><td>${label}</td><td class="num">${o ? o.n : 0}</td><td class="num">${o ? sgn(o.sum / o.n) : "–"}</td><td class="num">${o ? Math.round((100 * o.wins) / o.n) + "%" : "–"}</td></tr>`; }).join("")}
</tbody></table></div></section></div>
${trades.length ? `<section class="card"><h2>معاملات بسته‌شده</h2><div class="scroll"><table><thead><tr><th>زمان</th><th>نماد</th><th>جهت</th><th>ورود</th><th>خروج</th><th>نتیجه</th><th>سود/زیان</th></tr></thead><tbody>
${trades.map((r) => `<tr><td class="num">${esc(String(r.closed).slice(0, 16))}</td><td class="num">${esc(r.symbol)}</td><td>${FA[r.side] || esc(r.side)}</td><td class="num">${esc(r.entry)}</td><td class="num">${esc(r.exit)}</td><td class="num">${esc(r.reason)} ${sgn(r.r_multiple)}R</td><td class="num ${n(r.pnl_usd) >= 0 ? "good" : "bad"}">${money(r.pnl_usd)}</td></tr>`).join("")}
</tbody></table></div></section>` : ""}
${v ? `<section class="card"><h2>جمع‌بندی اخبار</h2><p>${esc(v.theme_fa || "")}</p><p class="muted small">${v.n_headlines} تیتر · ${esc(v.model || "")} · ${esc(String(d.news.scored_at).slice(0, 16).replace("T", " "))} UTC</p></section>` : ""}
<section class="card"><h2>دسترسی</h2><p>ربات تلگرام: <a href="https://t.me/Tradeinfoirbot">@Tradeinfoirbot</a> (دستورهای /status، /gold، /eurusd، /forward و پرسش آزاد).
کد و ژورنال کامل: <a href="${esc(env.REPO_URL)}">گیت‌هاب</a> · داده خام: <a href="${origin}/api/status">/api/status</a>
${env.DASHBOARD_URL ? ` · داشبورد کامل پژوهش و تحلیل: <a href="${esc(env.DASHBOARD_URL)}">claude.ai</a>` : ""}</p></section>
<footer>حساب کاغذی است و هیچ سفارشی به بروکر ارسال نمی‌شود. ربات هر ساعت روی GitHub Actions اجرا می‌شود و این صفحه روی Cloudflare از ژورنال آن خوانده می‌شود. تحلیل است، نه توصیه سرمایه‌گذاری.</footer>
</div></body></html>`;
}

// ------------------------------------------------------------------ entry points

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const origin = url.origin;
    if (url.pathname === "/tg" && request.method === "POST") {
      if (request.headers.get("x-telegram-bot-api-secret-token") !== env.TG_WEBHOOK_SECRET) {
        return new Response("forbidden", { status: 403 });
      }
      const update = await request.json().catch(() => null);
      if (update) ctx.waitUntil(handleUpdate(env, update, origin).catch(() => {}));
      return new Response("ok");
    }
    const cacheable = request.method === "GET" && (url.pathname === "/" || url.pathname === "/api/status");
    const cacheKey = new Request(origin + url.pathname);
    if (cacheable) {   // the journal changes hourly; a 60 s edge cache keeps CPU time per visit tiny
      const hit = await caches.default.match(cacheKey);
      if (hit) return hit;
    }
    const put = (resp) => { ctx.waitUntil(caches.default.put(cacheKey, resp.clone())); return resp; };
    if (url.pathname === "/api/status") {
      const d = await loadAll(env);
      const body = { updated: d.updated, market_open: marketOpen(), stale_minutes: Math.round(staleMinutes(d)),
                     accounts: books(d), latest: d.latest, positions: d.state?.tactical?.positions,
                     swing: d.state?.swing?.positions, forward: forwardStats(d.forward),
                     news: d.news?.view ? { theme_fa: d.news.view.theme_fa, symbols: d.news.view.symbols } : null };
      return put(new Response(JSON.stringify(body, null, 1), {
        headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*",
                   "cache-control": "public, max-age=60" } }));
    }
    if (url.pathname === "/health") return new Response("ok");
    if (url.pathname === "/" || url.pathname === "/index.html") {
      const d = await loadAll(env);
      return put(new Response(page(d, env, origin), {
        headers: { "content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=60" } }));
    }
    return new Response("not found", { status: 404 });
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(watchdog(env));
  },
};
