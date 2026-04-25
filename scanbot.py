import asyncio
import aiohttp
import time
import requests
from datetime import datetime

# ================= CONFIG =================
TELEGRAM_TOKEN = "8649950519:AAHb4UUejJZJuVuQjqL8nBqj69FW1k3tTmg"

TELEGRAM_CHAT_ID_BULL = "-1003965900583"
TELEGRAM_CHAT_ID_BEAR = "-1003723283209"

BINANCE = "https://fapi.binance.com"
SCAN_INTERVAL = 60
CONCURRENCY = 10
ALERT_COOLDOWN = 60 * 60 * 2  # 2 hours

_alerted = {}

# ================= TELEGRAM =================
def send_telegram(msg, direction):
    chat_id = TELEGRAM_CHAT_ID_BULL if direction == "BULL" else TELEGRAM_CHAT_ID_BEAR

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "HTML"
    }

    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print("Telegram error:", e)

# ================= UTIL =================
def now():
    return int(time.time())

def seconds_to_close(tf_sec):
    return tf_sec - (now() % tf_sec)

def near_close(tf_sec, window=180):
    return seconds_to_close(tf_sec) <= window

def cooldown_key(symbol, direction):
    return f"{symbol}:{direction}"

def on_cooldown(key):
    return now() - _alerted.get(key, 0) < ALERT_COOLDOWN

def mark_alert(key):
    _alerted[key] = now()

# ================= FETCH =================
async def fetch_json(session, url, retries=3):
    for _ in range(retries):
        try:
            async with session.get(url) as r:
                return await r.json()
        except:
            await asyncio.sleep(1)
    return None

async def get_symbols(session):
    data = await fetch_json(session, f"{BINANCE}/fapi/v1/exchangeInfo")
    return [
        s["symbol"] for s in data["symbols"]
        if s["status"] == "TRADING" and s["symbol"].endswith("USDT")
    ]

async def get_klines(session, symbol, interval, limit=50):
    url = f"{BINANCE}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    return await fetch_json(session, url)

async def get_oi(session, symbol):
    url = f"{BINANCE}/fapi/v1/openInterest?symbol={symbol}"
    data = await fetch_json(session, url)
    return float(data["openInterest"]) if data else None

# ================= ANALYSIS =================
def candle_delta(c):
    return float(c[4]) - float(c[1])

def volume_ratio(klines):
    vols = [float(k[5]) for k in klines[:-1]]
    avg = sum(vols) / len(vols)
    last = float(klines[-1][5])
    return last / avg if avg > 0 else 0

def oi_change(prev, curr):
    if prev == 0:
        return 0
    return (curr - prev) / prev * 100

def get_direction(delta):
    return "BULL" if delta > 0 else "BEAR"

def rejection(direction, delta):
    return (direction == "BULL" and delta < 0) or (direction == "BEAR" and delta > 0)

def score_model(r):
    score = 0
    if r["htf_bias"] == r["direction"]: score += 6
    if r["exh_vol"]: score += 5
    if abs(r["ltf_oi_pct"]) > 2.5: score += 6
    if r["delta_flip"]: score += 5
    if r["rejection"]: score += 5
    return score

def get_tier(score):
    if score >= 26:
        return "A+"
    elif score >= 22:
        return "A"
    elif score >= 18:
        return "B"
    return "C"

# ================= CORE =================
async def analyze(session, symbol):
    k15 = await get_klines(session, symbol, "15m", 50)
    k1h = await get_klines(session, symbol, "1h", 50)
    k4h = await get_klines(session, symbol, "4h", 50)

    if not k15 or not k1h or not k4h:
        return None

    ltf_delta = candle_delta(k15[-1])
    mtf_delta = candle_delta(k1h[-1])
    htf_delta = candle_delta(k4h[-1])

    direction = get_direction(ltf_delta)
    htf_bias = get_direction(htf_delta)

    vol_ratio_val = volume_ratio(k15)
    exh_vol = vol_ratio_val > 2

    oi_now = await get_oi(session, symbol)
    await asyncio.sleep(0.05)
    oi_prev = await get_oi(session, symbol)

    if not oi_now or not oi_prev:
        return None

    ltf_oi_pct = oi_change(oi_prev, oi_now)
    delta_flip = (ltf_delta * mtf_delta) < 0
    rej = rejection(direction, ltf_delta)

    r = {
        "symbol": symbol,
        "direction": direction,
        "htf_bias": htf_bias,
        "ltf_delta": ltf_delta,
        "mtf_delta": mtf_delta,
        "htf_delta": htf_delta,
        "vol_ratio": vol_ratio_val,
        "exh_vol": exh_vol,
        "ltf_oi_pct": ltf_oi_pct,
        "delta_flip": delta_flip,
        "rejection": rej
    }

    r["score"] = score_model(r)
    r["tier"] = get_tier(r["score"])

    return r

# ================= SCAN =================
async def scan_symbol(session, symbol, sem):
    async with sem:
        r = await analyze(session, symbol)
        if not r:
            return

        key = cooldown_key(symbol, r["direction"])
        if on_cooldown(key):
            return

        if not near_close(900, 180):
            return

        mark_alert(key)

        msg = f"""
🚨 <b>{r['tier']} SIGNAL {r['direction']} — {symbol}</b>

📊 Score: {r['score']}
📈 HTF Bias: {r['htf_bias']}
📊 OI Δ: {round(r['ltf_oi_pct'],2)}%
📊 Volume: {round(r['vol_ratio'],2)}x
🔁 Delta Flip: {r['delta_flip']}
🧠 Rejection: {r['rejection']}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""

        send_telegram(msg, r["direction"])

# ================= MAIN =================
async def main():
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        symbols = await get_symbols(session)

        while True:
            sem = asyncio.Semaphore(CONCURRENCY)
            tasks = [scan_symbol(session, s, sem) for s in symbols]
            await asyncio.gather(*tasks)
            await asyncio.sleep(SCAN_INTERVAL)

if _name_ == "_main_":
    asyncio.run(main())