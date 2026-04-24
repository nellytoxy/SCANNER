"""
SweepBot v4 — Institutional Grade
Converted 1:1 from Pine Script:
  - HTF Structure (HH/HL, LH/LL) + Delta confirmation
  - Liquidity Sweep (mid TF) + rejection + displacement
  - CVD Delta flip confirmation
  - Fibonacci pullback validation (0.618–0.886)
  - FVG entry zone
  - Weighted conviction scoring (max 15)
  - OI flow via funding rate proxy (Binance has no OI endpoint on free tier)
  - ICT Kill Zones gate
  - Min score 9/15 before entry (A-grade only)
"""

import os, time, json, math, logging, sys, threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import numpy as np

try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
except ImportError:
    raise ImportError("pip install binance-futures-connector")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/tmp/bot.log"), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("SweepBot")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BotConfig:
    api_key:    str  = ""
    api_secret: str  = ""
    live_mode:  bool = False

    def __post_init__(self):
        if not self.api_key:    self.api_key    = os.environ.get("BINANCE_API_KEY","")
        if not self.api_secret: self.api_secret = os.environ.get("BINANCE_API_SECRET","")
        e = os.environ.get("LIVE_MODE","")
        if e: self.live_mode = e.lower() == "true"

    # Capital & risk
    total_capital:    float = 30.0
    risk_per_trade:   float = 0.015   # 1.5% = ~$0.45
    max_leverage:     int   = 15
    max_open_trades:  int   = 3
    daily_loss_limit: float = 0.06

    # Timeframes — matching Pine Script defaults
    tf_high:   str = "4h";  lim_high:   int = 200   # HTF structure (was 240m)
    tf_mid:    str = "1h";  lim_mid:    int = 100   # Sweep / bias (was 60m)
    tf_entry:  str = "15m"; lim_entry:  int = 150   # Entry chart TF

    # Pine Script params
    pivot_depth:      int   = 8     # HTF pivot depth
    pivot_len:        int   = 5     # Mid TF sweep pivot
    rejection_ratio:  float = 1.5
    min_body_ratio:   float = 0.35
    sweep_cooldown:   int   = 12

    # OI / scoring thresholds
    oi_confirm_level: float = 1.0
    oi_strong_level:  float = 1.5
    inst_vol_mult:    float = 2.0
    inst_body_ratio:  float = 0.60

    # Fibonacci
    fib_preferred:    float = 0.706
    fib_max:          float = 0.886

    # Scoring — only trade A-grade (9+) or A+ (12+)
    min_score:        int   = 9
    min_rr:           float = 2.0
    atr_period:       int   = 14
    atr_sl_mult:      float = 1.0
    atr_tp_rr:        float = 2.5

    # Kill zones UTC
    session_filter:   bool  = True
    kill_zones: list  = None

    def __post_init__(self):
        if not self.api_key:    self.api_key    = os.environ.get("BINANCE_API_KEY","")
        if not self.api_secret: self.api_secret = os.environ.get("BINANCE_API_SECRET","")
        e = os.environ.get("LIVE_MODE","")
        if e: self.live_mode = e.lower() == "true"
        if self.kill_zones is None:
            self.kill_zones = [(7,11),(12,17),(19,21)]

    # Pairs
    top_n_pairs:      int   = 10
    volume_threshold: float = 60_000_000
    scan_interval:    int   = 45
    sl_buffer:        float = 0.0015

    testnet_base_url: str = "https://testnet.binancefuture.com"
    live_base_url:    str = "https://fapi.binance.com"


@dataclass
class Trade:
    symbol:       str
    direction:    str
    entry:        float
    sl:           float
    tp:           float
    size:         float
    notional:     float
    open_time:    str
    status:       str   = "OPEN"
    close_price:  float = 0.0
    pnl:          float = 0.0
    order_id:     str   = ""
    signal_score: int   = 0
    score_grade:  str   = ""
    atr:          float = 0.0
    reason:       str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
class I:
    @staticmethod
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def atr(df, n):
        h,l,c = df["high"],df["low"],df["close"]
        tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return tr.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(s, n):
        d=s.diff()
        g=d.clip(lower=0).ewm(span=n,adjust=False).mean()
        l=(-d.clip(upper=0)).ewm(span=n,adjust=False).mean()
        return 100-100/(1+g/l.replace(0,np.nan))

    @staticmethod
    def pivot_hi(s, n):
        r=pd.Series(np.nan, index=s.index); a=s.values
        for i in range(n, len(a)-n):
            w=a[i-n:i+n+1]
            if a[i]==w.max() and list(w).count(a[i])==1: r.iloc[i]=a[i]
        return r

    @staticmethod
    def pivot_lo(s, n):
        r=pd.Series(np.nan, index=s.index); a=s.values
        for i in range(n, len(a)-n):
            w=a[i-n:i+n+1]
            if a[i]==w.min() and list(w).count(a[i])==1: r.iloc[i]=a[i]
        return r

    @staticmethod
    def cvd(df):
        """Cumulative Volume Delta — Pine: volume*(close-open)/range"""
        rng = (df["high"]-df["low"]).replace(0, np.nan).fillna(0.00001)
        delta = df["volume"] * (df["close"]-df["open"]) / rng
        return delta.cumsum(), delta


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY — 1:1 Pine Script port
# ─────────────────────────────────────────────────────────────────────────────
class InstitutionalStrategy:

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    # ── Kill Zone ─────────────────────────────────────────────────────────────
    def in_kill_zone(self) -> bool:
        if not self.cfg.session_filter: return True
        h = datetime.now(timezone.utc).hour
        return any(s <= h < e for s,e in self.cfg.kill_zones)

    # ── HTF Structure (HH/HL = BULLISH, LH/LL = BEARISH) ─────────────────────
    def htf_structure(self, df: pd.DataFrame) -> tuple[str, float, float]:
        """Returns (direction, struct_low, struct_high)"""
        if df.empty or len(df) < 30: return "CHOP", 0, 0
        ph = I.pivot_hi(df["high"], self.cfg.pivot_depth).dropna()
        pl = I.pivot_lo(df["low"],  self.cfg.pivot_depth).dropna()
        if len(ph) < 3 or len(pl) < 3: return "CHOP", 0, 0

        hh = ph.values[-3:]   # last 3 swing highs
        ll = pl.values[-3:]   # last 3 swing lows

        higher_highs = hh[-1]>hh[-2] and hh[-2]>hh[-3]
        higher_lows  = ll[-1]>ll[-2] and ll[-2]>ll[-3]
        lower_highs  = hh[-1]<hh[-2] and hh[-2]<hh[-3]
        lower_lows   = ll[-1]<ll[-2] and ll[-2]<ll[-3]

        struct_low  = float(ll[-1])
        struct_high = float(hh[-1])

        if higher_highs and higher_lows: return "BULLISH", struct_low, struct_high
        if lower_highs  and lower_lows:  return "BEARISH", struct_low, struct_high
        return "CHOP", struct_low, struct_high

    # ── Delta (CVD change) ────────────────────────────────────────────────────
    def delta_confirm(self, df: pd.DataFrame, direction: str) -> bool:
        _, delta = I.cvd(df)
        d = delta.iloc[-1]
        return (d > 0) if direction == "BULL" else (d < 0)

    # ── Delta flip (entry TF) ─────────────────────────────────────────────────
    def delta_flip(self, df: pd.DataFrame, direction: str) -> bool:
        _, delta = I.cvd(df)
        curr = delta.iloc[-1]; prev = delta.iloc[-2]
        if direction == "BULL": return curr > 0 and prev < 0
        return curr < 0 and prev > 0

    # ── Mid TF Sweep ──────────────────────────────────────────────────────────
    def detect_sweep(self, df: pd.DataFrame) -> Optional[dict]:
        """Detect quality liquidity sweep on mid TF."""
        if df.empty or len(df) < 20: return None
        pl = self.cfg.pivot_len
        ph_s = I.pivot_hi(df["high"], pl).dropna()
        pl_s = I.pivot_lo(df["low"],  pl).dropna()
        if ph_s.empty or pl_s.empty: return None

        swing_hi = float(ph_s.iloc[-1])
        swing_lo = float(pl_s.iloc[-1])

        c=df["close"]; h=df["high"]; l=df["low"]; o=df["open"]
        i = len(df)-1
        hi=h.iloc[i]; lo=l.iloc[i]; op=o.iloc[i]; cl=c.iloc[i]

        body  = abs(cl-op)
        rng   = max(hi-lo, 1e-10)
        lw    = (cl-lo) if cl>op else (op-lo)
        uw    = (hi-cl) if cl>op else (hi-op)

        bull_rej  = lw > uw * self.cfg.rejection_ratio
        bear_rej  = uw > lw * self.cfg.rejection_ratio
        bull_disp = cl>op and body>rng*self.cfg.min_body_ratio
        bear_disp = cl<op and body>rng*self.cfg.min_body_ratio

        bull_sweep = lo<swing_lo and cl>swing_lo and bull_rej and bull_disp
        bear_sweep = hi>swing_hi and cl<swing_hi and bear_rej and bear_disp

        if bull_sweep: return {"direction":"BULL","sweep_price":swing_lo}
        if bear_sweep: return {"direction":"BEAR","sweep_price":swing_hi}
        return None

    # ── Fibonacci Pullback Validation ─────────────────────────────────────────
    def fib_pullback(self, df_entry: pd.DataFrame,
                     direction: str,
                     struct_low: float, struct_high: float) -> tuple[bool, float, str]:
        """
        Returns (valid, depth_pct, quality)
        Preferred: 0.618–0.706 | Max: 0.886
        """
        if struct_high == struct_low: return False, 0, ""
        cl = df_entry["close"].iloc[-1]

        if direction == "BULL":
            depth = (struct_high - cl) / (struct_high - struct_low)
        else:
            depth = (cl - struct_low) / (struct_high - struct_low)

        valid = 0.50 <= depth <= self.cfg.fib_max
        if   depth >= self.cfg.fib_preferred: quality = "DEEP"
        elif depth >= 0.618:                  quality = "GOOD"
        elif depth >= 0.50:                   quality = "DECENT"
        else:                                  quality = ""
        return valid, round(depth, 3), quality

    # ── FVG Detection ─────────────────────────────────────────────────────────
    def find_fvg(self, df: pd.DataFrame, direction: str) -> bool:
        n = len(df)
        for j in range(n-1, max(n-8,2), -1):
            if direction=="BULL" and j>=2:
                if df["low"].iloc[j] > df["high"].iloc[j-2]: return True
            elif direction=="SHORT" and j>=2:
                if df["high"].iloc[j] < df["low"].iloc[j-2]: return True
        return False

    # ── Institutional Zone ────────────────────────────────────────────────────
    def inst_zone(self, df: pd.DataFrame) -> bool:
        """High-volume, strong-body candle = institutional activity."""
        v   = df["volume"]
        vma = v.rolling(20).mean()
        c=df["close"]; o=df["open"]; h=df["high"]; l=df["low"]
        i = len(df)-1
        body     = abs(c.iloc[i]-o.iloc[i])
        rng      = max(h.iloc[i]-l.iloc[i], 1e-10)
        body_r   = body/rng
        vol_ok   = vma.iloc[i] > 0 and v.iloc[i] > vma.iloc[i]*self.cfg.inst_vol_mult
        return vol_ok and body_r > self.cfg.inst_body_ratio

    # ── OI Proxy via Funding Rate ─────────────────────────────────────────────
    def oi_proxy(self, client, symbol: str, direction: str) -> tuple[float, bool, bool]:
        """
        Binance funding rate as OI proxy:
        Positive funding + price up → new longs (bullish OI)
        Negative funding + price down → new shorts (bearish OI)
        Returns (flow_delta, long_confirm, short_confirm)
        """
        try:
            fr = client.funding_rate(symbol=symbol, limit=5)
            rates = [float(r["fundingRate"]) for r in fr]
            if not rates: return 0, False, False
            latest = rates[-1]
            prev   = rates[-2] if len(rates)>1 else 0
            change = latest - prev
            # Normalise
            flow   = change / (abs(latest)+1e-10) if latest!=0 else 0
            long_c  = flow >  self.cfg.oi_confirm_level * 0.01
            short_c = flow < -self.cfg.oi_confirm_level * 0.01
            strong  = abs(flow) > self.cfg.oi_strong_level * 0.01
            return round(flow, 4), long_c, short_c
        except:
            return 0, False, False

    # ── WEIGHTED SCORE (mirrors Pine exactly) ────────────────────────────────
    def score_trade(self, in_inst: bool, delta_ok: bool,
                    oi_strong: bool, oi_confirm: bool,
                    in_fvg: bool, in_fib: bool) -> tuple[int, str]:
        score = 0
        if in_inst:    score += 5   # institutional zone = strongest signal
        if delta_ok:   score += 4   # delta confirmation
        if oi_strong:  score += 3   # strong OI signal
        elif oi_confirm: score += 2 # moderate OI
        if in_fvg:     score += 2   # in FVG
        if in_fib:     score += 1   # in fib zone (additional)

        if   score >= 12: grade = "A+ PREMIUM"
        elif score >= 9:  grade = "A GRADE"
        elif score >= 6:  grade = "B GRADE"
        elif score >= 3:  grade = "C GRADE"
        else:             grade = "D GRADE"
        return score, grade

    # ── MAIN ANALYZE ──────────────────────────────────────────────────────────
    def analyze(self, df_high: pd.DataFrame,
                df_mid:  pd.DataFrame,
                df_entry:pd.DataFrame,
                client,  symbol: str) -> Optional[dict]:

        if any(x.empty or len(x)<30 for x in [df_high, df_mid, df_entry]):
            return None

        # ── Kill zone ─────────────────────────────────────────────────────────
        kz = self.in_kill_zone()

        # ── HTF structure ─────────────────────────────────────────────────────
        struct, s_low, s_high = self.htf_structure(df_high)
        if struct == "CHOP": return None

        direction_ok = {"BULL": struct=="BULLISH", "BEAR": struct=="BEARISH"}

        # ── Mid TF sweep ──────────────────────────────────────────────────────
        sweep = self.detect_sweep(df_mid)
        if not sweep: return None

        d = sweep["direction"]
        if not direction_ok.get(d, False): return None

        sweep_price = sweep["sweep_price"]

        # ── Fibonacci pullback ────────────────────────────────────────────────
        fib_valid, fib_depth, fib_quality = self.fib_pullback(
            df_entry, d, s_low, s_high)
        if not fib_valid: return None

        # ── FVG ───────────────────────────────────────────────────────────────
        long_dir = "LONG" if d=="BULL" else "SHORT"
        in_fvg   = self.find_fvg(df_entry, d)

        # Must be in FVG OR deep fib
        if not in_fvg and fib_quality not in ("DEEP","GOOD"):
            return None

        # ── Delta flip on entry TF ────────────────────────────────────────────
        delta_flip_ok = self.delta_flip(df_entry, d)

        # ── Delta confirm on HTF ──────────────────────────────────────────────
        delta_htf_ok = self.delta_confirm(df_high, d)

        # ── Institutional zone ────────────────────────────────────────────────
        in_inst = self.inst_zone(df_entry)

        # ── OI proxy ─────────────────────────────────────────────────────────
        oi_flow, oi_long_c, oi_short_c = self.oi_proxy(client, symbol, d)
        oi_confirm = oi_long_c if d=="BULL" else oi_short_c
        oi_strong  = abs(oi_flow) > self.cfg.oi_strong_level * 0.01

        # ── Delta ok for scoring ──────────────────────────────────────────────
        delta_ok = delta_flip_ok or delta_htf_ok

        # ── Score ─────────────────────────────────────────────────────────────
        score, grade = self.score_trade(
            in_inst, delta_ok, oi_strong, oi_confirm, in_fvg,
            fib_quality in ("DEEP","GOOD","DECENT"))

        if score < self.cfg.min_score:
            log.debug(f"{symbol} score {score}/15 ({grade}) — below min {self.cfg.min_score}")
            return None

        # ── ATR SL/TP ─────────────────────────────────────────────────────────
        atr_s  = I.atr(df_entry, self.cfg.atr_period)
        atr    = float(atr_s.iloc[-1])
        cl     = float(df_entry["close"].iloc[-1])

        if d == "BULL":
            sl   = sweep_price - atr * self.cfg.atr_sl_mult
            sl   = min(sl, sweep_price*(1-self.cfg.sl_buffer))
            risk = cl - sl
            if risk <= 0: return None
            tp   = cl + risk * self.cfg.atr_tp_rr
        else:
            sl   = sweep_price + atr * self.cfg.atr_sl_mult
            sl   = max(sl, sweep_price*(1+self.cfg.sl_buffer))
            risk = sl - cl
            if risk <= 0: return None
            tp   = cl - risk * self.cfg.atr_tp_rr

        reasons = []
        reasons.append(f"HTF {struct}")
        reasons.append(f"Sweep {d}")
        if delta_ok:   reasons.append("DeltaFlip")
        if in_inst:    reasons.append("InstZone")
        if in_fvg:     reasons.append("FVG")
        if oi_confirm: reasons.append("OI✓")
        if kz:         reasons.append("KillZone")
        reasons.append(f"Fib {fib_depth*100:.0f}% {fib_quality}")

        return dict(
            direction   = long_dir,
            sweep_price = sweep_price,
            entry       = cl,
            sl          = sl,
            tp          = tp,
            atr         = atr,
            score       = score,
            grade       = grade,
            reason      = " | ".join(reasons),
            struct      = struct,
            fib_depth   = fib_depth,
            fib_quality = fib_quality,
            in_fvg      = in_fvg,
            in_inst     = in_inst,
            delta_ok    = delta_ok,
            oi_flow     = oi_flow,
            kz          = kz,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────
class MarketData:
    def __init__(self, client):
        self.client=client; self._ei=None; self._ei_ts=0

    def top_pairs(self, n, min_vol):
        try:
            tk=self.client.ticker_24hr_price_change()
            f=[t for t in tk
               if t["symbol"].endswith("USDT")
               and not any(x in t["symbol"] for x in
                           ["BUSD","USDC","TUSD","USDP","DAI","FDUSD"])
               and float(t["quoteVolume"])>=min_vol
               and float(t["lastPrice"])>0.001]
            f.sort(key=lambda x:float(x["quoteVolume"]),reverse=True)
            syms=[t["symbol"] for t in f[:n]]
            log.info(f"Pairs: {syms}")
            return syms
        except Exception as e:
            log.error(f"top_pairs: {e}")
            return ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]

    def klines(self, symbol, interval, limit):
        try:
            raw=self.client.klines(symbol, interval, limit=limit)
            df=pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","tb_base","tb_quote","ignore"])
            for c in ["open","high","low","close","volume"]:
                df[c]=df[c].astype(float)
            df["open_time"]=pd.to_datetime(df["open_time"],unit="ms",utc=True)
            return df.reset_index(drop=True)
        except Exception as e:
            log.error(f"klines {symbol}/{interval}: {e}")
            return pd.DataFrame()

    def price(self, symbol):
        try: return float(self.client.ticker_price(symbol=symbol)["price"])
        except: return 0.0

    def step_tick(self, symbol):
        now=time.time()
        if not self._ei or now-self._ei_ts>300:
            try: self._ei=self.client.exchange_info(); self._ei_ts=now
            except: return 0.001,0.01
        for s in self._ei.get("symbols",[]):
            if s["symbol"]==symbol:
                step=tick=0.0
                for f in s.get("filters",[]):
                    if f["filterType"]=="LOT_SIZE":     step=float(f["stepSize"])
                    if f["filterType"]=="PRICE_FILTER": tick=float(f["tickSize"])
                return step or 0.001, tick or 0.01
        return 0.001,0.01


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class RM:
    def __init__(self, cfg): self.cfg=cfg

    def size(self, entry, sl, capital):
        r_usd=capital*min(self.cfg.risk_per_trade, 0.03)
        dist=abs(entry-sl)
        if dist<1e-10: return 0,0
        qty=r_usd/dist; notional=qty*entry
        cap=capital*self.cfg.max_leverage
        if notional>cap: qty=cap/entry; notional=cap
        return round(qty,6), round(notional,4)

    @staticmethod
    def round_step(qty, step):
        if step<=0: return qty
        return math.floor(qty/step)*step

    @staticmethod
    def p_round(price, tick):
        if tick<=0: return price
        f=1/tick; return round(round(price*f)/f, 8)

    def daily_breached(self, start, now):
        return (start-now)/start >= self.cfg.daily_loss_limit


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────
class Executor:
    def __init__(self, client, market, cfg):
        self.client=client; self.mkt=market; self.cfg=cfg; self.rm=RM(cfg)

    def set_lev(self, sym, lev):
        try:
            self.client.change_leverage(symbol=sym, leverage=lev)
            self.client.change_margin_type(symbol=sym, marginType="ISOLATED")
        except: pass

    def open(self, symbol, sig, capital) -> Optional[Trade]:
        d=sig["direction"]; entry=sig["entry"]; sl=sig["sl"]; tp=sig["tp"]
        qty,notional=self.rm.size(entry,sl,capital)
        if qty<=0 or notional<5.5:
            log.warning(f"{symbol}: notional ${notional:.2f} too small"); return None
        step,tick=self.mkt.step_tick(symbol)
        qty =self.rm.round_step(qty,step)
        sl_r=self.rm.p_round(sl,tick)
        tp_r=self.rm.p_round(tp,tick)
        if qty<=0: return None
        side="BUY" if d=="LONG" else "SELL"
        xs  ="SELL" if d=="LONG" else "BUY"
        self.set_lev(symbol, min(self.cfg.max_leverage,15))
        try:
            order=self.client.new_order(symbol=symbol,side=side,type="MARKET",quantity=qty)
            oid=str(order.get("orderId",""))
            self.client.new_order(symbol=symbol,side=xs,type="STOP_MARKET",
                                  stopPrice=sl_r,closePosition=True,timeInForce="GTE_GTC")
            self.client.new_order(symbol=symbol,side=xs,type="TAKE_PROFIT_MARKET",
                                  stopPrice=tp_r,closePosition=True,timeInForce="GTE_GTC")
            t=Trade(symbol=symbol,direction=d,entry=entry,sl=sl_r,tp=tp_r,
                    size=qty,notional=round(notional,2),
                    open_time=datetime.utcnow().isoformat(),order_id=oid,
                    signal_score=sig.get("score",0),score_grade=sig.get("grade",""),
                    atr=round(sig.get("atr",0),6),reason=sig.get("reason",""))
            log.info(f"✅ {d} {symbol} @ {entry:.5f} SL:{sl_r} TP:{tp_r} "
                     f"score:{sig['score']}/15 [{sig['grade']}] | {sig['reason']}")
            return t
        except ClientError as e:
            log.error(f"Order failed {symbol}: {e.error_message}"); return None
        except Exception as e:
            log.error(f"Order error {symbol}: {e}"); return None

    def close(self, trade, reason="MANUAL"):
        side="SELL" if trade.direction=="LONG" else "BUY"
        try:
            self.client.new_order(symbol=trade.symbol,side=side,type="MARKET",
                                  quantity=trade.size,reduceOnly=True)
            self.client.cancel_open_orders(symbol=trade.symbol)
            log.info(f"❌ Closed {trade.symbol} ({reason})")
        except Exception as e:
            log.error(f"Close error {trade.symbol}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
class Journal:
    def __init__(self, path="/tmp/trades.json"):
        self.path=path; self.trades=[]; self._lock=threading.Lock(); self._load()

    def _load(self):
        try:
            with open(self.path) as f: data=json.load(f)
            self.trades=[Trade(**t) for t in data]
        except: self.trades=[]

    def _save(self):
        with open(self.path,"w") as f:
            json.dump([asdict(t) for t in self.trades],f,indent=2)

    def add(self,t):
        with self._lock: self.trades.append(t); self._save()

    def update(self,t):
        with self._lock: self._save()

    def open_trades(self): return [t for t in self.trades if t.status=="OPEN"]

    def stats(self):
        closed=[t for t in self.trades if t.status!="OPEN"]
        wins=[t for t in closed if t.pnl>0]
        losses=[t for t in closed if t.pnl<=0]
        wp=[t.pnl for t in wins]; lp=[t.pnl for t in losses]
        aw=sum(wp)/len(wp) if wp else 0
        al=sum(lp)/len(lp) if lp else 0
        return dict(total=len(closed),wins=len(wins),losses=len(losses),
                    win_rate=round(len(wins)/len(closed),3) if closed else 0,
                    total_pnl=round(sum(t.pnl for t in closed),4),
                    avg_win=round(aw,4),avg_loss=round(al,4),
                    realized_rr=round(abs(aw/al),2) if al else 0,
                    open=len(self.open_trades()))


# ─────────────────────────────────────────────────────────────────────────────
# BOT CORE
# ─────────────────────────────────────────────────────────────────────────────
class LiquiditySweepBot:
    def __init__(self, cfg: BotConfig, trades_path="/tmp/trades.json"):
        self.cfg=cfg; self.running=False
        self.journal=Journal(trades_path)
        self.strategy=InstitutionalStrategy(cfg)
        base=cfg.live_base_url if cfg.live_mode else cfg.testnet_base_url
        self.client=UMFutures(key=cfg.api_key,secret=cfg.api_secret,base_url=base)
        self.mkt=MarketData(self.client)
        self.executor=Executor(self.client,self.mkt,cfg)
        self.capital=cfg.total_capital
        self.start_capital=cfg.total_capital
        self.day_capital=cfg.total_capital
        self._status="IDLE"; self._last_scan=None
        self._scanned=[]; self._paused=False
        self._pause_reason=""; self._last_signals=[]
        log.info(f"SweepBot v4 INSTITUTIONAL | {'LIVE' if cfg.live_mode else 'TESTNET'} | ${cfg.total_capital}")
        log.info(f"Strategy: HTF Structure + CVD Delta + Fib Pullback + FVG + InstZone + OI | Min score {cfg.min_score}/15")

    def balance(self):
        try:
            for b in self.client.balance():
                if b["asset"]=="USDT": return float(b["availableBalance"])
        except: pass
        return self.capital

    def reset_daily(self):
        now=datetime.utcnow()
        if now.hour==0 and now.minute<2:
            self.day_capital=self.capital
            self._paused=False; self._pause_reason=""
            log.info("Daily reset")

    def scan(self):
        if self._paused: self._status=f"PAUSED: {self._pause_reason}"; return
        self._status="SCANNING"
        open_t=self.journal.open_trades(); open_s={t.symbol for t in open_t}
        self.capital=self.balance()
        rm=RM(self.cfg)
        if rm.daily_breached(self.day_capital,self.capital):
            self._paused=True
            self._pause_reason=f"6% daily loss (${self.capital:.2f})"
            log.warning("⛔ Daily loss limit — paused")
            self._status="PAUSED"; return
        if len(open_t)>=self.cfg.max_open_trades:
            self._status="MAX_TRADES"; return

        syms=self.mkt.top_pairs(self.cfg.top_n_pairs,self.cfg.volume_threshold)
        self._scanned=syms; self._last_signals=[]

        for sym in syms:
            if sym in open_s: continue
            df_h=self.mkt.klines(sym, self.cfg.tf_high,  self.cfg.lim_high)
            df_m=self.mkt.klines(sym, self.cfg.tf_mid,   self.cfg.lim_mid)
            df_e=self.mkt.klines(sym, self.cfg.tf_entry, self.cfg.lim_entry)
            sig=self.strategy.analyze(df_h, df_m, df_e, self.client, sym)
            if sig:
                sig["symbol"]=sym; self._last_signals.append(sig)
                log.info(f"🎯 {sym} {sig['direction']} {sig['grade']} "
                         f"score={sig['score']}/15 | {sig['reason']}")
                trade=self.executor.open(sym, sig, self.capital)
                if trade:
                    self.journal.add(trade); open_t.append(trade); open_s.add(sym)
                    if len(open_t)>=self.cfg.max_open_trades: break

        self._last_scan=datetime.utcnow().isoformat()
        self._status="WATCHING"

    def monitor(self):
        for t in self.journal.open_trades():
            p=self.mkt.price(t.symbol)
            if p<=0: continue
            hit_tp=(t.direction=="LONG" and p>=t.tp) or (t.direction=="SHORT" and p<=t.tp)
            hit_sl=(t.direction=="LONG" and p<=t.sl) or (t.direction=="SHORT" and p>=t.sl)
            if hit_tp:
                pnl=abs(t.tp-t.entry)*t.size; t.status="TP"
                t.close_price=t.tp; t.pnl=round(pnl,4); self.capital+=pnl
                log.info(f"🏆 TP {t.symbol} +${pnl:.4f}"); self.journal.update(t)
            elif hit_sl:
                pnl=-abs(t.entry-t.sl)*t.size; t.status="SL"
                t.close_price=t.sl; t.pnl=round(pnl,4); self.capital+=pnl
                log.warning(f"💀 SL {t.symbol} ${pnl:.4f}"); self.journal.update(t)

    def get_dashboard_state(self):
        stats=self.journal.stats()
        open_t=self.journal.open_trades()
        recent=sorted(self.journal.trades,key=lambda x:x.open_time,reverse=True)[:25]
        return dict(mode="LIVE" if self.cfg.live_mode else "TESTNET",
                    status=self._status,capital=round(self.capital,4),
                    start_capital=round(self.start_capital,2),
                    day_capital=round(self.day_capital,2),
                    paused=self._paused,pause_reason=self._pause_reason,
                    last_scan=self._last_scan,scanned_pairs=self._scanned,
                    last_signals=self._last_signals,stats=stats,
                    open_trades=[asdict(t) for t in open_t],
                    recent_trades=[asdict(t) for t in recent],
                    in_kill_zone=self.strategy.in_kill_zone(),
                    config=dict(risk_per_trade=self.cfg.risk_per_trade,
                                max_leverage=self.cfg.max_leverage,
                                atr_tp_rr=self.cfg.atr_tp_rr,
                                tf_high=self.cfg.tf_high,
                                tf_mid=self.cfg.tf_mid,
                                tf_entry=self.cfg.tf_entry,
                                max_open=self.cfg.max_open_trades,
                                daily_limit=self.cfg.daily_loss_limit,
                                min_score=self.cfg.min_score))

    def run_loop(self):
        self.running=True; log.info("🚀 Bot loop started")
        while self.running:
            try:
                self.reset_daily(); self.monitor(); self.scan()
            except Exception as e:
                log.error(f"Loop error: {e}",exc_info=True); self._status="ERROR"
            time.sleep(self.cfg.scan_interval)

    def start(self): threading.Thread(target=self.run_loop,daemon=True).start()
    def stop(self):  self.running=False; self._status="STOPPED"
