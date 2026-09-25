"""Full-history technical structure: trend, volume, volatility, levels, shape.

The verified snapshot answers "where is price today". This module answers the
questions a trader actually asks before acting: how did it get here, on what
volume, is the trend accelerating or decaying, where are the levels that
matter, and how does it look next to the index. Everything is computed from
the full 5-year OHLCV history the loader already caches — no extra fetch.

Output is text, because the consumer is an analyst reading a brief, and the
two ASCII charts exist so the *shape* of the last year and last quarter can be
seen rather than inferred from a table of numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _load(ticker: str, date: str) -> pd.DataFrame:
    df = load_ohlcv(ticker, date)
    if df is None or df.empty:
        raise ValueError(f"no OHLCV for {ticker}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] <= pd.to_datetime(date)]
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"]).reset_index(drop=True)


def _pct(a: float, b: float) -> float:
    return (a / b - 1.0) if b else float("nan")


def _fmt_pct(x: float, plus: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.1%}" if plus else f"{x:.1%}"


def _slope_pct(series: pd.Series, lookback: int) -> float:
    """Percent change of a series over ``lookback`` bars (trend direction)."""
    if len(series.dropna()) <= lookback:
        return float("nan")
    s = series.dropna()
    return _pct(float(s.iloc[-1]), float(s.iloc[-1 - lookback]))


def _ret(c: pd.Series, n: int) -> float:
    return _pct(float(c.iloc[-1]), float(c.iloc[-1 - n])) if len(c) > n else float("nan")


def _pivots(df: pd.DataFrame, span: int = 5, window: int = 120):
    """Swing highs/lows: a bar whose high/low is the extreme of ±span bars."""
    d = df.tail(window).reset_index(drop=True)
    highs, lows = [], []
    for i in range(span, len(d) - span):
        seg = d.iloc[i - span:i + span + 1]
        if d.loc[i, "High"] >= seg["High"].max():
            highs.append((d.loc[i, "Date"], float(d.loc[i, "High"])))
        if d.loc[i, "Low"] <= seg["Low"].min():
            lows.append((d.loc[i, "Date"], float(d.loc[i, "Low"])))
    return highs, lows


def _percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float((s < value).mean())


# ----------------------------------------------------------------------------
# ASCII chart
# ----------------------------------------------------------------------------

def ascii_chart(df: pd.DataFrame, bars: int, height: int = 12, vol_height: int = 4,
                title: str = "", overlay: pd.Series | None = None, overlay_label: str = "") -> str:
    """Close line over ``bars`` sessions, optional MA overlay, volume bars below.

    One column per bar, so the chart is exactly ``bars`` wide. '*' is close,
    'o' is the overlay (e.g. SMA50) where it differs from close, '#' is volume.
    """
    d = df.tail(bars).reset_index(drop=True)
    if len(d) < 2:
        return "(not enough data to chart)"
    c = d["Close"].to_numpy()
    lo, hi = float(c.min()), float(c.max())
    ov = overlay.tail(bars).to_numpy() if overlay is not None else None
    if ov is not None:
        ovv = ov[~np.isnan(ov)]
        if ovv.size:
            lo, hi = min(lo, float(ovv.min())), max(hi, float(ovv.max()))
    rng = (hi - lo) or 1.0

    grid = [[" "] * len(d) for _ in range(height)]
    def row(v):  # price -> row index (0 = top)
        return height - 1 - int(round((v - lo) / rng * (height - 1)))
    if ov is not None:
        for i, v in enumerate(ov):
            if not np.isnan(v):
                grid[row(v)][i] = "o"
    for i, v in enumerate(c):
        grid[row(v)][i] = "*"

    v = d["Volume"].fillna(0).to_numpy()
    vmax = float(v.max()) or 1.0
    vgrid = [[" "] * len(d) for _ in range(vol_height)]
    for i, x in enumerate(v):
        h = int(round(x / vmax * vol_height))
        up = i > 0 and c[i] >= c[i - 1]
        ch = "#" if up else "="   # '#' up-day volume, '=' down-day volume
        for r in range(vol_height - h, vol_height):
            vgrid[r][i] = ch

    out = []
    if title:
        out.append(title)
    for r, line in enumerate(grid):
        price = hi - (r / (height - 1)) * rng
        out.append(f"{price:>10.2f} |{''.join(line)}")
    out.append(f"{'':>10} +{'-' * len(d)}")
    for line in vgrid:
        out.append(f"{'':>10} |{''.join(line)}")
    first, last = d["Date"].iloc[0].date(), d["Date"].iloc[-1].date()
    legend = "* close" + (f"  o {overlay_label}" if overlay_label else "") + "  # up-vol  = down-vol"
    out.append(f"{'':>10}  {first}  ...  {last}    {legend}")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# main report
# ----------------------------------------------------------------------------

def technical_structure(ticker: str, date: str, benchmark: str = "SPY") -> str:
    df = _load(ticker, date)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    px = float(c.iloc[-1])
    out: list[str] = []

    # ---- 1. multi-horizon returns & relative strength -----------------------
    try:
        bdf = _load(benchmark, date) if benchmark != ticker else None
    except Exception:
        bdf = None
    horizons = [("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]
    out.append("### Returns and relative strength")
    out.append("")
    hdr = f"| Horizon | {ticker} | {benchmark} | RS (diff) |"
    out += [hdr, "|---|---:|---:|---:|"]
    for label, k in horizons:
        r = _ret(c, k)
        b = _ret(bdf["Close"], k) if bdf is not None and len(bdf) > k else float("nan")
        rs = r - b if not (np.isnan(r) or np.isnan(b)) else float("nan")
        out.append(f"| {label} | {_fmt_pct(r)} | {_fmt_pct(b)} | {_fmt_pct(rs)} |")

    # ---- 2. trend structure ------------------------------------------------
    out += ["", "### Trend structure (moving averages)", ""]
    mas = {}
    for w in (20, 50, 100, 200):
        if n >= w:
            mas[w] = c.rolling(w).mean()
    out += ["| MA | Value | Price vs MA | MA slope (20d) |", "|---|---:|---:|---:|"]
    for w, s in mas.items():
        val = float(s.iloc[-1])
        out.append(f"| SMA{w} | {val:,.2f} | {_fmt_pct(_pct(px, val))} | {_fmt_pct(_slope_pct(s, 20))} |")
    if len(mas) == 4:
        vals = [float(mas[w].iloc[-1]) for w in (20, 50, 100, 200)]
        if vals == sorted(vals, reverse=True):
            stack = "BULLISH — fully aligned (20 > 50 > 100 > 200)"
        elif vals == sorted(vals):
            stack = "BEARISH — fully inverted (20 < 50 < 100 < 200)"
        else:
            stack = "MIXED — averages crossing; trend in transition"
        above = sum(px > x for x in vals)
        out += ["", f"- MA stack: **{stack}**; price above {above}/4 averages"]
    # higher-highs / higher-lows from swing points
    sw_h, sw_l = _pivots(df, span=5, window=120)
    if len(sw_h) >= 2 and len(sw_l) >= 2:
        hh = sw_h[-1][1] > sw_h[-2][1]
        hl = sw_l[-1][1] > sw_l[-2][1]
        seq = {(True, True): "higher highs + higher lows (uptrend intact)",
               (False, False): "lower highs + lower lows (downtrend)",
               (True, False): "higher high but lower low (widening / unstable)",
               (False, True): "lower high but higher low (compressing triangle)"}[(hh, hl)]
        out.append(f"- Swing structure (last ~6 months): **{seq}**")
        out.append(f"  - last two swing highs: {sw_h[-2][1]:,.2f} ({sw_h[-2][0].date()}) → "
                   f"{sw_h[-1][1]:,.2f} ({sw_h[-1][0].date()})")
        out.append(f"  - last two swing lows:  {sw_l[-2][1]:,.2f} ({sw_l[-2][0].date()}) → "
                   f"{sw_l[-1][1]:,.2f} ({sw_l[-1][0].date()})")

    # ---- 3. volume -----------------------------------------------------------
    out += ["", "### Volume and accumulation / distribution", ""]
    v20, v50 = float(v.tail(20).mean()), float(v.tail(50).mean())
    v200 = float(v.tail(200).mean()) if n >= 200 else float("nan")
    out.append(f"- Avg volume: 20d {v20:,.0f} · 50d {v50:,.0f} · 200d {v200:,.0f}  "
               f"→ 20d vs 50d {_fmt_pct(_pct(v20, v50))}, 20d vs 200d {_fmt_pct(_pct(v20, v200))}")
    chg = c.pct_change()
    for win in (20, 50):
        seg = df.tail(win)
        upv = float(seg.loc[chg.tail(win) > 0, "Volume"].mean() or 0)
        dnv = float(seg.loc[chg.tail(win) < 0, "Volume"].mean() or 0)
        ratio = upv / dnv if dnv else float("nan")
        tag = "accumulation" if ratio > 1.15 else "distribution" if ratio < 0.87 else "neutral"
        out.append(f"- Up-day vs down-day avg volume, last {win}: {upv:,.0f} vs {dnv:,.0f} "
                   f"→ ratio {ratio:.2f} (**{tag}**)")
    # OBV
    obv = (np.sign(chg.fillna(0)) * v).cumsum()
    out.append(f"- OBV slope: 20d {_fmt_pct(_slope_pct(obv - obv.min() + 1, 20))} · "
               f"50d {_fmt_pct(_slope_pct(obv - obv.min() + 1, 50))}  "
               f"(price 20d {_fmt_pct(_ret(c, 20))} / 50d {_fmt_pct(_ret(c, 50))})")
    # distribution / accumulation days (IBD-style), last 25 sessions
    seg = df.tail(26).reset_index(drop=True)
    dist = acc = 0
    for i in range(1, len(seg)):
        r = _pct(seg.loc[i, "Close"], seg.loc[i - 1, "Close"])
        higher_vol = seg.loc[i, "Volume"] > seg.loc[i - 1, "Volume"]
        if r <= -0.002 and higher_vol:
            dist += 1
        elif r >= 0.002 and higher_vol:
            acc += 1
    out.append(f"- Last 25 sessions: **{dist} distribution days**, {acc} accumulation days "
               f"(down/up ≥0.2% on rising volume)")
    # biggest-volume sessions in last 60 — climax / capitulation detection
    big = df.tail(60).copy()
    big["chg"] = big["Close"].pct_change()
    big = big.sort_values("Volume", ascending=False).head(4)
    out.append("- Highest-volume sessions (last 60): " + "; ".join(
        f"{r.Date.date()} {r.Volume/1e6:,.1f}M ({_fmt_pct(r.chg)})" for r in big.itertuples()))

    # ---- 4. volatility regime -----------------------------------------------
    out += ["", "### Volatility regime", ""]
    lr = np.log(c / c.shift(1))
    rv20 = float(lr.tail(20).std() * np.sqrt(TRADING_DAYS))
    rv_hist = lr.rolling(20).std() * np.sqrt(TRADING_DAYS)
    out.append(f"- 20d realized vol: **{rv20:.0%}** annualized — "
               f"{_percentile(rv_hist.tail(252), rv20):.0%} percentile of the last year")
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_now, atr_3m = float(atr14.iloc[-1]), float(atr14.tail(63).mean())
    out.append(f"- ATR(14): {atr_now:,.2f} ({atr_now/px:.2%} of price); vs 3-month avg ATR "
               f"{_fmt_pct(_pct(atr_now, atr_3m))} → "
               f"{'EXPANDING' if atr_now > atr_3m * 1.1 else 'contracting' if atr_now < atr_3m * 0.9 else 'stable'}")
    sma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    bw = (4 * sd20 / sma20)
    bw_now = float(bw.iloc[-1])
    out.append(f"- Bollinger bandwidth: {bw_now:.1%} — {_percentile(bw.tail(252), bw_now):.0%} "
               f"percentile (low = squeeze, high = extended move)")

    # ---- 5. price structure / levels ----------------------------------------
    out += ["", "### Price structure and key levels", ""]
    yr = df.tail(252)
    hi52, lo52 = float(yr["High"].max()), float(yr["Low"].min())
    hi_date = yr.loc[yr["High"].idxmax(), "Date"].date()
    out.append(f"- 52w high {hi52:,.2f} ({hi_date}) → price is **{_fmt_pct(_pct(px, hi52))}** off high; "
               f"52w low {lo52:,.2f} → {_fmt_pct(_pct(px, lo52))} above low; "
               f"range position {((px-lo52)/(hi52-lo52) if hi52>lo52 else 0):.0%}")
    res = sorted({round(p, 2) for _, p in sw_h if p > px})[:3]
    sup = sorted({round(p, 2) for _, p in sw_l if p < px}, reverse=True)[:3]
    if res:
        out.append("- Resistance (swing highs above): " + ", ".join(f"{x:,.2f} ({_fmt_pct(_pct(x, px))})" for x in res))
    if sup:
        out.append("- Support (swing lows below):     " + ", ".join(f"{x:,.2f} ({_fmt_pct(_pct(x, px))})" for x in sup))
    # unfilled gaps (last 60 sessions)
    gaps = []
    seg = df.tail(61).reset_index(drop=True)
    for i in range(1, len(seg)):
        if seg.loc[i, "Low"] > seg.loc[i - 1, "High"]:          # gap up
            top, bot = seg.loc[i, "Low"], seg.loc[i - 1, "High"]
            if (seg.loc[i:, "Low"] > bot).all():
                gaps.append(f"UP gap {bot:,.2f}–{top:,.2f} ({seg.loc[i,'Date'].date()}) unfilled")
        elif seg.loc[i, "High"] < seg.loc[i - 1, "Low"]:        # gap down
            top, bot = seg.loc[i - 1, "Low"], seg.loc[i, "High"]
            if (seg.loc[i:, "High"] < top).all():
                gaps.append(f"DOWN gap {bot:,.2f}–{top:,.2f} ({seg.loc[i,'Date'].date()}) unfilled")
    if gaps:
        out.append("- Unfilled gaps (last 60): " + "; ".join(gaps[-4:]))
    # drawdown stats
    roll_max = c.tail(252).cummax()
    dd = (c.tail(252) / roll_max - 1)
    out.append(f"- Max drawdown (1y): {dd.min():.1%}; current drawdown from 1y peak: {float(dd.iloc[-1]):.1%}")

    # ---- 6. momentum oscillators trend --------------------------------------
    out += ["", "### Momentum", ""]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; sig = macd.ewm(span=9, adjust=False).mean(); hist = macd - sig
    out.append(f"- RSI(14): {float(rsi.iloc[-1]):.1f} (5 sessions ago {float(rsi.iloc[-6]):.1f}) → "
               f"{'rising' if rsi.iloc[-1] > rsi.iloc[-6] else 'falling'}")
    out.append(f"- MACD hist: {float(hist.iloc[-1]):+.2f} (5 ago {float(hist.iloc[-6]):+.2f}) → "
               f"{'momentum building' if abs(hist.iloc[-1]) > abs(hist.iloc[-6]) and np.sign(hist.iloc[-1])==np.sign(hist.iloc[-6]) else 'momentum fading / turning'}")
    # divergence check: price new 20d high but RSI lower than at prior high (and vice versa)
    last20 = c.tail(20)
    if px >= float(last20.max()) and float(rsi.iloc[-1]) < float(rsi.tail(20).max()) - 5:
        out.append("- ⚠ Bearish divergence: price at 20d high while RSI is lower than its recent peak")
    if px <= float(last20.min()) and float(rsi.iloc[-1]) > float(rsi.tail(20).min()) + 5:
        out.append("- ✓ Bullish divergence: price at 20d low while RSI holds above its recent trough")

    # ---- 7. charts ------------------------------------------------------------
    out += ["", "### Chart — last 12 months (weekly closes)", "", "```"]
    wk = df.set_index("Date").resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"]).reset_index()
    wk_ma = wk["Close"].rolling(10).mean()   # ~50-day equivalent
    out.append(ascii_chart(wk, bars=min(52, len(wk)), overlay=wk_ma, overlay_label="10wk MA"))
    out += ["```", "", "### Chart — last 3 months (daily closes)", "", "```"]
    out.append(ascii_chart(df, bars=min(63, n), overlay=mas.get(50), overlay_label="SMA50"))
    out.append("```")

    return "\n".join(out)
