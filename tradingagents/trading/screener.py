"""Universe screener: scan every listed common stock, rank by evidence, keep the top N.

The watchlist was hand-picked from a dozen names. This widens the funnel to the
whole exchange and lets an algorithm do the first cut, so human (or Claude)
judgment is spent on fifty candidates that already clear a quantitative bar
instead of on whichever tickers happened to be in the news.

Pipeline, cheapest step first:

1. **Universe** — the exchange's official symbol directory (nasdaqtrader.com,
   free, no key). ETFs, test issues, warrants, units, rights, preferreds,
   notes, closed-end funds and SPAC shells are dropped by flag and by name.
2. **Prices** — one year of daily OHLCV for every survivor via batched
   yfinance downloads (~250 symbols per request, threaded). Cached per
   exchange per date so re-runs and re-ranks are free.
3. **Hard filters** — price, dollar liquidity, history length, realized
   volatility, and trend (above the 200-day). These are not scores; a name
   that fails any one is out regardless of how it ranks elsewhere.
4. **Composite score** — percentile ranks of momentum (12-1 and 6-month),
   relative strength vs the index, volume accumulation, OBV, 50-day slope,
   proximity to the 52-week high, distribution-day count, and volatility,
   weighted and averaged. A parabolic-extension penalty trims names more than
   50% above their 200-day.

What this is: a momentum + accumulation + liquidity tilt — the factor family
with the most robust out-of-sample evidence. What it is not: a prediction.
The output is a shortlist for analysis, not a buy list.

    python -m tradingagents.trading.desk screen --top 50
    python -m tradingagents.trading.desk screen --exchange all --top 100
"""

from __future__ import annotations

import io
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tradingagents.dataflows.config import get_config

_NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Security-name patterns that mark non-common-stock instruments. Matched
# case-insensitively. Kept explicit rather than clever so a false positive is
# easy to diagnose by reading the list.
_EXCLUDE_NAME = re.compile(
    r"warrant|\bunits?\b|\brights?\b|acquisition corp|acquisition co\b|"
    r"blank check|\bspac\b|preferred|preference|notes due|debenture|"
    r"\bnotes\b|\bbond\b|% |\bfund\b|\btrust units\b|\betn\b|"
    r"depositary shares?,? each representing .*(?:preferred|interest in)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# 1. universe
# ----------------------------------------------------------------------------

def _read_pipe_table(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if lines and lines[-1].lower().startswith("file creation time"):
        lines = lines[:-1]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)


def fetch_universe(exchange: str = "nasdaq") -> pd.DataFrame:
    """Return DataFrame[symbol, name, exchange] of listed common stocks.

    ``exchange``: "nasdaq" (default) or "all" (Nasdaq + NYSE + NYSE American).
    """
    frames = []
    nq = _read_pipe_table(_NASDAQ_URL)
    nq = nq[(nq["ETF"] == "N") & (nq["Test Issue"] == "N")]
    frames.append(pd.DataFrame({
        "symbol": nq["Symbol"], "name": nq["Security Name"], "exchange": "NASDAQ",
    }))
    if exchange == "all":
        ot = _read_pipe_table(_OTHER_URL)
        ot = ot[(ot["ETF"] == "N") & (ot["Test Issue"] == "N")]
        ot = ot[ot["Exchange"].isin(["N", "A"])]  # NYSE, NYSE American
        frames.append(pd.DataFrame({
            "symbol": ot["ACT Symbol"], "name": ot["Security Name"],
            "exchange": ot["Exchange"].map({"N": "NYSE", "A": "NYSE-AMER"}),
        }))
    u = pd.concat(frames, ignore_index=True).dropna(subset=["symbol"])
    u["symbol"] = u["symbol"].str.strip()
    # Preferred / special-class markers on non-Nasdaq feeds use '$' and '.'.
    u = u[~u["symbol"].str.contains(r"\$", regex=True)]
    u["symbol"] = u["symbol"].str.replace(".", "-", regex=False)
    # Nasdaq 5-letter symbols carry an instrument suffix (W/U/R/P...). Real
    # five-letter common stocks exist but are rare; the trade-off favours
    # removing the far larger warrant/unit population.
    is_nasdaq = u["exchange"] == "NASDAQ"
    u = u[~(is_nasdaq & (u["symbol"].str.len() == 5))]
    u = u[~u["name"].fillna("").str.contains(_EXCLUDE_NAME)]
    return u.drop_duplicates("symbol").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. prices
# ----------------------------------------------------------------------------

def _cache_dir() -> Path:
    p = Path(get_config()["data_cache_dir"]) / "screens"
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_panel(tickers: list[str], date: str, period: str = "1y",
                   chunk: int = 250, cache_key: str = "nasdaq", refresh: bool = False,
                   log=print) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for every ticker; returns {"Close","High","Low","Volume"}
    as wide frames (rows = dates, columns = tickers), filtered to rows <= date.

    Cached to disk per (cache_key, date). A failed symbol is simply absent.
    """
    import yfinance as yf

    cache = _cache_dir() / f"panel_{cache_key}_{date}.pkl"
    if cache.exists() and not refresh:
        panel = pd.read_pickle(cache)
        log(f"[screen] loaded cached panel: {panel['Close'].shape[1]} tickers")
        return panel

    fields = ("Close", "High", "Low", "Volume")
    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in fields}
    t0 = time.time()
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                data = yf.download(batch, period=period, interval="1d", group_by="ticker",
                                   auto_adjust=True, threads=True, progress=False)
            except Exception as exc:  # one bad batch must not kill the scan
                log(f"[screen] batch {i//chunk} failed: {exc}")
                continue
        if data is None or data.empty:
            continue
        if not isinstance(data.columns, pd.MultiIndex):   # single-ticker batch
            data = pd.concat({batch[0]: data}, axis=1)
        for f in fields:
            cols = [(t, f) for t in batch if (t, f) in data.columns]
            if cols:
                sub = data.loc[:, cols]
                sub.columns = [t for t, _ in cols]
                parts[f].append(sub)
        done = min(i + chunk, len(tickers))
        log(f"[screen] {done}/{len(tickers)} downloaded ({time.time()-t0:.0f}s)")

    panel = {}
    cutoff = pd.to_datetime(date)
    for f in fields:
        if not parts[f]:
            panel[f] = pd.DataFrame()
            continue
        wide = pd.concat(parts[f], axis=1)
        wide = wide.loc[:, ~wide.columns.duplicated()]
        wide.index = pd.to_datetime(wide.index).tz_localize(None)
        panel[f] = wide[wide.index <= cutoff].sort_index()
    pd.to_pickle(panel, cache)
    log(f"[screen] panel saved: {panel['Close'].shape[1]} tickers × {len(panel['Close'])} days")
    return panel


# ----------------------------------------------------------------------------
# 3 + 4. factors, filters, score
# ----------------------------------------------------------------------------

def _ret(C: pd.DataFrame, k: int) -> pd.Series:
    if len(C) <= k:
        return pd.Series(np.nan, index=C.columns)
    return C.iloc[-1] / C.iloc[-1 - k] - 1


def compute_factors(panel: dict[str, pd.DataFrame], benchmark: str = "SPY") -> pd.DataFrame:
    C, H, V = panel["Close"], panel["High"], panel["Volume"]
    C = C.apply(pd.to_numeric, errors="coerce")
    V = V.apply(pd.to_numeric, errors="coerce").fillna(0)
    n = C.notna().sum()
    last = C.ffill().iloc[-1]

    f = pd.DataFrame(index=C.columns)
    f["price"] = last
    f["rows"] = n
    f["ret_1m"] = _ret(C, 21)
    f["ret_3m"] = _ret(C, 63)
    f["ret_6m"] = _ret(C, 126)
    # 12-1 momentum: 12-month return skipping the most recent month, the
    # canonical academic definition (short-term reversal is excluded).
    if len(C) >= 230:
        f["mom_12_1"] = C.iloc[-22] / C.iloc[0] - 1
    else:
        f["mom_12_1"] = np.nan

    sma50 = C.rolling(50).mean()
    sma200 = C.rolling(200).mean()
    f["sma50"] = sma50.iloc[-1]
    f["sma200"] = sma200.iloc[-1]
    f["sma50_slope"] = sma50.iloc[-1] / sma50.iloc[-21] - 1 if len(C) > 70 else np.nan
    f["above_50"] = last > f["sma50"]
    f["above_200"] = last > f["sma200"]
    f["ext_200"] = last / f["sma200"] - 1
    f["off_high"] = last / H.max() - 1

    f["dollar_vol_50"] = (C * V).tail(50).mean()
    chg = C.pct_change()
    upv = V.where(chg > 0).tail(50).mean()
    dnv = V.where(chg < 0).tail(50).mean()
    f["ud_vol_50"] = upv / dnv.replace(0, np.nan)
    upv20 = V.where(chg > 0).tail(20).mean()
    dnv20 = V.where(chg < 0).tail(20).mean()
    f["ud_vol_20"] = upv20 / dnv20.replace(0, np.nan)
    obv = (np.sign(chg.fillna(0)) * V).cumsum()
    if len(obv) > 51:
        f["obv_slope_50"] = (obv.iloc[-1] - obv.iloc[-51]) / (50 * V.tail(50).mean()).replace(0, np.nan)
    else:
        f["obv_slope_50"] = np.nan

    lr = np.log(C / C.shift(1))
    f["rvol_20"] = lr.tail(20).std() * np.sqrt(252)
    hl = (H - panel["Low"]).abs()
    hc = (H - C.shift(1)).abs()
    lc = (panel["Low"] - C.shift(1)).abs()
    tr = np.maximum(np.maximum(hl, hc), lc)
    f["atr_pct_14"] = tr.tail(14).mean() / last

    seg_c = chg.tail(25)
    seg_v_up = (V.tail(25) > V.shift(1).tail(25))
    f["dist_days_25"] = ((seg_c <= -0.002) & seg_v_up).sum()
    f["acc_days_25"] = ((seg_c >= 0.002) & seg_v_up).sum()

    # relative strength vs the benchmark (downloaded as part of the panel)
    if benchmark in C.columns:
        f["rs_3m"] = f["ret_3m"] - float(_ret(C[[benchmark]], 63).iloc[0])
        f["rs_6m"] = f["ret_6m"] - float(_ret(C[[benchmark]], 126).iloc[0])
    else:
        f["rs_3m"] = f["ret_3m"]
        f["rs_6m"] = f["ret_6m"]
    return f


WEIGHTS = {
    "mom_12_1": 1.0, "ret_6m": 1.0, "rs_3m": 1.0, "ud_vol_50": 1.0,
    "obv_slope_50": 0.5, "sma50_slope": 0.5, "off_high": 0.5,
    "neg_dist_days": 0.5, "neg_rvol": 0.5, "above_50": 0.25,
}


# ----------------------------------------------------------------------------
# sector enrichment (for diversification caps)
# ----------------------------------------------------------------------------

def fetch_sectors(tickers: list[str], log=print, workers: int = 8) -> dict[str, tuple[str, str]]:
    """{ticker: (sector, industry)} via yfinance profile data.

    Sectors are stable, so the cache is a single persistent JSON that grows
    across runs — the first scan pays the lookup cost, later ones don't.
    A lookup that fails records "Unknown" rather than blocking the screen.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    cache = _cache_dir() / "sectors.json"
    known: dict[str, list] = {}
    if cache.exists():
        try:
            known = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            known = {}
    missing = [t for t in tickers if t not in known]
    if missing:
        log(f"[screen] looking up sector for {len(missing)} tickers ...")

        def _one(t):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    i = yf.Ticker(t).info or {}
                return t, [i.get("sector") or "Unknown", i.get("industry") or "Unknown"]
            except Exception:
                return t, ["Unknown", "Unknown"]

        with ThreadPoolExecutor(workers) as ex:
            for t, si in ex.map(_one, missing):
                known[t] = si
        cache.write_text(json.dumps(known, indent=0), encoding="utf-8")
    return {t: tuple(known.get(t, ["Unknown", "Unknown"])) for t in tickers}


def diversify(ranked: pd.DataFrame, top: int, max_per_sector: int) -> pd.DataFrame:
    """Greedy fill in score order with a per-sector cap.

    If the cap leaves the list short (a narrow market where one sector owns
    the leaderboard), a second pass fills the remainder uncapped and marks
    those rows so the reader knows the cap was relaxed for them.
    """
    counts: dict[str, int] = {}
    chosen, overflow = [], []
    for tkr, row in ranked.iterrows():
        sec = row.get("sector", "Unknown")
        if counts.get(sec, 0) < max_per_sector:
            counts[sec] = counts.get(sec, 0) + 1
            chosen.append(tkr)
        else:
            overflow.append(tkr)
        if len(chosen) >= top:
            break
    out = ranked.loc[chosen].copy()
    out["cap_relaxed"] = False
    if len(out) < top and overflow:
        extra = ranked.loc[overflow[: top - len(out)]].copy()
        extra["cap_relaxed"] = True
        out = pd.concat([out, extra])
    out["rank"] = range(1, len(out) + 1)
    return out


def screen(date: str, exchange: str = "nasdaq", top: int = 50,
           min_price: float = 5.0, min_dollar_vol: float = 20e6,
           max_rvol: float = 1.20, min_rvol: float = 0.08, require_above_200: bool = True,
           max_ext_200: float = 1.00, max_per_sector: int | None = 8,
           candidate_pool: int | None = None, refresh: bool = False, log=print):
    """Run the full pipeline; returns (ranked DataFrame, stats dict).

    ``max_per_sector`` caps how many names one sector may place in the final
    list (None disables). Sectors are looked up only for the candidate pool
    (default ``max(3*top, 150)`` best-scoring names), not the whole universe.
    """
    uni = fetch_universe(exchange)
    log(f"[screen] universe ({exchange}): {len(uni)} common stocks")
    tickers = uni["symbol"].tolist()
    if "SPY" not in tickers:
        tickers.append("SPY")
    panel = download_panel(tickers, date, cache_key=exchange, refresh=refresh, log=log)
    f = compute_factors(panel)
    f = f.join(uni.set_index("symbol")[["name", "exchange"]], how="left")
    f = f.drop(index="SPY", errors="ignore")

    stats = {"universe": len(uni), "priced": int(f["price"].notna().sum())}
    # Deal-pinned stubs: a takeover gap followed by near-zero volatility scores
    # beautifully on every momentum factor (at the high, no distribution days,
    # low vol) while offering nothing but the arb spread. No freely trading
    # common stock runs single-digit annualized realized vol, so a floor on
    # rvol is a clean discriminator. Found the hard way: APGE, CRNX, TECH and
    # SAFT all made an unfiltered top-50 in Aug 2026 while under agreement.
    deal_pinned = f["rvol_20"] < min_rvol
    stats["deal_pinned_excluded"] = int((deal_pinned & (f["price"] >= min_price)
                                         & (f["dollar_vol_50"] >= min_dollar_vol)).sum())
    mask = (
        (f["price"] >= min_price)
        & (f["dollar_vol_50"] >= min_dollar_vol)
        & (f["rows"] >= 200)
        & (f["rvol_20"] <= max_rvol)
        & ~deal_pinned
        & (f["ext_200"] <= max_ext_200)
    )
    if require_above_200:
        mask &= f["above_200"].fillna(False)
    g = f[mask].copy()
    stats["passed_filters"] = len(g)
    if g.empty:
        return g, stats

    # percentile ranks within the filtered set; higher = better everywhere
    r = pd.DataFrame(index=g.index)
    r["mom_12_1"] = g["mom_12_1"].rank(pct=True)
    r["ret_6m"] = g["ret_6m"].rank(pct=True)
    r["rs_3m"] = g["rs_3m"].rank(pct=True)
    r["ud_vol_50"] = g["ud_vol_50"].rank(pct=True)
    r["obv_slope_50"] = g["obv_slope_50"].rank(pct=True)
    r["sma50_slope"] = g["sma50_slope"].rank(pct=True)
    r["off_high"] = g["off_high"].rank(pct=True)          # closer to high → higher
    r["neg_dist_days"] = (-g["dist_days_25"]).rank(pct=True)
    r["neg_rvol"] = (-g["rvol_20"]).rank(pct=True)
    r["above_50"] = g["above_50"].astype(float)
    w = pd.Series(WEIGHTS)
    score = (r[w.index] * w).sum(axis=1, skipna=True) / (r[w.index].notna() * w).sum(axis=1)
    # Parabolic caution: >50% above the 200-day is where blow-offs live.
    score = score.where(g["ext_200"] <= 0.5, score * 0.85)
    g["score"] = score
    g = g.sort_values("score", ascending=False)
    g.insert(0, "rank", range(1, len(g) + 1))

    if max_per_sector is None:
        return g.head(top), stats

    pool_n = candidate_pool or max(3 * top, 150)
    pool = g.head(pool_n).copy()
    sec = fetch_sectors(pool.index.tolist(), log=log)
    pool["sector"] = [sec[t][0] for t in pool.index]
    pool["industry"] = [sec[t][1] for t in pool.index]
    stats["candidate_pool"] = len(pool)
    stats["sector_counts_in_pool"] = pool["sector"].value_counts().to_dict()
    out = diversify(pool, top=top, max_per_sector=max_per_sector)
    stats["sector_counts_in_top"] = out["sector"].value_counts().to_dict()
    return out, stats


def format_table(g: pd.DataFrame) -> str:
    if g.empty:
        return "(no names passed the filters)"
    has_sector = "sector" in g.columns
    hdr = (f"{'#':>3} {'Ticker':<7}{'Name':<22}" + (f"{'Sector':<15}" if has_sector else "")
           + f"{'Price':>9}{'$Vol(M)':>8}{'3m':>7}{'6m':>7}{'RS3m':>7}{'vs200':>7}{'OffHi':>7}"
           f"{'U/D':>6}{'OBV':>6}{'Vol':>5}{'Dst':>4}{'Score':>7}")
    lines = [hdr, "-" * len(hdr)]
    abbrev = {"Healthcare": "Health", "Technology": "Tech", "Financial Services": "Financial",
              "Consumer Cyclical": "Cons.Cycl", "Consumer Defensive": "Cons.Def",
              "Communication Services": "Comm.Svc", "Industrials": "Industrial",
              "Basic Materials": "Materials", "Real Estate": "RealEstate"}
    for tkr, row in g.iterrows():
        name = str(row.get("name", ""))[:21]
        sec = ""
        if has_sector:
            s = abbrev.get(row.get("sector", ""), str(row.get("sector", ""))[:12])
            sec = f"{(s + ('*' if row.get('cap_relaxed') else '')):<15}"
        lines.append(
            f"{int(row['rank']):>3} {tkr:<7}{name:<22}{sec}{row['price']:>9.2f}{row['dollar_vol_50']/1e6:>8.0f}"
            f"{row['ret_3m']:>+7.0%}{row['ret_6m']:>+7.0%}{row['rs_3m']:>+7.0%}{row['ext_200']:>+7.0%}"
            f"{row['off_high']:>+7.0%}{row['ud_vol_50']:>6.2f}{row['obv_slope_50']:>+6.2f}"
            f"{row['rvol_20']:>5.0%}{int(row['dist_days_25']):>4}{row['score']:>7.3f}"
        )
    if has_sector and g.get("cap_relaxed", pd.Series(dtype=bool)).any():
        lines.append("  * sector cap relaxed to fill the list")
    return "\n".join(lines)


def save_results(g: pd.DataFrame, date: str, exchange: str) -> Path:
    out = Path.home() / ".tradingagents" / "screens"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"screen_{exchange}_{date}.csv"
    g.to_csv(p)
    return p
