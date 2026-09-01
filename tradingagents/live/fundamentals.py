"""What the company reported, in numbers, with the arithmetic left visible.

The rest of this desk reasons about price. Price is a fast, honest and
completely uninformed signal: it knows that a name went up 40% in a quarter and
knows nothing about whether revenue grew, whether the growth was bought with
debt, or whether the earnings multiple already contains the next two years. A
report that ranks on momentum and never prints a financial statement is asking
the reader to take the trend on faith.

So this module pulls the statements — keyless, from the same Yahoo endpoint the
prices come from — and prints them as a table rather than a verdict. Three
deliberate choices:

* **Quarters and years, both.** A single TTM number cannot distinguish a
  business that is accelerating from one that is rolling over, and the
  year-over-year comparison is the only one that survives seasonality.
* **Ratios are shown with their inputs.** "P/E 37" is a number to argue with
  only when the EPS it divides by is on the same page.
* **A missing number stays missing.** Pre-revenue biotech has no P/E, no margin
  and no PEG, and a screen full of them is exactly what a Nasdaq momentum scan
  returns. Printing 0.00 for those turns "unknowable" into "bad", which is a
  different claim.

The caveat that belongs on every one of these numbers: they are a vendor's
transcription of a filing, not the filing. They are restated, they lag, and the
TTM window is Yahoo's, not the company's. :mod:`~.research` prints the EDGAR
link next to them for exactly this reason.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# Statements change four times a year. A day of staleness costs nothing and
# saves a network round trip per symbol on every re-run of the report.
CACHE_TTL_HOURS = 20.0

# Yahoo's rating scale, which is inverted relative to how a human reads it.
_RATING = {1: "强烈买入", 2: "买入", 3: "持有", 4: "减持", 5: "卖出"}

# Income-statement rows, in the order they are printed. Yahoo's labels.
_ROWS = [("Total Revenue", "营收"), ("Gross Profit", "毛利"),
         ("Operating Income", "营业利润"), ("Net Income", "净利润"),
         ("Diluted EPS", "摊薄EPS")]


def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


def cache_path() -> Path:
    return _home() / "fundamentals.json"


def _num(v, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def ok(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def money(v) -> str:
    """A dollar figure at the scale a human reads it in, or an em dash."""
    x = _num(v)
    if math.isnan(x):
        return "—"
    sign = "-" if x < 0 else ""
    a = abs(x)
    for cut, unit in ((1e12, "万亿"), (1e8, "亿"), (1e4, "万")):
        if a >= cut:
            return f"{sign}${a / cut:,.2f}{unit}"
    return f"{sign}${a:,.0f}"


def pct(v, digits: int = 1) -> str:
    x = _num(v)
    return "—" if math.isnan(x) else f"{x * 100:+.{digits}f}%"


def ratio(v, digits: int = 1) -> str:
    x = _num(v)
    return "—" if math.isnan(x) else f"{x:,.{digits}f}"


@dataclass
class Quarter:
    """One reported period. ``label`` is the period end, as the vendor dated it."""

    label: str = ""
    revenue: float = float("nan")
    gross: float = float("nan")
    operating: float = float("nan")
    net: float = float("nan")
    eps: float = float("nan")

    def margin(self, which: str) -> float:
        top = {"gross": self.gross, "operating": self.operating, "net": self.net}[which]
        if not ok(top) or not ok(self.revenue) or self.revenue == 0:
            return float("nan")
        return top / self.revenue


@dataclass
class Surprise:
    """One earnings date and what it did against the estimate."""

    when: str = ""
    estimate: float = float("nan")
    actual: float = float("nan")
    surprise_pct: float = float("nan")

    @property
    def reported(self) -> bool:
        return ok(self.actual)


@dataclass
class Fundamentals:
    """One symbol's financial state. Every field optional; absence is a fact."""

    symbol: str
    fetched_at: float = 0.0
    name: str = ""
    sector: str = ""
    industry: str = ""
    website: str = ""

    # scale and valuation
    market_cap: float = float("nan")
    pe_trailing: float = float("nan")
    pe_forward: float = float("nan")
    ps: float = float("nan")
    ev_ebitda: float = float("nan")
    peg: float = float("nan")
    eps_trailing: float = float("nan")
    eps_forward: float = float("nan")
    dividend_yield: float = float("nan")
    beta: float = float("nan")

    # growth and quality
    revenue_ttm: float = float("nan")
    revenue_growth: float = float("nan")
    earnings_growth: float = float("nan")
    gross_margin: float = float("nan")
    operating_margin: float = float("nan")
    profit_margin: float = float("nan")
    roe: float = float("nan")

    # balance sheet
    debt_to_equity: float = float("nan")
    current_ratio: float = float("nan")
    total_cash: float = float("nan")
    total_debt: float = float("nan")
    free_cashflow: float = float("nan")

    # the street
    target_mean: float = float("nan")
    target_high: float = float("nan")
    target_low: float = float("nan")
    rating_mean: float = float("nan")
    analysts: float = float("nan")
    short_pct_float: float = float("nan")
    held_institutions: float = float("nan")

    quarters: list = field(default_factory=list)     # list[Quarter], newest first
    years: list = field(default_factory=list)        # list[Quarter], newest first
    surprises: list = field(default_factory=list)    # list[Surprise], newest first
    error: str = ""

    # --- derived ---------------------------------------------------------
    @property
    def profitable(self) -> bool:
        return ok(self.eps_trailing) and self.eps_trailing > 0

    @property
    def rating_text(self) -> str:
        if not ok(self.rating_mean):
            return ""
        return _RATING.get(int(round(self.rating_mean)), "")

    def upside(self, price: float) -> float:
        """The street's mean target against a price, as a fraction."""
        p, t = _num(price), _num(self.target_mean)
        if math.isnan(p) or p <= 0 or math.isnan(t):
            return float("nan")
        return t / p - 1.0

    def yoy(self, which: str = "revenue") -> float:
        """Latest quarter against the same quarter a year earlier.

        Five quarters are needed for one comparison, which Yahoo usually but not
        always returns. Sequential growth is deliberately not substituted: for a
        seasonal business it is a different and much weaker statement.
        """
        qs = self.quarters
        if len(qs) < 5:
            return float("nan")
        now, then = getattr(qs[0], which, float("nan")), getattr(qs[4], which, float("nan"))
        if not ok(now) or not ok(then) or then == 0:
            return float("nan")
        return now / abs(then) - 1.0

    def beat_rate(self) -> tuple[int, int]:
        """(beats, reported) over the surprises carried."""
        done = [s for s in self.surprises if s.reported and ok(s.surprise_pct)]
        return sum(1 for s in done if s.surprise_pct > 0), len(done)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def _statement(frame, limit: int) -> list[Quarter]:
    """A yfinance income-statement frame → newest-first periods."""
    out: list[Quarter] = []
    if frame is None:
        return out
    try:
        if frame.empty:
            return out
        columns = list(frame.columns)[:limit]
    except Exception:
        return out
    for col in columns:
        q = Quarter(label=str(col)[:10])
        for row, attr in (("Total Revenue", "revenue"), ("Gross Profit", "gross"),
                          ("Operating Income", "operating"), ("Net Income", "net"),
                          ("Diluted EPS", "eps")):
            try:
                if row in frame.index:
                    setattr(q, attr, _num(frame.loc[row, col]))
            except Exception:
                continue
        out.append(q)
    return out


def _surprises(frame, limit: int) -> list[Surprise]:
    out: list[Surprise] = []
    if frame is None:
        return out
    try:
        if frame.empty:
            return out
        rows = list(frame.iterrows())[:limit]
    except Exception:
        return out
    for when, row in rows:
        try:
            out.append(Surprise(
                when=str(when)[:10],
                estimate=_num(row.get("EPS Estimate")),
                actual=_num(row.get("Reported EPS")),
                surprise_pct=_num(row.get("Surprise(%)")) / 100.0
                if ok(row.get("Surprise(%)")) else float("nan"),
            ))
        except Exception:
            continue
    return out


def _yield(info: dict) -> float:
    """A dividend yield as a fraction, from the field whose units are not ambiguous.

    ``dividendYield`` is unusable on its own: Yahoo ships 0.34 for Apple's
    0.34% and has historically shipped 0.0034 for the same number, and no
    magnitude test separates "0.05 = 5%" from "0.05 = 0.05%". Guessing there
    puts a factor of 100 into a printed figure.

    ``trailingAnnualDividendYield`` is documented as a fraction and agrees with
    the rate ÷ price on every symbol checked, so it is preferred. Falling back
    to ``dividendYield`` only when it is unmistakably a percentage (above 1)
    keeps a yield on the page for the symbols missing the better field, and
    returns NaN rather than a coin flip for the rest.
    """
    v = _num((info or {}).get("trailingAnnualDividendYield"))
    if not math.isnan(v):
        return v
    raw = _num((info or {}).get("dividendYield"))
    if math.isnan(raw) or raw <= 0:
        return float("nan")
    return raw / 100.0 if raw > 1 else float("nan")


def fetch(symbol: str, *, quarters: int = 6, years: int = 4,
          log=logger.debug) -> Fundamentals:
    """One symbol, live. Never raises; a failure comes back in ``error``."""
    f = Fundamentals(symbol=symbol.upper(), fetched_at=time.time())
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = {}
        try:
            info = t.info or {}
        except Exception as exc:                       # the profile is optional
            log(f"{symbol}: no info ({exc})")

        g = info.get
        f.name = str(g("longName") or g("shortName") or "")
        f.sector = str(g("sector") or "")
        f.industry = str(g("industry") or "")
        f.website = str(g("website") or "")

        f.market_cap = _num(g("marketCap"))
        f.pe_trailing = _num(g("trailingPE"))
        f.pe_forward = _num(g("forwardPE"))
        f.ps = _num(g("priceToSalesTrailing12Months"))
        f.ev_ebitda = _num(g("enterpriseToEbitda"))
        f.peg = _num(g("pegRatio") or g("trailingPegRatio"))
        f.eps_trailing = _num(g("trailingEps"))
        f.eps_forward = _num(g("forwardEps"))
        f.dividend_yield = _yield(info)
        f.beta = _num(g("beta"))

        f.revenue_ttm = _num(g("totalRevenue"))
        f.revenue_growth = _num(g("revenueGrowth"))
        f.earnings_growth = _num(g("earningsGrowth") or g("earningsQuarterlyGrowth"))
        f.gross_margin = _num(g("grossMargins"))
        f.operating_margin = _num(g("operatingMargins"))
        f.profit_margin = _num(g("profitMargins"))
        f.roe = _num(g("returnOnEquity"))

        f.debt_to_equity = _num(g("debtToEquity"))
        f.current_ratio = _num(g("currentRatio"))
        f.total_cash = _num(g("totalCash"))
        f.total_debt = _num(g("totalDebt"))
        f.free_cashflow = _num(g("freeCashflow"))

        f.target_mean = _num(g("targetMeanPrice"))
        f.target_high = _num(g("targetHighPrice"))
        f.target_low = _num(g("targetLowPrice"))
        f.rating_mean = _num(g("recommendationMean"))
        f.analysts = _num(g("numberOfAnalystOpinions"))
        f.short_pct_float = _num(g("shortPercentOfFloat"))
        f.held_institutions = _num(g("heldPercentInstitutions"))

        for attr, getter, limit in (("quarters", "quarterly_income_stmt", quarters),
                                    ("years", "income_stmt", years)):
            try:
                setattr(f, attr, _statement(getattr(t, getter), limit))
            except Exception as exc:
                log(f"{symbol}: no {getter} ({exc})")
        try:
            f.surprises = _surprises(t.get_earnings_dates(limit=12), 8)
        except Exception as exc:
            log(f"{symbol}: no earnings dates ({exc})")

        if not any((f.quarters, f.years, ok(f.market_cap))):
            f.error = "该代码没有可用的财务数据（可能是 ETF、ADR 或新上市）"
    except Exception as exc:
        f.error = f"{type(exc).__name__}: {exc}"
    return f


def _rehydrate(raw: dict) -> Fundamentals:
    f = Fundamentals(symbol=str(raw.get("symbol", "")))
    for k, v in raw.items():
        if k in ("quarters", "years", "surprises"):
            continue
        if hasattr(f, k):
            setattr(f, k, v)
    f.quarters = [Quarter(**q) for q in raw.get("quarters") or []]
    f.years = [Quarter(**q) for q in raw.get("years") or []]
    f.surprises = [Surprise(**s) for s in raw.get("surprises") or []]
    return f


class FundamentalsBook:
    """Disk-cached fundamentals, one file for the whole desk.

    Same shape as :class:`~.earnings.EarningsBook` on purpose: both wrap a slow
    per-symbol vendor call behind a TTL, and two different caching idioms for
    the same problem is one more thing to get wrong at 4am.
    """

    def __init__(self, path: Path | None = None, *, ttl_hours: float = CACHE_TTL_HOURS,
                 fetcher=fetch):
        self.path = path or cache_path()
        self.ttl = ttl_hours * 3600.0
        self._fetch = fetcher
        self._cache: dict[str, Fundamentals] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("fundamentals cache unreadable (%s); starting empty", exc)
            return
        if not isinstance(raw, dict):
            return
        for sym, entry in raw.items():
            try:
                self._cache[str(sym).upper()] = _rehydrate(entry)
            except Exception:
                continue

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in self._cache.items()},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as exc:
            # A cache that cannot be written is a slow desk, not a broken one.
            logger.warning("could not write the fundamentals cache: %s", exc)

    def fresh(self, symbol: str) -> bool:
        got = self._cache.get(symbol.upper())
        return bool(got) and (time.time() - _num(got.fetched_at, 0.0)) < self.ttl

    def get(self, symbols: list[str], *, refresh: bool = False,
            log=logger.debug) -> dict[str, Fundamentals]:
        wanted = [s.upper() for s in symbols if s]
        stale = [s for s in wanted if refresh or not self.fresh(s)]
        for sym in stale:
            log(f"fundamentals: fetching {sym}")
            self._cache[sym] = self._fetch(sym, log=log)
        if stale:
            self._save()
        return {s: self._cache[s] for s in wanted if s in self._cache}


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------

def valuation_read(f: Fundamentals) -> str:
    if not f.profitable:
        if ok(f.ps) and ok(f.revenue_ttm) and f.revenue_ttm > 0:
            return (f"仍未盈利（TTM EPS {ratio(f.eps_trailing, 2)}），市盈率没有意义；"
                    f"只能看市销率 {ratio(f.ps)} 倍。未盈利公司的估值锚在于现金能烧多久，"
                    f"不在倍数。")
        return "既无盈利也无有意义的营收，估值倍数一栏本页留空——这不是低估，是无法用倍数衡量。"
    bits = [f"TTM 市盈率 {ratio(f.pe_trailing)} 倍"]
    if ok(f.pe_forward):
        if ok(f.pe_trailing) and f.pe_trailing > 0:
            gap = f.pe_forward / f.pe_trailing - 1
            trend = "市场预期未来一年盈利上升" if gap < -0.05 else (
                "市场预期未来一年盈利下滑" if gap > 0.05 else "市场预期盈利基本持平")
            bits.append(f"前瞻市盈率 {ratio(f.pe_forward)} 倍（{trend}）")
        else:
            bits.append(f"前瞻市盈率 {ratio(f.pe_forward)} 倍")
    if ok(f.ps):
        bits.append(f"市销率 {ratio(f.ps)} 倍")
    if ok(f.ev_ebitda):
        bits.append(f"EV/EBITDA {ratio(f.ev_ebitda)} 倍")
    if ok(f.peg):
        judged = "低于 1，增长尚未被价格吃掉" if f.peg < 1 else (
            "在 1–2 之间，价格与增长大致匹配" if f.peg <= 2 else "高于 2，价格已经计入了不少增长")
        bits.append(f"PEG {ratio(f.peg, 2)}（{judged}）")
    return "；".join(bits) + "。"


def growth_read(f: Fundamentals) -> str:
    bits = []
    yoy = f.yoy("revenue")
    if ok(yoy):
        bits.append(f"最新季度营收同比 {pct(yoy)}")
    elif ok(f.revenue_growth):
        bits.append(f"营收同比 {pct(f.revenue_growth)}（厂商口径）")
    if ok(f.earnings_growth):
        bits.append(f"盈利同比 {pct(f.earnings_growth)}")
    qs = [q for q in f.quarters if ok(q.revenue)]
    if len(qs) >= 3:
        # Three points is the minimum that can distinguish a trend from a step.
        a, b, c = qs[0].revenue, qs[1].revenue, qs[2].revenue
        if a > b > c:
            bits.append("最近三个季度营收连续环比上升")
        elif a < b < c:
            bits.append("最近三个季度营收连续环比下滑——这是趋势买入最该警惕的组合")
    beats, done = f.beat_rate()
    if done:
        bits.append(f"最近 {done} 次财报里 {beats} 次超预期")
    return "；".join(bits) + "。" if bits else "没有拿到可比的增长数据。"


def quality_read(f: Fundamentals) -> str:
    bits = []
    for label, value in (("毛利率", f.gross_margin), ("营业利润率", f.operating_margin),
                         ("净利率", f.profit_margin)):
        if ok(value):
            bits.append(f"{label} {pct(value, 1).lstrip('+')}")
    if ok(f.roe):
        bits.append(f"ROE {pct(f.roe, 1).lstrip('+')}")
    if not bits:
        return "没有拿到利润率数据。"
    out = "、".join(bits) + "。"
    if ok(f.operating_margin) and f.operating_margin < 0:
        out += " 营业利润为负，意味着主营业务本身还在消耗现金。"
    return out


def balance_read(f: Fundamentals) -> str:
    bits = []
    if ok(f.total_cash):
        bits.append(f"现金 {money(f.total_cash)}")
    if ok(f.total_debt):
        bits.append(f"有息负债 {money(f.total_debt)}")
    if ok(f.debt_to_equity):
        level = ("负债很轻" if f.debt_to_equity < 50 else
                 "负债适中" if f.debt_to_equity < 150 else "负债偏重")
        bits.append(f"负债/权益 {ratio(f.debt_to_equity)}%（{level}）")
    if ok(f.current_ratio):
        bits.append(f"流动比率 {ratio(f.current_ratio, 2)}"
                    + ("（短期偿付紧）" if f.current_ratio < 1 else ""))
    if ok(f.free_cashflow):
        sign = "为正" if f.free_cashflow > 0 else "为负，公司在净烧钱"
        bits.append(f"自由现金流 {money(f.free_cashflow)}，{sign}")
    return "；".join(bits) + "。" if bits else "没有拿到资产负债数据。"


def street_read(f: Fundamentals, price: float = float("nan")) -> str:
    if not ok(f.analysts) or f.analysts <= 0:
        return "没有覆盖这只股票的分析师数据。"
    bits = [f"{int(f.analysts)} 位分析师覆盖"]
    if f.rating_text:
        bits.append(f"平均评级「{f.rating_text}」（{ratio(f.rating_mean, 2)}/5，1 为最看多）")
    if ok(f.target_mean):
        up = f.upside(price)
        span = (f"，区间 {ratio(f.target_low, 2)}–{ratio(f.target_high, 2)}"
                if ok(f.target_low) and ok(f.target_high) else "")
        bits.append(f"目标价均值 {ratio(f.target_mean, 2)}"
                    + (f"（较现价 {pct(up)}）" if ok(up) else "") + span)
    tail = ("。分析师目标价是共识而不是预测，且随价格一起移动——"
            "它更适合用来看分歧有多大（区间宽度），而不是用来当目标位。")
    return "；".join(bits) + tail


def read(f: Fundamentals, price: float = float("nan")) -> list[tuple[str, str]]:
    """``(标题, 段落)`` pairs. Empty when the symbol has no statements at all."""
    if f.error and not (f.quarters or f.years or ok(f.market_cap)):
        return [("财报", f"读不到财务数据：{f.error}")]
    return [("估值", valuation_read(f)),
            ("增长", growth_read(f)),
            ("盈利质量", quality_read(f)),
            ("资产负债", balance_read(f)),
            ("卖方预期", street_read(f, price))]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _period_table(periods: list, title: str, unit: str) -> list[str]:
    rows = [p for p in periods if ok(p.revenue) or ok(p.eps)]
    if not rows:
        return []
    head = "| 指标 | " + " | ".join(p.label for p in rows) + " |"
    rule = "|---|" + "---:|" * len(rows)
    out = [f"**{title}**（{unit}）", "", head, rule]
    out.append("| 营收 | " + " | ".join(money(p.revenue) for p in rows) + " |")
    out.append("| 毛利率 | " + " | ".join(
        pct(p.margin("gross")).lstrip("+") for p in rows) + " |")
    out.append("| 营业利润率 | " + " | ".join(
        pct(p.margin("operating")).lstrip("+") for p in rows) + " |")
    out.append("| 净利率 | " + " | ".join(
        pct(p.margin("net")).lstrip("+") for p in rows) + " |")
    out.append("| 净利润 | " + " | ".join(money(p.net) for p in rows) + " |")
    out.append("| 摊薄EPS | " + " | ".join(ratio(p.eps, 2) for p in rows) + " |")
    return out + [""]


def _surprise_table(surprises: list) -> list[str]:
    rows = [s for s in surprises if ok(s.estimate) or ok(s.actual)][:6]
    if not rows:
        return []
    out = ["**财报兑现记录**", "",
           "| 财报日 | 预期EPS | 实际EPS | 超预期 |", "|---|---:|---:|---:|"]
    for s in rows:
        mark = "尚未公布" if not s.reported else pct(s.surprise_pct)
        out.append(f"| {s.when} | {ratio(s.estimate, 2)} | "
                   f"{ratio(s.actual, 2) if s.reported else '—'} | {mark} |")
    return out + [""]


def markdown_block(f: Fundamentals, price: float = float("nan"),
                   heading: str = "### 财报与基本面") -> str:
    """The whole financial section for one symbol's page."""
    out = [heading, ""]
    if f.error and not (f.quarters or f.years or ok(f.market_cap)):
        return "\n".join(out + [f"_读不到财务数据：{f.error}_", ""])

    facts = []
    if ok(f.market_cap):
        facts.append(f"市值 {money(f.market_cap)}")
    if f.sector:
        facts.append(f"行业 {f.sector}" + (f" / {f.industry}" if f.industry else ""))
    if ok(f.revenue_ttm):
        facts.append(f"TTM 营收 {money(f.revenue_ttm)}")
    if ok(f.eps_trailing):
        facts.append(f"TTM EPS {ratio(f.eps_trailing, 2)}")
    if ok(f.dividend_yield) and f.dividend_yield > 0:
        facts.append(f"股息率 {pct(f.dividend_yield, 2).lstrip('+')}")
    if ok(f.beta):
        facts.append(f"Beta {ratio(f.beta, 2)}")
    if ok(f.short_pct_float) and f.short_pct_float > 0:
        facts.append(f"空头占流通股 {pct(f.short_pct_float, 1).lstrip('+')}")
    if facts:
        out += [" · ".join(facts), ""]

    for title, text in read(f, price):
        out.append(f"- **{title}**：{text}")
    out.append("")
    out += _period_table(f.quarters[:5], "分季度损益", "季度数据，最新在左")
    out += _period_table(f.years[:4], "分年度损益", "年度数据，最新在左")
    out += _surprise_table(f.surprises)
    out += ["_财务数字来自 Yahoo Finance 对申报文件的转录：会被重述、会滞后，"
            "TTM 窗口是厂商口径而非公司财年口径。以上任何一个数字如果会改变你的决定，"
            "请点开本页下方的 SEC EDGAR 链接看原文。_", ""]
    return "\n".join(out)
