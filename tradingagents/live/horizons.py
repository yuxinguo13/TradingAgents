"""Three books, three clocks — so the report stops being a new list every day.

The complaint this module answers is a real defect, not a preference: a page
that prints only the names issued *this morning* looks like a fresh portfolio
every session, even when the engine underneath is carrying the same positions.
The screen's ordering also moves a little each day, so the reader sees churn
that the book does not actually contain.

Splitting by holding period fixes it, because "what should I own" and "what
should I trade" are two questions with different evidence, different position
sizes and different clocks:

* **核心长仓 (core)** — months to years. A hand-maintained list in
  ``core.json``, reviewed *monthly*, with hysteresis: a name leaves only when it
  breaks a stated long-term rule, never because it slipped in this week's
  ranking. This is the section that must look the same tomorrow as it does
  today. If it does not, it is not a core.
* **波段 (swing)** — one to four weeks. The daily advisor's own ideas, but with
  the positions already open printed *first* and the new ones only filling free
  slots. The number of slots is the cap on churn: a full book proposes nothing,
  which is the correct behaviour and previously looked like a bug.
* **日内 (day trade)** — one session. Deliberately separate, deliberately not
  sized by the same rule, and deliberately honest about its own limit: this
  desk has daily bars, and daily bars cannot produce an intraday entry. What
  they *can* produce is the levels tomorrow's session will be measured against
  — the prior day's high and low — plus the names liquid and volatile enough
  for that game to be playable at all. So this section publishes 盯盘对象与关键
  价位, never an order.

Nothing here places a trade, and the core section deliberately does not size
anything: a long-term weight is a decision about the reader's whole net worth,
which this program does not know.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from pathlib import Path

logger = logging.getLogger(__name__)

# --- core rules -------------------------------------------------------------
# All conventions, all stated. None of them are fitted.

# A core name is allowed this far below its 200-day average before the rule
# fires. Without the buffer a name oscillating around the line would be sold
# and rebought monthly, which is the opposite of a core holding.
CORE_SMA_BUFFER = 0.04
# And this much of a twelve-month loss. A core is a bet on a business; one down
# year is ordinary, and a third of the value gone is a different question.
CORE_MAX_12M_LOSS = -0.15
# How far a position may drift from its target weight before a rebalance is
# even mentioned. Below this the transaction cost is the larger number.
CORE_DRIFT_BAND = 0.25
# Reviewed on the first session of a month. Any cadence is arbitrary; what
# matters is that it is *not* daily, and that the report says which it is.
CORE_REVIEW_NOTE = "核心组合每月复核一次；两次复核之间只在破位规则触发时才动"

# --- what may be proposed as a core holding --------------------------------
# A core list is a bet on businesses, and the first draft of one must not be a
# momentum screen wearing a different heading. Ranking on the twelve-month
# return alone put a +1,975% microcap at the top of a section labelled "长期",
# which is the exact opposite of what the section is for. The four filters
# below are the difference, and the cap is the load-bearing one: a stock that
# went up twentyfold in a year is evidence of a repricing, not of durability,
# and leaving it uncapped lets one such name outrank every real business.
CORE_MIN_DOLLAR_VOL = 1e8      # a core position must be exitable in size
CORE_RET_CAP = 1.0             # credit for the year's return stops at +100%
CORE_MAX_DRAWDOWN = -0.35      # already 35% off its high is a different question
CORE_MAX_ATR_PCT = 0.05        # 5%/day is a trading vehicle, not a holding

# Two filters the price data cannot supply, and both were missing from the
# first draft this module produced.
#
# *Profitability.* The seed list came back holding a diagnostics company with a
# TTM EPS of -3.51 and a consensus that has it still losing money next year —
# while the swing engine, reading the same symbol on the same morning, refused
# it for having too little reward per unit of risk. Two halves of one report
# disagreeing about one name is the tell: a core holding is a claim about a
# business, and a business that does not earn anything cannot support one.
CORE_REQUIRE_PROFIT = True
# *Concentration.* Ranking eight names off one screen returned NVDA, AMD, ASML
# and TSM — a GPU designer, its competitor, the foundry that makes both, and
# the sole supplier of that foundry's lithography. Four slots, one supply
# chain. Equal weights across correlated names is not diversification; it is
# the same bet placed four times at 7.5% each.
CORE_MAX_PER_SECTOR = 2

# Cash held back on top of what the swing book can claim. The seeded core used
# a flat 60%, which collided with a full swing book: six slots at the 8%
# position cap is 48% of equity, and 60 + 48 = 108% of an account that does not
# have 108% to spend. The invested fraction is derived from those two numbers
# instead — see :func:`core_budget` — so the two books cannot be allocated past
# the account again.
CORE_CASH_BUFFER = 0.10
CORE_MIN_INVESTED = 0.20
CORE_MAX_INVESTED = 0.60


def core_budget(slots: int, position_cap: float,
                buffer: float = CORE_CASH_BUFFER) -> float:
    """What fraction of equity the core may hold, given the swing book's claim.

    ``slots * position_cap`` is the most the swing book can be holding at once;
    the rest, less a cash buffer, is what a long-term list may occupy. Bounded
    at both ends: a desk with a huge swing book still gets a core worth having,
    and one with no swing book does not get its whole account allocated to
    eight names it was handed rather than chose.
    """
    reserved = max(0.0, _num(slots, 0.0)) * max(0.0, _num(position_cap, 0.0))
    free = 1.0 - reserved - max(0.0, _num(buffer, 0.0))
    return round(max(CORE_MIN_INVESTED, min(CORE_MAX_INVESTED, free)), 4)

# --- swing rules ------------------------------------------------------------
DEFAULT_SWING_SLOTS = 6            # concurrent swing ideas the book may carry
SWING_HOLD_DAYS = (7, 28)          # the horizon this section claims

# --- day-trade filters ------------------------------------------------------
DT_MIN_ATR_PCT = 0.025             # under this the intraday range pays no one
DT_MIN_DOLLAR_VOL = 5e7            # a day trade needs an exit at any moment
DT_MIN_PRICE = 5.0
DT_MIN_RVOL = 1.3                  # yesterday's volume vs its 20-day average
DT_STOP_ATR = 0.5                  # intraday stops are a fraction of a swing's
DT_TARGET_ATR = 1.0


def _num(v, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


# ===========================================================================
# 核心长仓
# ===========================================================================

def core_path() -> Path:
    """Where the long-term list lives. Hand-editable, never auto-pruned."""
    env = os.getenv("TRADINGAGENTS_CORE_PATH")
    if env:
        return Path(env).expanduser()
    return _home() / "core.json"


@dataclass
class CoreHolding:
    """One long-term position the reader has decided to own."""

    symbol: str
    weight: float = 0.0            # target share of equity, as a fraction
    thesis: str = ""               # why, in the reader's own words
    added: str = ""                # ISO date
    tag: str = ""


@dataclass
class CoreLine:
    """One core name as of today: where it is, and whether a rule fired."""

    holding: CoreHolding
    price: float = float("nan")
    sma200: float = float("nan")
    ext_200: float = float("nan")
    ret_6m: float = float("nan")
    ret_12m: float = float("nan")
    off_high: float = float("nan")
    spark: str = ""
    status: str = "持有"
    action: str = "不动"
    note: str = ""
    breached: bool = False


def load_core(path: Path | None = None) -> list[CoreHolding]:
    """The core list, or empty. Never raises.

    Accepts both the terse form a human types (``{"MSFT": 0.08}``) and the full
    one this module writes. A file the reader has hand-edited into either shape
    must keep working; refusing to parse it would make the file feel dangerous
    to touch, and a core list nobody edits is not a core list.
    """
    p = path or core_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("core list unreadable (%s); treating it as empty", exc)
        return []
    if not isinstance(raw, dict):
        return []
    out: list[CoreHolding] = []
    for sym, value in raw.items():
        symbol = str(sym).strip().upper()
        if not symbol:
            continue
        if isinstance(value, dict):
            out.append(CoreHolding(
                symbol=symbol,
                weight=_num(value.get("weight"), 0.0),
                thesis=str(value.get("thesis") or ""),
                added=str(value.get("added") or ""),
                tag=str(value.get("tag") or ""),
            ))
        else:
            out.append(CoreHolding(symbol=symbol, weight=_num(value, 0.0)))
    return out


def save_core(holdings: list[CoreHolding], path: Path | None = None) -> Path:
    """Write the list atomically, in the full form, sorted by weight."""
    p = path or core_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {h.symbol: {k: v for k, v in asdict(h).items() if k != "symbol"}
            for h in sorted(holdings, key=lambda h: (-h.weight, h.symbol))}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def is_review_day(day: _date) -> bool:
    """True on the first four calendar days of a month.

    A window rather than a date: the first session of a month lands on a
    different number every month, and a report skipped on the 1st must not
    silently skip the whole quarter's review.
    """
    return 1 <= day.day <= 4


def review_core(holdings: list[CoreHolding], facts: dict, *,
                held_weights: dict | None = None,
                review_day: bool = False) -> list[CoreLine]:
    """Today's read on every core name. ``facts`` is ``{symbol: {...}}``.

    Expected keys per symbol: ``price``, ``sma200``, ``ret_6m``, ``ret_12m``,
    ``off_high``, ``spark``. Anything missing degrades that line rather than
    dropping it — a core holding whose data failed today is exactly the line a
    reader must still see.

    The rules, in the order they fire:

    1. **破位** — below the 200-day by more than the buffer, or a twelve-month
       loss past the floor. Fires on any day, because a thesis break is not
       something to notice next month.
    2. **偏离** — weight drifted outside the band. Mentioned only on a review
       day; on every other day the drift is shown and no action is proposed.
    3. Otherwise: 持有, explicitly, with nothing to do.
    """
    weights = held_weights or {}
    out: list[CoreLine] = []
    for h in holdings:
        f = facts.get(h.symbol) or {}
        line = CoreLine(
            holding=h,
            price=_num(f.get("price")),
            sma200=_num(f.get("sma200")),
            ret_6m=_num(f.get("ret_6m")),
            ret_12m=_num(f.get("ret_12m")),
            off_high=_num(f.get("off_high")),
            spark=str(f.get("spark") or ""),
        )
        if _ok(line.price) and _ok(line.sma200) and line.sma200 > 0:
            line.ext_200 = line.price / line.sma200 - 1.0

        if not _ok(line.price):
            line.status, line.action = "数据缺失", "先去核对"
            line.note = "今天没能取到价格；这一行不是「没事」，是「不知道」"
            out.append(line)
            continue

        broke_sma = _ok(line.ext_200) and line.ext_200 < -CORE_SMA_BUFFER
        broke_year = _ok(line.ret_12m) and line.ret_12m < CORE_MAX_12M_LOSS
        if broke_sma or broke_year:
            reasons = []
            if broke_sma:
                reasons.append(f"收在 200 日线下方 {abs(line.ext_200) * 100:.1f}%"
                               f"（容忍度 {CORE_SMA_BUFFER * 100:.0f}%）")
            if broke_year:
                reasons.append(f"近一年 {line.ret_12m * 100:+.1f}%，跌破 "
                               f"{CORE_MAX_12M_LOSS * 100:.0f}% 的长期底线")
            line.status, line.action, line.breached = "破位", "复核长期逻辑", True
            line.note = ("；".join(reasons) + "。规则触发，但规则不替你决定："
                         "先回答「当初买它的理由还成立吗」，再决定减仓还是剔除")
            out.append(line)
            continue

        target, actual = h.weight, _num(weights.get(h.symbol))
        if review_day and target > 0 and _ok(actual):
            drift = actual / target - 1.0 if target else float("nan")
            if _ok(drift) and abs(drift) > CORE_DRIFT_BAND:
                line.status = "偏离目标"
                line.action = "减回目标" if drift > 0 else "补回目标"
                line.note = (f"实际权重 {actual * 100:.1f}% vs 目标 {target * 100:.1f}%"
                             f"（偏离 {drift * 100:+.0f}%，超出 ±{CORE_DRIFT_BAND * 100:.0f}% 的带宽）")
                out.append(line)
                continue

        line.status, line.action = "持有", "不动"
        line.note = ("趋势与长期底线都没破；核心仓位的价值一半来自它不动"
                     if not review_day else "本月复核：规则未触发，维持原状")
        out.append(line)
    return out


def propose_core(symbols: list, facts: dict, *, count: int = 8,
                 fundamentals: dict | None = None, sectors: dict | None = None,
                 min_dollar_vol: float = CORE_MIN_DOLLAR_VOL,
                 ret_cap: float = CORE_RET_CAP,
                 max_drawdown: float = CORE_MAX_DRAWDOWN,
                 max_atr_pct: float = CORE_MAX_ATR_PCT,
                 require_profit: bool = CORE_REQUIRE_PROFIT,
                 max_per_sector: int = CORE_MAX_PER_SECTOR,
                 invested: float = 0.6) -> list[CoreHolding]:
    """A first draft of a core list, for a reader who has not written one.

    Long-horizon and deliberately dull. Price filters first — above the 200-day,
    a positive year, liquid enough to leave, not already deep in a drawdown or
    swinging 5% a day — then a blend of *capped* twelve-month return and size,
    and finally the two filters that need something price cannot tell you:
    the company earns money, and no sector takes more than
    ``max_per_sector`` slots.

    The return cap is the point of the ranking. Sorting on the raw
    twelve-month number sorts a Nasdaq universe by which microcap repriced
    hardest, which is a momentum screen with a long-term label on it; capping
    the credit at ``ret_cap`` means a good year and a spectacular one score the
    same, and liquidity breaks the tie toward the larger business.

    The sector cap is the point of the *selection*. It is applied after the
    ranking rather than inside it so the reason a name was dropped is
    reportable: "third-best in a sector already holding two" is a sentence, and
    a penalty term buried in a score is not. Note what it cannot do: the label
    is an industry classification, and the concentration that matters is a
    *theme*. Nothing here knows that a GPU designer, a foundry and a
    lithography supplier are one bet on AI capex. That judgement is the
    reader's, and it is the main reason this output is a draft.

    ``roe`` joins the ranking whenever statements are supplied. Ranking a list
    meant to be held for years on price alone, while holding the income
    statement, leaves the better evidence unused — and it showed: on capped
    return and liquidity alone the seeder preferred a 10%-ROE name to a
    117%-ROE one in the same industry, purely because the first had the larger
    twelve-month move.

    ``symbols`` is tickers in any order. ``facts`` is the same mapping
    :func:`review_core` takes. ``fundamentals`` is ``{symbol: obj}`` where the
    object exposes ``profitable``; a symbol missing from it is dropped when
    ``require_profit`` is set, because "not checked" and "checked and earning"
    must not resolve the same way in a list meant to be held for years.
    ``sectors`` is ``{symbol: label}``; anything unlabelled shares one bucket.

    Weights are equal and sum to ``invested``. The caller must present the
    result as a draft: an equal weight is the only allocation this program can
    justify, because a conviction weighting would be inventing convictions the
    reader has not stated.
    """
    kept: list[tuple[str, float, float]] = []       # symbol, capped return, size
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        f = facts.get(sym) or {}
        price, sma200 = _num(f.get("price")), _num(f.get("sma200"))
        r12 = _num(f.get("ret_12m"))
        dv = _num(f.get("dollar_vol"))
        off_high, atr_pct = _num(f.get("off_high")), _num(f.get("atr_pct"))
        if not (_ok(price) and _ok(sma200) and sma200 > 0 and price > sma200):
            continue
        if not _ok(r12) or r12 <= 0:
            continue
        if not _ok(dv) or dv < min_dollar_vol:
            continue
        if _ok(off_high) and off_high < max_drawdown:
            continue
        if _ok(atr_pct) and atr_pct > max_atr_pct:
            continue
        if require_profit and not _earns(sym, fundamentals):
            continue
        seen.add(sym)
        kept.append((sym, min(r12, ret_cap), math.log10(dv)))
    if not kept:
        return []

    def _scaled(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        span = hi - lo
        return [0.5] * len(values) if span <= 0 else [(v - lo) / span for v in values]

    rets = _scaled([r for _, r, _ in kept])
    sizes = _scaled([z for _, _, z in kept])
    quality = [_num(getattr((fundamentals or {}).get(sym), "roe", None))
               for sym, _, _ in kept]
    if any(_ok(q) for q in quality):
        # Capped so one extraordinary ROE cannot carry a name past every
        # filter, for the same reason the twelve-month return is capped.
        qs = _scaled([min(q, 0.5) if _ok(q) else 0.0 for q in quality])
        scores = [0.4 * rets[i] + 0.3 * sizes[i] + 0.3 * qs[i] for i in range(len(kept))]
    else:
        scores = [0.5 * rets[i] + 0.5 * sizes[i] for i in range(len(kept))]
    ranked = sorted(((scores[i], kept[i][0]) for i in range(len(kept))),
                    key=lambda t: (-t[0], t[1]))

    # Industry when the statement carries one: "Technology" puts a phone maker,
    # a GPU designer and a lithography monopoly in one bucket, which makes the
    # cap fire on the wrong pair.
    labels = {}
    for sym, _, _ in kept:
        got = (fundamentals or {}).get(sym)
        label = (str(getattr(got, "industry", "") or "").strip()
                 or str((sectors or {}).get(sym, "") or "").strip() or "Unknown")
        labels[sym] = label
    used: dict[str, int] = {}
    picked: list[str] = []
    for _, sym in ranked:
        if len(picked) >= max(0, count):
            break
        bucket = labels.get(sym, "Unknown")
        if max_per_sector > 0 and used.get(bucket, 0) >= max_per_sector:
            continue
        used[bucket] = used.get(bucket, 0) + 1
        picked.append(sym)
    if not picked:
        return []
    weight = round(invested / len(picked), 4)
    today = _date.today().isoformat()
    return [CoreHolding(symbol=sym, weight=weight, added=today,
                        thesis="按长期规则初选：站上 200 日线、近一年为正、TTM 盈利为正、"
                               "成交额充足、回撤与波动在核心可接受范围内，"
                               f"且同一行业最多 {max_per_sector} 只",
                        tag="seed")
            for sym in picked]


def _earns(symbol: str, fundamentals: dict | None) -> bool:
    """Whether ``symbol`` has positive trailing earnings, as far as we know.

    Unknown counts as no. The alternative — treating an unfetched statement as
    a pass — makes the filter silently optional exactly when the network is
    slow, which is the failure mode a long-term list can least afford.
    """
    got = (fundamentals or {}).get(str(symbol).upper())
    return bool(getattr(got, "profitable", False))



# ===========================================================================
# 波段
# ===========================================================================

@dataclass
class OpenIdea:
    """One swing recommendation still open, marked at today's close."""

    rec: object
    price: float = float("nan")
    r_now: float = float("nan")
    pnl: float = float("nan")
    to_stop: float = float("nan")      # fraction; negative = stop is below
    to_target: float = float("nan")
    days_held: int = 0
    days_left: int = 0
    spark: str = ""
    status: str = ""

    @property
    def symbol(self) -> str:
        return getattr(self.rec, "symbol", "")


def open_swing(recs: list, marks: dict, as_of: _date) -> list[OpenIdea]:
    """Every open idea, priced, with its distance to both levels.

    This is the section whose absence made the report look like it started over
    every morning. It carries no new decision — the exit engine owns those — and
    exists so the reader can see the book they are actually holding.
    """
    out: list[OpenIdea] = []
    for rec in recs:
        sym = getattr(rec, "symbol", "")
        px = _num((marks or {}).get(sym))
        idea = OpenIdea(rec=rec, price=px)
        try:
            idea.days_held = rec.days_held(as_of) or 0
        except Exception:
            idea.days_held = 0
        idea.days_left = max(0, int(_num(getattr(rec, "horizon_days", 0), 0)) - idea.days_held)
        if _ok(px) and px > 0:
            idea.r_now = rec.r_at(px) if hasattr(rec, "r_at") else float("nan")
            idea.pnl = rec.pnl_at(px) if hasattr(rec, "pnl_at") else float("nan")
            stop, target = _num(getattr(rec, "stop_price", None)), _num(getattr(rec, "target_price", None))
            if _ok(stop) and stop > 0:
                idea.to_stop = stop / px - 1.0
            if _ok(target) and target > 0:
                idea.to_target = target / px - 1.0
        idea.status = _swing_status(idea)
        out.append(idea)
    # Worst first: the position closest to its stop is the one that needs a
    # decision, and a list sorted by profit puts it last.
    out.sort(key=lambda i: (i.r_now if _ok(i.r_now) else 0.0, i.symbol))
    return out


def _swing_status(idea: OpenIdea) -> str:
    if not _ok(idea.price):
        return "无法定价"
    if _ok(idea.to_stop) and idea.to_stop > -0.02:
        return "贴近止损"
    if _ok(idea.to_target) and idea.to_target < 0.02:
        return "贴近目标"
    if _ok(idea.r_now) and idea.r_now >= 1.0:
        return "已过 1R，止损应移到成本"
    if idea.days_left <= 3:
        return "时间止损将到期"
    if _ok(idea.r_now) and idea.r_now < 0:
        return "浮亏，规则内"
    return "持有中"


def free_slots(open_count: int, slots: int = DEFAULT_SWING_SLOTS) -> int:
    """How many new swing ideas the book has room for. Never negative."""
    return max(0, int(slots) - max(0, int(open_count)))


# ===========================================================================
# 日内
# ===========================================================================

@dataclass
class DayTradeIdea:
    """Levels for one session — not an order. See the module docstring."""

    symbol: str
    price: float = float("nan")
    prev_high: float = float("nan")
    prev_low: float = float("nan")
    atr: float = float("nan")
    atr_pct: float = float("nan")
    rvol: float = float("nan")
    dollar_vol: float = float("nan")
    change_pct: float = float("nan")
    long_trigger: float = float("nan")
    long_stop: float = float("nan")
    long_target: float = float("nan")
    short_trigger: float = float("nan")
    note: str = ""

    @property
    def rr(self) -> float:
        if not (_ok(self.long_trigger) and _ok(self.long_stop) and _ok(self.long_target)):
            return float("nan")
        risk = self.long_trigger - self.long_stop
        return (self.long_target - self.long_trigger) / risk if risk > 0 else float("nan")


def daytrade_candidates(facts: dict, *, count: int = 5,
                        min_atr_pct: float = DT_MIN_ATR_PCT,
                        min_dollar_vol: float = DT_MIN_DOLLAR_VOL,
                        min_price: float = DT_MIN_PRICE,
                        min_rvol: float = DT_MIN_RVOL) -> list[DayTradeIdea]:
    """Names whose last session says tomorrow is worth watching.

    ``facts`` needs ``price``, ``prev_high``, ``prev_low``, ``atr``, ``atr_pct``,
    ``vol_ratio``, ``dollar_vol`` and ``change_pct`` per symbol.

    The filters are liquidity and range, in that order, because an intraday
    trade that cannot be exited is not a trade. The levels are the prior day's
    high and low: the two prices every other participant is also watching, which
    is the entire reason they work often enough to be worth naming. The stop is
    half an ATR rather than two — an intraday stop that gives a position a
    swing-sized amount of room is a swing trade being called a day trade.
    """
    out: list[DayTradeIdea] = []
    for sym, f in (facts or {}).items():
        price = _num(f.get("price"))
        atr = _num(f.get("atr"))
        atr_pct = _num(f.get("atr_pct"))
        rvol = _num(f.get("vol_ratio"))
        dv = _num(f.get("dollar_vol"))
        if not (_ok(price) and price >= min_price):
            continue
        if not (_ok(atr_pct) and atr_pct >= min_atr_pct):
            continue
        if not (_ok(dv) and dv >= min_dollar_vol):
            continue
        if not (_ok(rvol) and rvol >= min_rvol):
            continue
        hi, lo = _num(f.get("prev_high")), _num(f.get("prev_low"))
        if not (_ok(hi) and _ok(lo) and _ok(atr) and atr > 0):
            continue
        tick = max(0.01, round(atr * 0.1, 2))
        idea = DayTradeIdea(
            symbol=sym, price=price, prev_high=hi, prev_low=lo,
            atr=atr, atr_pct=atr_pct, rvol=rvol, dollar_vol=dv,
            change_pct=_num(f.get("change_pct")),
            long_trigger=round(hi + tick, 2),
            long_stop=round(hi + tick - atr * DT_STOP_ATR, 2),
            long_target=round(hi + tick + atr * DT_TARGET_ATR, 2),
            short_trigger=round(lo - tick, 2),
        )
        idea.note = (f"放量 {rvol:.1f} 倍" if rvol >= 2 else
                     f"波动 {atr_pct * 100:.1f}%/日")
        out.append(idea)
    # Most volatile first, liquidity being already guaranteed by the filter:
    # among names you can definitely get out of, range is what pays.
    out.sort(key=lambda i: (-(i.atr_pct if _ok(i.atr_pct) else 0.0), i.symbol))
    return out[:max(0, count)]


DAYTRADE_CAVEAT = (
    "本节不是下单指令。这台机器只有日线数据，日线推不出盘中的进场点——"
    "它能给的是明天开盘后所有人都在看的两个价位（昨日高点与昨日低点）、"
    "以及这些名字的波动与流动性够不够做日内。真正的进场要看盘中的量价，"
    "止损是 0.5 ATR 而不是波段的 2 ATR，且日内仓位不进入本报告的推荐账本、"
    "不计入战绩统计。"
)
