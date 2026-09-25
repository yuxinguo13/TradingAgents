"""Settle what the desk said against what the market did, and keep score.

Upstream's graph settles every decision after a fixed holding window and
scores it as **alpha against a benchmark**, then aggregates hit rate and mean
alpha per rating so a rule change can rest on a sample rather than a memory.
The same idea, in the desk's own units:

* a report call (a row of the ranking table) is settled ``HOLD_SESSIONS``
  sessions after the session it was written for. If the market reached its
  entry the call is scored in R, exactly as a filled trade would be: stopped,
  target, or marked at the last close. Its raw move and its move relative to
  SPY are recorded whether or not the entry was reached, so "+2% on a day the
  index did +3%" reads as what it is.
* a closed trade in the book is scored the same way, in R and against SPY
  over its own holding period, per sleeve and per principle.

``desk review`` prints the aggregate. Manual §11 asks for three logged cases
before a rule moves; this is where the cases are counted.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from . import task_dir
from .market import Market, _num, _ok

logger = logging.getLogger(__name__)

HOLD_SESSIONS = 5           # sessions a report call is given before it is settled
REPORT_TASK = "report"
TRADE_TASK = "trade"


# ---------------------------------------------------------------------------
# returns over a window of bars
# ---------------------------------------------------------------------------

def _index_on_or_after(dates: list, when: date) -> int | None:
    for i, d in enumerate(dates):
        dd = d.date() if hasattr(d, "date") and not isinstance(d, date) else d
        if isinstance(dd, str):
            dd = date.fromisoformat(dd[:10])
        if dd >= when:
            return i
    return None


def _dates(bars) -> list[date]:
    out = []
    for d in bars.dates:
        if isinstance(d, date):
            out.append(d)
        elif hasattr(d, "date"):
            out.append(d.date())
        else:
            out.append(date.fromisoformat(str(d)[:10]))
    return out


def window_return(bars, start: date, end: date) -> float:
    """Close-to-close return from the last close before ``start`` to the close
    on or before ``end``. NaN when the bars do not cover the window."""
    if bars is None or not bars.closes:
        return float("nan")
    ds = _dates(bars)
    i0 = _index_on_or_after(ds, start)
    if i0 is None or i0 == 0:
        return float("nan")
    i1 = None
    for i in range(len(ds) - 1, -1, -1):
        if ds[i] <= end:
            i1 = i
            break
    if i1 is None or i1 < i0:
        return float("nan")
    base = bars.closes[i0 - 1]
    return bars.closes[i1] / base - 1.0 if base else float("nan")


def alpha_between(market: Market, symbol: str, start: date, end: date, as_of: date) -> tuple[float, float]:
    """(symbol return, return relative to SPY) over the holding window."""
    mine = window_return(market.bars(symbol, as_of), start, end)
    spy = window_return(market.bars(market.spy, as_of), start, end)
    alpha = mine - spy if _ok(mine) and _ok(spy) else float("nan")
    return mine, alpha


# ---------------------------------------------------------------------------
# settling a report call
# ---------------------------------------------------------------------------

@dataclass
class Settled:
    symbol: str
    report: str                 # the session the call was written for
    score: int
    entry: float | None
    stop: float | None
    target: float | None
    sessions: int = 0           # how many sessions the window covered
    entered: bool = False
    outcome: str = "未触发"      # 未触发 | 止损 | 到目标 | 持有中
    r: float = float("nan")     # R if entered, marked at exit or last close
    raw: float = float("nan")   # close-to-close move over the window
    spy: float = float("nan")
    alpha: float = float("nan")
    last: float = float("nan")


def settle_call(symbol: str, call: tuple, bars, spy_bars, session: date,
                hold: int = HOLD_SESSIONS) -> Settled | None:
    """Score one ranking-table row over ``hold`` sessions from ``session``.

    Returns None when the bars do not yet cover the whole window: a call is
    settled once, on the full window, never early on a partial one.
    """
    score, entry, stop, target, _r = call
    ds = _dates(bars)
    i0 = _index_on_or_after(ds, session)
    if i0 is None or i0 == 0 or i0 + hold - 1 >= len(ds):
        return None
    i1 = i0 + hold - 1
    out = Settled(symbol=symbol, report=session.isoformat(), score=int(score or 0),
                  entry=entry, stop=stop, target=target, sessions=hold)
    base = bars.closes[i0 - 1]
    out.last = bars.closes[i1]
    out.raw = out.last / base - 1.0 if base else float("nan")
    if spy_bars is not None and spy_bars.closes:
        sds = _dates(spy_bars)
        j0 = _index_on_or_after(sds, session)
        j1 = _index_on_or_after(sds, ds[i1])
        if j0 and j1 is not None and j1 >= j0 and spy_bars.closes[j0 - 1]:
            out.spy = spy_bars.closes[j1] / spy_bars.closes[j0 - 1] - 1.0
            out.alpha = out.raw - out.spy
    if not entry or not stop or entry <= stop:
        return out
    risk = entry - stop
    for i in range(i0, i1 + 1):
        lo, hi = bars.lows[i], bars.highs[i]
        if not out.entered:
            if _ok(lo) and lo <= entry:
                out.entered = True
                # a bar that also breaks the stop is a fill and a stop-out
                if lo <= stop:
                    out.outcome, out.r = "止损", -1.0
                    return out
                if target and _ok(hi) and hi >= target:
                    out.outcome, out.r = "到目标", (target - entry) / risk
                    return out
            continue
        if _ok(lo) and lo <= stop:
            out.outcome, out.r = "止损", -1.0
            return out
        if target and _ok(hi) and hi >= target:
            out.outcome, out.r = "到目标", (target - entry) / risk
            return out
    if out.entered:
        out.outcome, out.r = "持有中", (out.last - entry) / risk
    return out


def finals_before(before: str, rdir: Path | None = None) -> list[Path]:
    rdir = rdir or task_dir(REPORT_TASK)
    return sorted(p for p in rdir.glob("*-final.md") if p.name[:10] < before)


def settle_report(market: Market, final: Path, as_of: date, hold: int = HOLD_SESSIONS) -> list[Settled]:
    """Every call in one final report, settled if its window has traded."""
    from .site import my_scores
    try:
        calls = my_scores(final.read_text(encoding="utf-8"))
    except Exception:
        return []
    session = date.fromisoformat(final.name[:10])
    spy = market.bars(market.spy, as_of)
    out = []
    for sym, call in calls.items():
        bars = market.bars(sym, as_of)
        if not bars.closes:
            continue
        s = settle_call(sym, call, bars, spy, session, hold)
        if s is not None:
            out.append(s)
    return out


def latest_settleable(market: Market, as_of: date, before: str, hold: int = HOLD_SESSIONS,
                      rdir: Path | None = None) -> tuple[Path | None, list[Settled]]:
    """The newest final report whose ``hold``-session window has fully traded."""
    for final in reversed(finals_before(before, rdir)):
        settled = settle_report(market, final, as_of, hold)
        if settled:
            return final, settled
    return None, []


# ---------------------------------------------------------------------------
# the aggregate
# ---------------------------------------------------------------------------

@dataclass
class Bucket:
    n: int = 0
    hits: int = 0
    r_sum: float = 0.0
    alpha_sum: float = 0.0
    alpha_n: int = 0

    def add(self, r: float, alpha: float = float("nan")) -> None:
        self.n += 1
        if _ok(r) and r > 0:
            self.hits += 1
        if _ok(r):
            self.r_sum += r
        if _ok(alpha):
            self.alpha_sum += alpha
            self.alpha_n += 1

    @property
    def hit_rate(self) -> float:
        return self.hits / self.n if self.n else float("nan")

    @property
    def mean_r(self) -> float:
        return self.r_sum / self.n if self.n else float("nan")

    @property
    def mean_alpha(self) -> float:
        return self.alpha_sum / self.alpha_n if self.alpha_n else float("nan")

    def row(self, label: str) -> str:
        return (f"| {label} | {self.n} | {_pct(self.hit_rate, 0)} | {_f(self.mean_r, 2)} "
                f"| {_f(self.r_sum, 1)} | {_pct(self.mean_alpha)} |")


@dataclass
class Review:
    date: str
    trades: dict = field(default_factory=dict)        # sleeve → Bucket
    principles: dict = field(default_factory=dict)    # principle → Bucket
    calls: dict = field(default_factory=dict)         # score band → Bucket (entered calls)
    calls_all: Bucket = field(default_factory=Bucket)  # every call, alpha only
    reports: int = 0
    settled: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _band(score: int) -> str:
    if score >= 75:
        return "≥75"
    if score >= 65:
        return "65–74"
    return "<65"


def review_book(market: Market, book, as_of: date, rv: Review) -> None:
    """Closed trades: R and alpha vs SPY, per sleeve and per principle."""
    for c in book.closed:
        try:
            opened, closed = date.fromisoformat(c.opened), date.fromisoformat(c.closed)
        except ValueError:
            continue
        raw, alpha = alpha_between(market, c.symbol, opened, closed, as_of)
        sleeve = getattr(c, "sleeve", "core") or "core"
        rv.trades.setdefault(sleeve, Bucket()).add(_num(c.r), alpha)
        for p in c.principles or []:
            rv.principles.setdefault(str(p)[:24], Bucket()).add(_num(c.r), alpha)
        rv.closed.append({"symbol": c.symbol, "sleeve": sleeve, "opened": c.opened, "closed": c.closed,
                          "r": _num(c.r), "raw": raw, "alpha": alpha, "reason": c.reason})


def review_reports(market: Market, as_of: date, rv: Review, hold: int = HOLD_SESSIONS,
                   rdir: Path | None = None, limit: int = 60) -> None:
    """Every settled report call on record, banded by the score it was given."""
    for final in reversed(finals_before(as_of.isoformat(), rdir)[-limit:]):
        settled = settle_report(market, final, as_of, hold)
        if not settled:
            continue
        rv.reports += 1
        for s in settled:
            rv.calls_all.add(float("nan"), s.alpha)
            if s.entered:
                rv.calls.setdefault(_band(s.score), Bucket()).add(s.r, s.alpha)
            rv.settled.append(asdict(s))


def run(market: Market | None = None, *, book=None, as_of: date | None = None,
        hold: int = HOLD_SESSIONS) -> Review:
    from .orders import DeskBook
    market = market or Market(task=TRADE_TASK)
    as_of = as_of or date.today()
    rv = Review(date=as_of.isoformat())
    book = book or DeskBook()
    review_book(market, book, as_of, rv)
    review_reports(market, as_of, rv, hold)
    if not rv.closed:
        rv.notes.append("账上还没有平掉的仓位；交易这一栏要等第一笔出场")
    if not rv.reports:
        rv.notes.append(f"还没有满 {hold} 个交易日的日报可以结算")
    return rv


def _f(v, d=2) -> str:
    v = _num(v)
    return f"{v:,.{d}f}" if _ok(v) else "—"


def _pct(v, d=1) -> str:
    v = _num(v)
    return f"{v * 100:+.{d}f}%" if _ok(v) else "—"


def format_review(rv: Review) -> str:
    out = [f"# 决策质量汇总 {rv.date}", "",
           "每一栏都按 R 和相对标普的超额记分。样本不到 3 个的行只是记录，不是证据（手册 §11）。", ""]
    out.append("## 平仓交易（按账本）")
    if rv.trades:
        out += ["| 账 | 笔数 | 胜率 | 平均 R | 累计 R | 平均超额 |", "|---|---:|---:|---:|---:|---:|"]
        out += [b.row("主仓" if k == "core" else "进攻仓" if k == "aggressive" else k)
                for k, b in sorted(rv.trades.items())]
        out.append("")
        if rv.principles:
            out += ["| 原则 | 笔数 | 胜率 | 平均 R | 累计 R | 平均超额 |", "|---|---:|---:|---:|---:|---:|"]
            out += [b.row(k) for k, b in sorted(rv.principles.items(), key=lambda kv: -kv[1].n)]
            out.append("")
        out += ["| 代码 | 账 | 开 | 平 | R | 自身 | 超额 | 原因 |", "|---|---|---|---|---:|---:|---:|---|"]
        for c in rv.closed:
            out.append(f"| {c['symbol']} | {'进攻' if c['sleeve'] == 'aggressive' else '主'} | {c['opened']} | {c['closed']} "
                       f"| {_f(c['r'], 1)} | {_pct(c['raw'])} | {_pct(c['alpha'])} | {c['reason']} |")
    else:
        out.append("- 没有")
    out.append("")
    out.append(f"## 日报判断（{rv.reports} 份日报，每份满 {HOLD_SESSIONS} 个交易日结算）")
    if rv.reports:
        out.append(f"- 全部判断相对标普的平均超额：{_pct(rv.calls_all.mean_alpha)}（{rv.calls_all.alpha_n} 个）")
        if rv.calls:
            out += ["", "触发了入场位的判断，按当天给的分数分组：", "",
                    "| 分数 | 个数 | 胜率 | 平均 R | 累计 R | 平均超额 |", "|---|---:|---:|---:|---:|---:|"]
            out += [rv.calls[k].row(k) for k in ("≥75", "65–74", "<65") if k in rv.calls]
        out.append("")
    else:
        out.append("- 没有")
    if rv.notes:
        out += [""] + [f"- {n}" for n in rv.notes]
    return "\n".join(out) + "\n"


def _clean(v):
    """NaN → null, so the JSON is JSON."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def save(rv: Review) -> Path:
    d = task_dir(TRADE_TASK)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"review-{rv.date}.json").write_text(
        json.dumps(_clean(asdict(rv)), ensure_ascii=False, indent=1), encoding="utf-8")
    path = d / f"review-{rv.date}.md"
    path.write_text(format_review(rv), encoding="utf-8")
    return path


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk review",
                                description="hit rate, R and alpha vs SPY over the log: the sample §11 asks for")
    p.add_argument("--date", default=None, help="settle as of this date (default: today)")
    p.add_argument("--hold", type=int, default=HOLD_SESSIONS, help="sessions a report call is given")
    p.add_argument("--no-pull", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.no_pull:
        from . import state
        print(state.pull())
    as_of = date.fromisoformat(args.date) if args.date else date.today()
    rv = run(as_of=as_of, hold=args.hold)
    path = save(rv)
    print(format_review(rv))
    print(path)
    return 0

