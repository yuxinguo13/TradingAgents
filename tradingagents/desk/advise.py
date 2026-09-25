"""Task 3 — what to do with a portfolio you typed in.

The portfolio is a file you maintain by hand — a brokerage the desk has no
API for, or several. Three formats, whichever is easiest to keep current:

    my_portfolio.json   {"cash": 12000, "holdings": [{"symbol": "AAPL", "shares": 10, "cost": 150.5}]}
    my_portfolio.csv    symbol,shares,cost
    my_portfolio.txt    AAPL 10 150.5        (one holding per line; a line "cash 12000" is the cash)

Default location ``$TRADINGAGENTS_HOME/desk/advise/my_portfolio.json``;
``python -m tradingagents.desk advise --init`` writes a template there.

For every holding the adviser reads the same facts the report reads and
answers one question — keep, add, trim or sell — on published rules
(:func:`decide`), with the levels that go with the answer. Then the portfolio
as a whole: what it is concentrated in, how its sectors sit against the policy
tilt, which names report earnings soon, and how much is in cash. No account is
queried; the Alpaca book belongs to task 1 and is not consulted.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from tradingagents.live import clock
from tradingagents.live.advisor import last_completed_session, project_target
from tradingagents.live.policy import policy_brief, sector_pressure
from tradingagents.live.sizing import pullback_entry, size_position, structural_stop

from . import task_dir, universe
from .market import SECTOR_ZH, Facts, Market, _num, _ok
from .report import HORIZON_DAYS, Idea, chart_for

logger = logging.getLogger(__name__)

TASK = "advise"
HOLD, ADD, TRIM, SELL = "持有", "加仓", "减仓", "卖出"

MAX_WEIGHT = 0.20            # above this a single name is trimmed back
TRIM_TO = 0.15
ADD_BELOW_WEIGHT = 0.05
STRETCH = 0.50               # this far above the 200-day line, take some off
BROKEN_BELOW_200 = -0.08     # this far under the 200-day line is a broken trend
NEAR_STOP = 0.03
EARNINGS_SOON_DAYS = 14


# ---------------------------------------------------------------------------
# the portfolio file
# ---------------------------------------------------------------------------

@dataclass
class Holding:
    symbol: str
    shares: float
    cost: float = float("nan")
    note: str = ""


@dataclass
class Portfolio:
    cash: float = 0.0
    holdings: list = field(default_factory=list)
    source: str = ""

    def symbols(self) -> list[str]:
        return [h.symbol for h in self.holdings]


def default_path() -> Path:
    return task_dir(TASK) / "my_portfolio.json"


TEMPLATE = {
    "cash": 10000,
    "holdings": [
        {"symbol": "AAPL", "shares": 10, "cost": 180.0, "note": "示例，改成你自己的"},
        {"symbol": "MSFT", "shares": 5, "cost": 400.0},
    ],
}


def write_template(path: Path | None = None) -> Path:
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_portfolio(text: str, kind: str = "json") -> Portfolio:
    pf = Portfolio()
    if kind == "json":
        data = json.loads(text)
        if isinstance(data, list):
            data = {"holdings": data}
        pf.cash = _num(data.get("cash", 0), 0.0)
        rows = data.get("holdings") or []
        for r in rows:
            if isinstance(r, dict):
                sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
                if sym:
                    pf.holdings.append(Holding(sym, _num(r.get("shares") or r.get("quantity"), 0.0),
                                               _num(r.get("cost") or r.get("avg_cost")),
                                               str(r.get("note") or "")))
        return pf
    if kind == "csv":
        for row in csv.DictReader(text.splitlines()):
            keys = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            sym = (keys.get("symbol") or keys.get("ticker") or "").upper()
            if sym == "CASH":
                pf.cash = _num(keys.get("shares") or keys.get("amount"), 0.0)
            elif sym:
                pf.holdings.append(Holding(sym, _num(keys.get("shares") or keys.get("quantity"), 0.0),
                                           _num(keys.get("cost") or keys.get("avg_cost")),
                                           keys.get("note", "")))
        return pf
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        sym = parts[0].upper()
        if sym == "CASH" and len(parts) > 1:
            pf.cash = _num(parts[1], 0.0)
            continue
        shares = _num(parts[1], 0.0) if len(parts) > 1 else 0.0
        cost = _num(parts[2]) if len(parts) > 2 else float("nan")
        pf.holdings.append(Holding(sym, shares, cost, " ".join(parts[3:])))
    return pf


def load_portfolio(path: Path | str | None = None) -> Portfolio:
    path = Path(path) if path else default_path()
    kind = {"json": "json", "csv": "csv"}.get(path.suffix.lstrip(".").lower(), "txt")
    pf = parse_portfolio(path.read_text(encoding="utf-8"), kind)
    pf.source = str(path)
    return pf


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------

@dataclass
class Line:
    symbol: str
    name: str = ""
    sector: str = "Unknown"
    shares: float = 0.0
    cost: float = float("nan")
    price: float = float("nan")
    value: float = float("nan")
    weight: float = float("nan")
    pnl_pct: float = float("nan")
    action: str = HOLD
    act_shares: int = 0
    urgent: bool = False
    reasons: list = field(default_factory=list)
    cautions: list = field(default_factory=list)
    stop: float = float("nan")
    target: float = float("nan")
    earnings_date: str = ""
    earnings_in: float = float("nan")
    verdict: str = ""
    spark: str = ""
    page: str = ""
    facts: Facts | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("facts", None)
        return d


@dataclass
class Advice:
    date: str
    data_date: str
    generated_at: str = ""
    source: str = ""
    total: float = float("nan")
    cash: float = 0.0
    lines: list = field(default_factory=list)
    sector_weights: dict = field(default_factory=dict)
    tilt: dict = field(default_factory=dict)
    policy_brief: str = ""
    macro_read: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    path: str = ""

    @property
    def cash_pct(self) -> float:
        return self.cash / self.total if _ok(self.total) and self.total > 0 else float("nan")

    def to_dict(self) -> dict:
        return {"date": self.date, "data_date": self.data_date, "generated_at": self.generated_at,
                "source": self.source, "total": self.total, "cash": self.cash,
                "cash_pct": self.cash_pct, "lines": [ln.to_dict() for ln in self.lines],
                "sector_weights": self.sector_weights, "tilt": self.tilt,
                "policy_brief": self.policy_brief, "macro_read": self.macro_read,
                "alerts": self.alerts, "warnings": self.warnings}


def decide(line: Line, f: Facts, *, total: float, cash: float, today: date,
           tilt: float = 0.0) -> None:
    """Keep, add, trim or sell — the rules, in the order they are tested.

    Sell beats trim beats add beats hold: the first rule that fires names the
    action, and the ones after it only add reasons. Every threshold is a
    module constant with a name, not a number in a condition.
    """
    snap = f.snap
    px = f.price
    a50, a200 = f.above("sma50"), f.above("sma200")
    s200 = _num(snap.sma200)
    r3 = _num(snap.ret_3m)
    ext = f.ext_200()
    rsi = _num(snap.rsi14)
    bear = f.bearish_news()
    hard_bear = [n for n in bear if int(getattr(n, "materiality", 0) or 0) >= 7]
    w = line.weight if _ok(line.weight) else 0.0

    # levels first: every verdict carries them
    atr_pct = _num(snap.atr_pct)
    support = f.support() if _ok(f.support()) else None
    stop = structural_stop(px, support, atr_pct=atr_pct) if _ok(atr_pct) and atr_pct > 0 else None
    line.stop = _num(stop)
    target = project_target(px, snap, HORIZON_DAYS)
    line.target = _num(target)
    if f.earnings is not None:
        line.earnings_date = str(getattr(f.earnings, "next_date", "") or "")
        with contextlib.suppress(Exception):
            line.earnings_in = _num(f.earnings.days_to_next(today))

    broken = (a200 is False and a50 is False and _ok(r3) and r3 < 0) or \
             (_ok(ext) and ext < BROKEN_BELOW_200)
    if broken:
        line.action, line.act_shares, line.urgent = SELL, int(line.shares), True
        line.reasons.append(f"趋势已坏：价格 {px:,.2f} 在 50 日和 200 日线（{s200:,.2f}）下方"
                            + (f"，近三月 {r3 * 100:+.0f}%" if _ok(r3) else ""))
    elif hard_bear:
        line.action, line.urgent = (SELL if a200 is False else TRIM), True
        line.act_shares = int(line.shares) if line.action == SELL else max(1, int(line.shares // 2))
        line.reasons.append(f"近两天有重大利空：{hard_bear[0].title}")
    elif w > MAX_WEIGHT:
        line.action = TRIM
        line.act_shares = max(1, int(line.shares * (1 - TRIM_TO / w)))
        line.reasons.append(f"单一持仓占 {w * 100:.0f}%，超过 {MAX_WEIGHT * 100:.0f}% 的上限，减到 {TRIM_TO * 100:.0f}%")
    elif (_ok(ext) and ext > STRETCH) or (_ok(rsi) and rsi >= 80):
        line.action = TRIM
        line.act_shares = max(1, int(line.shares // 3))
        line.reasons.append(("高出 200 日线 %.0f%%" % (ext * 100)) if _ok(ext) and ext > STRETCH
                            else f"RSI {rsi:.0f}，短线过热" + "，先落袋三分之一")
    else:
        pull = pullback_entry(px, _num(snap.sma20), support, sma50=_num(snap.sma50), atr_pct=atr_pct) \
            if _ok(atr_pct) and atr_pct > 0 else None
        near_pull = pull is not None and abs(px - pull) / px <= 0.02
        soon = _ok(line.earnings_in) and 0 <= line.earnings_in <= EARNINGS_SOON_DAYS
        if a50 and a200 and w < ADD_BELOW_WEIGHT and cash > 0 and near_pull and not soon and not bear:
            sized = size_position(total, px, line.stop, risk_pct=1.0, cap_fraction=0.08) \
                if _ok(line.stop) and line.stop < px else None
            want = sized.quantity if sized else 0
            room = int(cash // px) if px else 0
            add = min(want, room)
            if add > 0:
                line.action, line.act_shares = ADD, add
                line.reasons.append(f"趋势完好且回调到均线附近（{pull:,.2f}），仓位只占 {w * 100:.1f}%，可以补")
        if line.action == HOLD:
            line.reasons.append(f.trend.verdict or "趋势未坏，继续持有")

    # the cautions everyone gets
    if _ok(line.stop) and px > 0 and (px - line.stop) / px <= NEAR_STOP:
        line.cautions.append(f"离止损 {line.stop:,.2f} 只有 {(px - line.stop) / px * 100:.1f}%")
        line.urgent = line.urgent or line.action != SELL
    if _ok(line.earnings_in) and 0 <= line.earnings_in <= EARNINGS_SOON_DAYS:
        line.cautions.append(f"{line.earnings_in:.0f} 天后财报（{line.earnings_date}），持仓过财报要想清楚")
    if bear and not hard_bear:
        line.cautions.append(f"有利空消息 {len(bear)} 条，看一眼")
    if tilt < -0.2:
        line.cautions.append(f"政策面对该板块不利（倾向 {tilt:+.2f}）")
    elif tilt > 0.2:
        line.reasons.append(f"政策面对该板块有利（倾向 {tilt:+.2f}）")
    if _ok(line.pnl_pct) and line.pnl_pct < -0.15 and line.action == HOLD:
        line.cautions.append(f"成本以上还差 {abs(line.pnl_pct) * 100:.0f}%，别因为亏着就不动")


# ---------------------------------------------------------------------------
# the adviser
# ---------------------------------------------------------------------------

class Adviser:
    def __init__(self, *, market: Market | None = None, now: datetime | None = None,
                 with_pages: bool = True):
        self.market = market or Market(task=TASK)
        self.now = now
        self.with_pages = with_pages

    def run(self, portfolio: Portfolio) -> Advice:
        now = self.now or datetime.now(clock.ET)
        data_day = last_completed_session(now)
        today = clock.market_state(now).trading_day
        advice = Advice(date=today.isoformat(), data_date=data_day.isoformat(),
                        generated_at=datetime.now().isoformat(), source=portfolio.source,
                        cash=_num(portfolio.cash, 0.0))
        if not portfolio.holdings:
            advice.warnings.append("组合里没有持仓")
            return self.save(advice)

        events = self.market.policy()
        try:
            advice.policy_brief = policy_brief(events)
            advice.tilt = sector_pressure(events) if events else {}
        except Exception as exc:
            advice.warnings.append(f"policy brief failed ({type(exc).__name__}: {exc})")
        try:
            advice.macro_read = self.market.macro(data_day).read()
        except Exception as exc:
            advice.warnings.append(f"macro board failed ({type(exc).__name__}: {exc})")

        syms = portfolio.symbols()
        facts = {s: self.market.facts(s, data_day) for s in syms}
        alive = [s for s in syms if facts[s].ok]
        earnings = self.market.earnings(alive, data_day)
        fundamentals = self.market.fundamentals(alive)
        by_symbol, _ = self.market.headlines(alive, macro=False)
        for s in alive:
            self.market.attach(facts[s], earnings=earnings, fundamentals=fundamentals, news=by_symbol)
            if not facts[s].sector or facts[s].sector == "Unknown":
                facts[s].sector = universe.sector_of(s)

        # value first, so every weight is against the same total
        lines = []
        total = advice.cash
        for h in portfolio.holdings:
            f = facts[h.symbol]
            ln = Line(symbol=h.symbol, name=f.name, sector=f.sector or "Unknown",
                      shares=h.shares, cost=h.cost, price=f.price, facts=f)
            if not f.ok:
                ln.action, ln.cautions = HOLD, ["拿不到行情，无法判断"]
                advice.warnings.append(f"{h.symbol}: 拿不到行情")
            else:
                ln.value = h.shares * f.price
                ln.pnl_pct = f.price / h.cost - 1.0 if _ok(h.cost) and h.cost > 0 else float("nan")
                total += ln.value
                ln.verdict, ln.spark = f.trend.verdict, f.trend.spark
            lines.append(ln)
        advice.total = total
        for ln in lines:
            if _ok(ln.value) and total > 0:
                ln.weight = ln.value / total
                advice.sector_weights[ln.sector] = advice.sector_weights.get(ln.sector, 0.0) + ln.weight
        for ln in lines:
            if ln.facts is not None and ln.facts.ok:
                decide(ln, ln.facts, total=total, cash=advice.cash, today=today,
                       tilt=advice.tilt.get(ln.sector, 0.0))
        order = {SELL: 0, TRIM: 1, ADD: 2, HOLD: 3}
        lines.sort(key=lambda ln: (order.get(ln.action, 9), not ln.urgent, -(ln.weight if _ok(ln.weight) else 0)))
        advice.lines = lines
        self.portfolio_alerts(advice)
        if self.with_pages:
            self.write_pages(advice)
        return self.save(advice)

    def portfolio_alerts(self, advice: Advice) -> None:
        weights = sorted((ln.weight for ln in advice.lines if _ok(ln.weight)), reverse=True)
        if weights and weights[0] > MAX_WEIGHT:
            top = next(ln for ln in advice.lines if ln.weight == weights[0])
            advice.alerts.append(f"{top.symbol} 一个名字占了 {weights[0] * 100:.0f}%")
        if len(weights) >= 3 and sum(weights[:3]) > 0.60:
            advice.alerts.append(f"前三大持仓合计 {sum(weights[:3]) * 100:.0f}%，集中度高")
        for sector, w in sorted(advice.sector_weights.items(), key=lambda kv: -kv[1]):
            if w > 0.40 and sector != "Unknown":
                advice.alerts.append(f"{SECTOR_ZH.get(sector, sector)}板块占 {w * 100:.0f}%，一个板块的风险")
            t = advice.tilt.get(sector, 0.0)
            if w > 0.15 and t < -0.2:
                advice.alerts.append(f"{SECTOR_ZH.get(sector, sector)}占 {w * 100:.0f}% 而政策面对它不利（{t:+.2f}）")
        cp = advice.cash_pct
        if _ok(cp) and cp < 0.05:
            advice.alerts.append(f"现金只剩 {cp * 100:.0f}%，没有补仓和抗跌的余地")
        soon = [ln for ln in advice.lines if _ok(ln.earnings_in) and 0 <= ln.earnings_in <= EARNINGS_SOON_DAYS]
        if soon:
            advice.alerts.append("两周内财报：" + "、".join(f"{ln.symbol}（{ln.earnings_date}）" for ln in soon))
        urgent = [ln.symbol for ln in advice.lines if ln.urgent]
        if urgent:
            advice.alerts.insert(0, "需要今天处理：" + "、".join(urgent))

    def write_pages(self, advice: Advice) -> None:
        from tradingagents.live.deepdive import SymbolAnalysis, render_page
        d = task_dir(TASK) / advice.date
        d.mkdir(parents=True, exist_ok=True)
        names = self.market.names()
        for ln in advice.lines:
            f = ln.facts
            if f is None or not f.ok:
                continue
            try:
                a = SymbolAnalysis(symbol=ln.symbol, zh=names.get(ln.symbol, ln.name),
                                   sector=SECTOR_ZH.get(ln.sector, ln.sector), bars=f.bars, snap=f.snap,
                                   trend=f.trend, fundamentals=f.fundamentals, earnings=f.earnings,
                                   news=f.news, tilt=advice.tilt.get(ln.sector, 0.0))
                (d / f"{ln.symbol}.md").write_text(
                    render_page(a, report_date=advice.date, data_date=advice.data_date,
                                back_link=f"../{advice.date}.md"), encoding="utf-8")
                ln.page = f"{advice.date}/{ln.symbol}.md"
            except Exception as exc:
                advice.warnings.append(f"{ln.symbol}: 详情页没生成（{type(exc).__name__}: {exc}）")

    def save(self, advice: Advice) -> Advice:
        try:
            d = task_dir(TASK)
            path = d / f"{advice.date}.md"
            path.write_text(format_advice(advice), encoding="utf-8")
            (d / f"{advice.date}.json").write_text(
                json.dumps(advice.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            advice.path = str(path)
        except OSError as exc:
            advice.warnings.append(f"the advice could not be saved: {exc}")
        return advice


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _f(v: float, digits: int = 2) -> str:
    return f"{v:,.{digits}f}" if _ok(v) else "—"


def _pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%" if _ok(v) else "—"


def _share(v: float) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


def format_advice(advice: Advice) -> str:
    out = [f"# 持仓建议 · {advice.date}", "",
           f"数据截至 {advice.data_date} 收盘。组合来自 `{advice.source or '（未指定）'}`；"
           f"总值 ${_f(advice.total)}，现金 ${_f(advice.cash)}（{_share(advice.cash_pct)}）。", ""]
    if advice.alerts:
        out += ["## 先看这里"] + [f"- {a}" for a in advice.alerts] + [""]
    out += ["## 每个持仓", "",
            "| 代码 | 板块 | 股数 | 成本 | 现价 | 盈亏 | 占比 | 建议 | 数量 | 止损 | 目标 |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for ln in advice.lines:
        act = f"**{ln.action}**" if ln.action != HOLD else ln.action
        if ln.urgent:
            act = "🔴 " + act
        out.append(f"| {ln.symbol} | {SECTOR_ZH.get(ln.sector, ln.sector)} | {ln.shares:g} | {_f(ln.cost)} "
                   f"| {_f(ln.price)} | {_pct(ln.pnl_pct)} | {_share(ln.weight)} | {act} "
                   f"| {ln.act_shares or '—'} | {_f(ln.stop)} | {_f(ln.target)} |")
    out.append("")
    for ln in advice.lines:
        out.append(f"### {ln.symbol} {ln.name} — {ln.action}" + (f" {ln.act_shares} 股" if ln.act_shares else ""))
        if ln.spark:
            out.append(f"`{ln.spark}`")
        idea = Idea(symbol=ln.symbol, stop=ln.stop, target=ln.target, facts=ln.facts)
        chart = chart_for(idea)
        if chart:
            out += ["```"] + chart + ["```"]
        out += [f"- ✓ {r}" for r in ln.reasons]
        out += [f"- ⚠ {c}" for c in ln.cautions]
        if ln.facts and ln.facts.news:
            for n in ln.facts.news[:3]:
                lean = {"bullish": "利好", "bearish": "利空"}.get(n.lean, "")
                out.append(f"- 新闻：[{n.title}]({n.link})（{n.source}{'，' + lean if lean else ''}）")
        if ln.page:
            out.append(f"- [详情页]({ln.page})")
        out.append("")
    if advice.sector_weights:
        out += ["## 板块分布", "", "| 板块 | 占比 | 政策倾向 |", "|---|---:|---:|"]
        for sector, w in sorted(advice.sector_weights.items(), key=lambda kv: -kv[1]):
            out.append(f"| {SECTOR_ZH.get(sector, sector)} | {w * 100:.0f}% | {advice.tilt.get(sector, 0.0):+.2f} |")
        out.append("")
    if advice.macro_read or advice.policy_brief:
        out.append("## 背景")
        out += [f"- {m}" for m in advice.macro_read]
        if advice.policy_brief:
            out += ["", advice.policy_brief.strip()]
        out.append("")
    if advice.warnings:
        out += ["## 说明"] + [f"- ⚠ {w}" for w in advice.warnings] + [""]
    out += ["---",
            f"规则：跌破 200 日线 {abs(BROKEN_BELOW_200) * 100:.0f}% 或同时失守 50/200 日线且三月为负 → 卖出；"
            f"重大利空 → 卖出/减半；单一持仓 > {MAX_WEIGHT * 100:.0f}% → 减到 {TRIM_TO * 100:.0f}%；"
            f"高出 200 日线 {STRETCH * 100:.0f}% 或 RSI ≥ 80 → 减三分之一；"
            f"趋势完好、回调到均线、占比 < {ADD_BELOW_WEIGHT * 100:.0f}%、有现金、两周内无财报 → 加仓。这不是投资建议。", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk advise",
                                description="what to do with a portfolio you typed in")
    p.add_argument("--portfolio", default=None, help="json / csv / txt file (default: desk/advise/my_portfolio.json)")
    p.add_argument("--init", action="store_true", help="write a template portfolio file and exit")
    p.add_argument("--no-pages", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.init:
        path = write_template(Path(args.portfolio) if args.portfolio else None)
        print(f"template written to {path}; edit it, then run again without --init")
        return 0
    try:
        pf = load_portfolio(args.portfolio)
    except FileNotFoundError:
        print(f"no portfolio file at {args.portfolio or default_path()}; run with --init to create one")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    advice = Adviser(with_pages=not args.no_pages).run(pf)
    print(advice.path if args.quiet else format_advice(advice))
    return 0
