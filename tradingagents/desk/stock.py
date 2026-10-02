"""Task 4 — one name, everything the desk knows about it, on demand.

    python -m tradingagents.desk stock NKE            # the pack for one symbol
    python -m tradingagents.desk stock site NKE       # its page as a web site

The daily report scores sixty names with one paragraph each. This is the
other direction: one symbol, the whole file — bars, indicators and the chart
read, the manual's score with its reasons, reference levels and R, breakout
triggers, fundamentals, the next earnings date, insider buying and selling,
the sector's week and month, the headlines, and every call the desk has made
on this name before, settled against what happened. The code gathers and
computes; the judgement (``<date>-<SYM>-final.md``) is written by the reader,
in the structure of MANUAL §12.

State: ``$TRADINGAGENTS_HOME/desk/stock/``. Reads no account.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from tradingagents.live import clock
from tradingagents.live.advisor import last_completed_session, sessions_for

from . import review, task_dir, universe
from .market import SECTOR_ETFS, SECTOR_ZH, Facts, Market, _num, _ok
from .report import HORIZON_DAYS, Idea, Reporter, breakout_triggers, chart_for, score

logger = logging.getLogger(__name__)

TASK = "stock"


@dataclass
class PastCall:
    report: str
    score: int
    entry: float | None
    stop: float | None
    target: float | None
    outcome: str = "未结算"
    r: float = float("nan")
    raw: float = float("nan")
    alpha: float = float("nan")


@dataclass
class StockPack:
    symbol: str
    date: str
    data_date: str
    generated_at: str = ""
    name: str = ""
    sector: str = "Unknown"
    price: float = float("nan")
    change_pct: float = float("nan")
    ret_1w: float = float("nan")
    ret_1m: float = float("nan")
    ret_3m: float = float("nan")
    ret_1y: float = float("nan")
    rsi: float = float("nan")
    vol_ratio: float = float("nan")
    atr_pct: float = float("nan")
    sma20: float = float("nan")
    sma50: float = float("nan")
    sma200: float = float("nan")
    ext_200: float = float("nan")
    high_52w: float = float("nan")
    low_52w: float = float("nan")
    off_high_52w: float = float("nan")
    support: float = float("nan")
    resistance: float = float("nan")
    rs_3m: float = float("nan")          # three-month return minus SPY's
    spy_ret_3m: float = float("nan")
    verdict: str = ""
    spark: str = ""
    score: float = 0.0
    reasons: list = field(default_factory=list)
    cautions: list = field(default_factory=list)
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    r: float = float("nan")
    triggers: list = field(default_factory=list)
    breakout_low: float = float("nan")
    earnings_date: str = ""
    earnings_in: float = float("nan")
    insiders: dict = field(default_factory=dict)
    insiders_read: str = ""
    fundamentals_md: str = ""
    sector_etf: str = ""
    sector_w1: float = float("nan")
    sector_m1: float = float("nan")
    tilt: float = 0.0
    news: list = field(default_factory=list)
    history: list = field(default_factory=list)      # PastCall rows, newest first
    chart: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    path: str = ""
    facts: Facts | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("facts", None)
        return _clean(d)


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def past_calls(symbol: str, market: Market, as_of: date, rdir: Path | None = None,
               limit: int = 12) -> list[PastCall]:
    """Every ranking-table row the desk wrote for this name, settled where it can be."""
    from .site import my_scores
    rdir = rdir or task_dir("report")
    out = []
    for final in sorted(rdir.glob("*-final.md"), reverse=True)[:limit]:
        try:
            call = my_scores(final.read_text(encoding="utf-8")).get(symbol.upper())
        except Exception:
            continue
        if not call:
            continue
        sc, entry, stop, target, _r = call
        pc = PastCall(report=final.name[:10], score=int(sc or 0), entry=entry, stop=stop, target=target)
        session = date.fromisoformat(final.name[:10])
        bars = market.bars(symbol, as_of)
        spy = market.bars(market.spy, as_of)
        if bars.closes:
            s = review.settle_call(symbol, call, bars, spy, session, review.HOLD_SESSIONS)
            if s is not None:
                pc.outcome, pc.r, pc.raw, pc.alpha = s.outcome, s.r, s.raw, s.alpha
        out.append(pc)
    return out


class Stocker:
    def __init__(self, *, market: Market | None = None, now: datetime | None = None):
        self.market = market or Market(task=TASK)
        self.now = now

    def run(self, symbol: str, when: str | date | None = None) -> StockPack:
        now = self.now or datetime.now(clock.ET)
        try:
            today, data_day = sessions_for(when, now)        # the session the analysis is for, and its data
        except ValueError as exc:
            today = data_day = last_completed_session(now)
            logger.warning("%s", exc)
        sym = symbol.strip().upper()
        pack = StockPack(symbol=sym, date=today.isoformat(), data_date=data_day.isoformat(),
                         generated_at=datetime.now().isoformat())
        m = self.market
        f = m.facts(sym, data_day)
        if not f.ok:
            pack.warnings.append(f"{sym}: 拿不到行情")
            return self.save(pack)
        earnings = m.earnings([sym], data_day)
        fundamentals = m.fundamentals([sym])
        insiders = m.insiders([sym], as_of=data_day)
        by_symbol, _ = m.headlines([sym], macro=False)
        m.attach(f, earnings=earnings, fundamentals=fundamentals, news=by_symbol, insiders=insiders)
        if not f.sector or f.sector == "Unknown":
            f.sector = universe.sector_of(sym)
        f.name = f.name or m.names().get(sym, "")
        pack.facts, pack.name, pack.sector = f, f.name, f.sector or "Unknown"

        snap, bars = f.snap, f.bars
        pack.price, pack.change_pct = f.price, _num(snap.change_pct)
        pack.ret_1w, pack.ret_1m = bars.ret(5), _num(snap.ret_1m)
        pack.ret_3m, pack.ret_1y = _num(snap.ret_3m), bars.ret(252)
        pack.rsi, pack.vol_ratio, pack.atr_pct = _num(snap.rsi14), _num(snap.vol_ratio), _num(snap.atr_pct)
        pack.sma20, pack.sma50, pack.sma200 = _num(snap.sma20), _num(snap.sma50), _num(snap.sma200)
        pack.ext_200 = f.ext_200()
        highs, lows = bars.highs[-252:], bars.lows[-252:]
        if highs and lows:
            pack.high_52w, pack.low_52w = max(highs), min(lows)
            pack.off_high_52w = pack.price / pack.high_52w - 1 if pack.high_52w else float("nan")
        pack.support, pack.resistance = f.support(), f.resistance()
        pack.verdict, pack.spark = f.trend.verdict, f.trend.spark

        spy = m.facts(m.spy, data_day, benchmark=False)
        pack.spy_ret_3m = _num(spy.snap.ret_3m)
        if _ok(pack.ret_3m) and _ok(pack.spy_ret_3m):
            pack.rs_3m = pack.ret_3m - pack.spy_ret_3m

        # the sector's week and month, and the policy tilt
        etf_of = {yf: etf for etf, _, yf in SECTOR_ETFS}
        etf = etf_of.get(pack.sector, "")
        if etf:
            eb = m.bars(etf, data_day)
            if eb.closes:
                pack.sector_etf, pack.sector_w1, pack.sector_m1 = etf, eb.ret(5), eb.ret(21)
        with contextlib.suppress(Exception):
            from tradingagents.live.policy import sector_pressure
            events = m.policy()
            pack.tilt = float((sector_pressure(events) if events else {}).get(pack.sector, 0.0))

        # the manual's score, levels and triggers
        pack.score, pack.reasons, pack.cautions = score(f, pack.tilt, pack.spy_ret_3m)
        idea = Idea(symbol=sym, sector=pack.sector, facts=f)
        Reporter.levels(None, idea, f)
        pack.entry, pack.stop, pack.target, pack.r = idea.entry, idea.stop, idea.target, idea.r
        pack.triggers = breakout_triggers(f)
        if bars.lows:
            pack.breakout_low = bars.lows[-1]

        if f.earnings is not None:
            pack.earnings_date = str(getattr(f.earnings, "next_date", "") or "")
            with contextlib.suppress(Exception):
                pack.earnings_in = _num(f.earnings.days_to_next(today))
        ins = f.insiders
        if ins is not None:
            with contextlib.suppress(Exception):
                pack.insiders = asdict(ins)
                pack.insiders_read = ins.read()
        if f.fundamentals is not None:
            with contextlib.suppress(Exception):
                from tradingagents.live.fundamentals import markdown_block
                pack.fundamentals_md = markdown_block(f.fundamentals, pack.price)
        pack.news = [{"title": n.title, "link": n.link, "source": n.source, "published": n.published,
                      "lean": n.lean, "materiality": n.materiality} for n in f.news[:8]]
        pack.history = [asdict(c) for c in past_calls(sym, m, data_day)]
        pack.chart = chart_for(Idea(symbol=sym, stop=pack.stop, target=pack.target, facts=f))
        pack.warnings += [w for w in m.errors if w not in pack.warnings]
        return self.save(pack)

    def save(self, pack: StockPack) -> StockPack:
        d = task_dir(TASK)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{pack.date}-{pack.symbol}"
        try:
            (d / f"{stem}.json").write_text(json.dumps(pack.to_dict(), ensure_ascii=False, indent=1),
                                            encoding="utf-8")
            (d / f"{stem}.md").write_text(format_pack(pack), encoding="utf-8")
            pack.path = str(d / f"{stem}.md")
            if pack.facts is not None:
                self.write_deepdive(pack, d / f"{stem}-deepdive.md")
        except OSError as exc:
            pack.warnings.append(f"the pack could not be saved: {exc}")
        return pack

    def write_deepdive(self, pack: StockPack, path: Path) -> None:
        """The full analysis page the report writes per name, for this one name."""
        try:
            from tradingagents.live.deepdive import SymbolAnalysis, render_page
            f = pack.facts
            a = SymbolAnalysis(symbol=pack.symbol, zh=self.market.names().get(pack.symbol),
                               sector=SECTOR_ZH.get(pack.sector, pack.sector), bars=f.bars, snap=f.snap,
                               trend=f.trend, fundamentals=f.fundamentals, earnings=f.earnings,
                               news=f.news, tilt=pack.tilt)
            path.write_text(render_page(a, report_date=pack.date, data_date=pack.data_date), encoding="utf-8")
        except Exception as exc:
            pack.warnings.append(f"详情页没生成（{type(exc).__name__}: {exc}）")


# ---------------------------------------------------------------------------
# the pack as markdown
# ---------------------------------------------------------------------------

def _f(v, d=2) -> str:
    v = _num(v)
    return f"{v:,.{d}f}" if _ok(v) else "—"


def _pct(v, d=1) -> str:
    v = _num(v)
    return f"{v * 100:+.{d}f}%" if _ok(v) else "—"


def format_pack(p: StockPack) -> str:
    out = [f"# {p.symbol} · {p.name or p.symbol} · 个股数据包 · {p.date}", "",
           f"数据截至 {p.data_date} 收盘。板块 {SECTOR_ZH.get(p.sector, p.sector)}。不看任何账户。", ""]
    if p.warnings:
        out += [f"- ⚠ {w}" for w in p.warnings] + [""]
    if not _ok(p.price):
        return "\n".join(out) + "\n"
    out += ["## 一、价格与图形", "",
            "| 现价 | 日 | 周 | 月 | 三月 | 一年 | RSI | 量比 | ATR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {_f(p.price)} | {_pct(p.change_pct)} | {_pct(p.ret_1w)} | {_pct(p.ret_1m)} | {_pct(p.ret_3m)} "
            f"| {_pct(p.ret_1y)} | {_f(p.rsi, 0)} | {_f(p.vol_ratio, 1)} | {_pct(p.atr_pct)} |", "",
            "| 20 日 | 50 日 | 200 日 | 距 200 日 | 52 周高 | 52 周低 | 距 52 周高 | 支撑 | 阻力 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {_f(p.sma20)} | {_f(p.sma50)} | {_f(p.sma200)} | {_pct(p.ext_200, 0)} | {_f(p.high_52w)} "
            f"| {_f(p.low_52w)} | {_pct(p.off_high_52w, 0)} | {_f(p.support)} | {_f(p.resistance)} |", ""]
    if p.chart:
        out += ["```"] + p.chart + ["```", ""]
    out += [f"- 图形：{p.verdict}" if p.verdict else "- 图形：—",
            f"- 相对强弱：三个月 {_pct(p.ret_3m)}，标普 {_pct(p.spy_ret_3m)}，差 {_pct(p.rs_3m)}"]
    if p.sector_etf:
        out.append(f"- 板块 {SECTOR_ZH.get(p.sector, p.sector)}（{p.sector_etf}）：周 {_pct(p.sector_w1)}，月 {_pct(p.sector_m1)}，"
                   f"政策倾向 {p.tilt:+.2f}")
    out.append("")
    out += ["## 二、代码的读数", "", f"参考分 **{p.score:+.0f}**（趋势 ≤40、动量 ±20、量能 ±5、相对强弱 ±10、政策 ±10、消息 ±15、扣分）", ""]
    out += [f"- ✓ {r}" for r in p.reasons] + [f"- ⚠ {c}" for c in p.cautions]
    out += ["", f"- 参考位：入场 {_f(p.entry)} · 止损 {_f(p.stop)} · 目标 {_f(p.target)}"
            + (f"（{p.r:.1f}R，{HORIZON_DAYS} 天）" if _ok(p.r) else "")]
    trig = "、".join(p.triggers) if p.triggers else "无"
    out.append(f"- 进攻仓触发（手册 §10，图上能读出的）：{trig}；突破日低 {_f(p.breakout_low)}")
    if p.earnings_date:
        days = f"（{p.earnings_in:.0f} 天后）" if _ok(p.earnings_in) else ""
        out.append(f"- 下次财报：{p.earnings_date}{days}")
    ins = p.insiders or {}
    if ins:
        out.append(f"- 内部人（近 {ins.get('window_days', 90)} 天）：买 {ins.get('buys', 0)} 笔 ${_num(ins.get('buy_value'), 0) / 1e6:.1f}M"
                   f"（{'、'.join(ins.get('buyers', [])[:3]) or '—'}） · 卖 {ins.get('sells', 0)} 笔 "
                   f"${_num(ins.get('sell_value'), 0) / 1e6:.1f}M（{len(ins.get('sellers', []))} 人）"
                   + (f" — {p.insiders_read}" if p.insiders_read else ""))
    out.append("")
    if p.fundamentals_md:
        out += [p.fundamentals_md.rstrip(), ""]
    out.append("## 四、新闻")
    if p.news:
        for n in p.news:
            lean = {"bullish": "利好", "bearish": "利空"}.get(n["lean"], "")
            out.append(f"- [{n['title']}]({n['link']})（{n['source']}{'，' + lean if lean else ''}，分量 {n['materiality']}）")
    else:
        out.append("- 近两天没有抓到标题（新闻源空白不等于没有消息）")
    out.append("")
    out.append("## 五、这个名字的历史判断（日报的表，五日结算）")
    if p.history:
        out += ["| 日报 | 我的分 | 入场 | 止损 | 目标 | 结果 | R | 五日 | 相对标普 |", "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
        for c in p.history:
            out.append(f"| {c['report']} | {c['score']} | {_f(c['entry'])} | {_f(c['stop'])} | {_f(c['target'])} "
                       f"| {c['outcome']} | {_f(c['r'], 1)} | {_pct(c['raw'])} | {_pct(c['alpha'])} |")
    else:
        out.append("- 日报里没有对它打过分")
    out += ["", "---", "终稿写在 `<date>-<SYM>-final.md`，结构见手册 §12。这不是投资建议。", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk stock",
                                description="one name, everything the desk knows about it")
    p.add_argument("what", help="a symbol, or 'site' followed by a symbol")
    p.add_argument("symbol", nargs="?", default=None)
    p.add_argument("--date", default=None, help="the pack's date (default: today's session)")
    p.add_argument("--out", default=None, help="site: the folder to build into")
    p.add_argument("--no-pull", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the path")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.what.lower() == "site":
        if not args.symbol:
            p.error("site needs a symbol")
        from .site import build_stock
        out = build_stock(args.symbol.upper(), args.date, out_dir=Path(args.out) if args.out else None)
        print(f"site: {out}")
        print(f"publish: file_path={out}/index.html root={out} files={out}/files.json")
        return 0
    if not args.no_pull:
        from . import state
        print(state.pull())
    pack = Stocker().run(args.what, args.date)
    print(pack.path if args.quiet else format_pack(pack))
    return 0
