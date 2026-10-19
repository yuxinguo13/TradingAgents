"""The data pack: everything Claude reads before it decides, as tables.

    python -m tradingagents.desk pack trade [--add NVDA,ARM]   # the morning pack
    python -m tradingagents.desk pack facts NVDA,ARM            # a few more names, same table

The report and the advice already are packs — ``desk report`` and ``desk
advise`` write the tables, the charts, the reference scores and the JSON —
so this module only builds the one the trade task lacks: the account as the
venue sees it, the book with its levels and its reasons, what the venue did
since the last run (stops that fired → the post-mortem queue), positions
with no stop resting at the venue, the regime, and the candidates with the
same indicator table the report uses.

Nothing here decides anything. The reference levels in the candidate table
(entry, stop, target, R) are the manual's arithmetic applied mechanically so
Claude has a starting point to agree with or overrule; the gate in
``orders.py`` is what enforces the limits.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from tradingagents.live import clock
from tradingagents.live.advisor import last_completed_session

from . import review, task_dir, universe
from .market import SECTOR_ZH, Facts, Market, _num, _ok
from .orders import MAX_POSITIONS, SLEEVE, DeskBook, Position
from .report import Idea, chart_for, score

PENDING_DAYS = 2   # a resting buy that has not filled in this many SESSIONS is cancelled


def sessions_since(opened: str, today: date) -> int:
    """Trading days between two dates (weekends excluded). 2026-10-19: counted
    in calendar days, a Thursday's limit was three days old on Monday."""
    try:
        import numpy as np
        return int(np.busday_count(date.fromisoformat(opened), today))
    except (ValueError, TypeError):
        return 99

logger = logging.getLogger(__name__)

TASK = "trade"


@dataclass
class Row:
    symbol: str
    sector: str = ""
    price: float = float("nan")
    change_pct: float = float("nan")
    ret_1m: float = float("nan")
    ret_3m: float = float("nan")
    rsi: float = float("nan")
    vol_ratio: float = float("nan")
    atr_pct: float = float("nan")
    sma20: float = float("nan")
    sma50: float = float("nan")
    sma200: float = float("nan")
    ext_200: float = float("nan")
    support: float = float("nan")
    resistance: float = float("nan")
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    r: float = float("nan")
    earnings_date: str = ""
    earnings_in: float = float("nan")
    ref_score: float = float("nan")
    verdict: str = ""
    spark: str = ""
    news: list = field(default_factory=list)
    source: str = ""
    screen_rank: int = 0


@dataclass
class HeldRow:
    symbol: str
    shares: int
    avg_cost: float
    price: float
    value: float
    pnl_pct: float
    stop: float = float("nan")
    target: float = float("nan")
    r_now: float = float("nan")
    opened: str = ""
    days_left: int | None = None
    protected: bool = False
    in_book: bool = False
    thesis: str = ""
    invalidation: str = ""
    earnings_in: float = float("nan")
    verdict: str = ""
    sleeve: str = "core"
    news: list = field(default_factory=list)


@dataclass
class TradePack:
    date: str
    data_date: str
    generated_at: str = ""
    equity: float = float("nan")
    last_equity: float = float("nan")
    cash: float = float("nan")
    market_open: bool = False
    held: list = field(default_factory=list)
    postmortems: list = field(default_factory=list)      # Closed rows awaiting review
    fills: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    breakouts: list = field(default_factory=list)     # (symbol, triggers, breakout-day low)
    regime: list = field(default_factory=list)
    macro_news: list = field(default_factory=list)
    policy_brief: str = ""
    tilt: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    reminders: list = field(default_factory=list)   # resting buys and 1R trims the trader tends to miss
    warnings: list = field(default_factory=list)
    path: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("held", "candidates"):
            for row in d[key]:
                row["news"] = [_news_dict(n) for n in row["news"]]
        d["macro_news"] = [_news_dict(n) for n in self.macro_news]
        return d


def _news_dict(n) -> dict:
    if isinstance(n, dict):
        return n
    return {"title": n.title, "link": n.link, "source": n.source, "published": n.published,
            "lean": n.lean, "materiality": n.materiality}


def row_for(f: Facts, tilt: float, spy_r3: float, *, source: str = "", rank: int = 0,
            today: date | None = None) -> Row:
    """One name's line in the table, with the manual's reference levels."""
    idea = Idea(symbol=f.symbol, sector=f.sector, facts=f)
    from .report import Reporter
    Reporter.levels(None, idea, f)
    sc, _, _ = score(f, tilt, spy_r3)
    row = Row(symbol=f.symbol, sector=f.sector, price=f.price, change_pct=_num(f.snap.change_pct),
              ret_1m=_num(f.snap.ret_1m), ret_3m=_num(f.snap.ret_3m), rsi=_num(f.snap.rsi14),
              vol_ratio=_num(f.snap.vol_ratio), atr_pct=_num(f.snap.atr_pct),
              sma20=_num(f.snap.sma20), sma50=_num(f.snap.sma50), sma200=_num(f.snap.sma200),
              ext_200=f.ext_200(), support=f.support(), resistance=f.resistance(),
              entry=idea.entry, stop=idea.stop, target=idea.target, r=idea.r, ref_score=sc,
              verdict=f.trend.verdict, spark=f.trend.spark, news=list(f.news)[:4],
              source=source, screen_rank=rank)
    if f.earnings is not None:
        row.earnings_date = str(getattr(f.earnings, "next_date", "") or "")
        with contextlib.suppress(Exception):
            row.earnings_in = _num(f.earnings.days_to_next(today or date.today()))
    return row


def last_buy_record(symbol: str, path: Path | None = None) -> dict:
    """The most recent accepted buy of ``symbol`` in the decision log, or {}."""
    path = path or (task_dir(TASK) / "decisions.jsonl")
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if (e.get("kind") == "order" and e.get("action") == "buy" and e.get("ok")
                and str(e.get("symbol", "")).upper() == symbol.upper() and not e.get("dry_run")):
            return e
    return {}


class Packer:
    def __init__(self, *, broker=None, market: Market | None = None, book: DeskBook | None = None,
                 screen=None, now: datetime | None = None, top: int = 40, per_sector: int = 3,
                 exchange: str = "all"):
        self._broker = broker
        self.market = market or Market(task=TASK)
        self.book = book or DeskBook()
        self.screen = screen
        self.now = now
        self.top, self.per_sector, self.exchange = top, per_sector, exchange

    def broker(self):
        if self._broker is None:
            from tradingagents.live.broker import open_broker
            self._broker = open_broker("alpaca")
        return self._broker

    def run(self, extra: list[str] = ()) -> TradePack:
        now = self.now or datetime.now(clock.ET)
        st = clock.market_state(now)
        today, data_day = st.trading_day, last_completed_session(now)
        pack = TradePack(date=today.isoformat(), data_date=data_day.isoformat(),
                         generated_at=datetime.now().isoformat(), market_open=st.is_open)
        from . import orders as O
        pack.limits = {"risk_pct": O.RISK_PCT, "max_name_weight": O.MAX_NAME_WEIGHT,
                       "max_daily_new_risk": O.MAX_DAILY_NEW_RISK, "daily_drawdown_halt": O.DAILY_DRAWDOWN_HALT,
                       "max_positions": O.MAX_POSITIONS, "max_aggressive": O.MAX_AGGRESSIVE, "max_per_sector": O.MAX_PER_SECTOR, "min_r": O.MIN_R,
                       "min_stop_atrs": O.MIN_STOP_ATRS, "max_stop_pct": O.MAX_STOP_PCT,
                       "max_limit_deviation": O.MAX_LIMIT_DEVIATION,
                       "earnings_blackout_days": O.EARNINGS_BLACKOUT_DAYS, "reentry_days": O.REENTRY_DAYS}

        # the account
        try:
            account = self.broker().account()
        except Exception as exc:
            pack.warnings.append(f"账户读不到：{type(exc).__name__}: {exc}")
            account = None
        if account is not None:
            pack.cash = _num(account.cash)
            eq, last = (self.broker().equity_today() if hasattr(self.broker(), "equity_today")
                        else (_num(account.account_value), float("nan")))
            pack.equity = eq if _ok(eq) else _num(account.account_value)
            pack.last_equity = last
            if hasattr(self.broker(), "market_open"):
                with contextlib.suppress(Exception):
                    pack.market_open = bool(self.broker().market_open())
        held_syms = [str(h.symbol).upper() for h in (account.holdings or [])] if account else []

        # what the venue did since the last run: a stop that fired closes the book row
        self.reconcile(account, today, pack)
        # a held position the book has lost but the venue still protects is
        # rebuilt from its resting orders, so a lost file never means a lost stop
        self.recover(account, today, pack)

        # regime
        try:
            board = self.market.macro(data_day)
            pack.regime = board.read()
        except Exception as exc:
            pack.warnings.append(f"宏观看板没拿到：{exc}")
        events = self.market.policy()
        try:
            from tradingagents.live.policy import policy_brief, sector_pressure
            pack.policy_brief = policy_brief(events)
            pack.tilt = sector_pressure(events) if events else {}
        except Exception:
            pass

        # names: held + screen leaders + extras
        leaders = universe.screen_leaders(data_day, exchange=self.exchange, top=self.top,
                                          per_sector=self.per_sector, screen=self.screen)
        if not leaders:
            pack.warnings.append("筛选没有跑出结果；候选表只有你自己加的名字")
        names = {n.symbol: n for n in leaders}
        for s in extra:
            s = s.strip().upper()
            if s and s not in names:
                names[s] = universe.Name(symbol=s, sector=universe.sector_of(s), source="claude")
        wanted = list(dict.fromkeys(held_syms + list(names)))
        spy = self.market.facts("SPY", data_day, benchmark=False)
        spy_r3 = _num(spy.snap.ret_3m)
        facts = {s: self.market.facts(s, data_day) for s in wanted}
        alive = [s for s in wanted if facts[s].ok]
        earnings = self.market.earnings(alive, data_day)
        fundamentals = self.market.fundamentals(alive)
        by_symbol, macro_news = self.market.headlines(alive[:40], macro=True)
        pack.macro_news = macro_news[:10]
        for s in alive:
            f = facts[s]
            n = names.get(s)
            if n is not None and n.sector != "Unknown":
                f.sector = n.sector
            self.market.attach(f, earnings=earnings, fundamentals=fundamentals, news=by_symbol)
            if not f.sector or f.sector == "Unknown":
                f.sector = universe.sector_of(s)

        # held rows
        if account is not None:
            resting = {}
            if hasattr(self.broker(), "open_orders"):
                for o in self.broker().open_orders():
                    if o.get("side") == "sell" and o.get("type") in ("stop", "stop_limit"):
                        resting[o["symbol"].upper()] = o
            for h in sorted(account.holdings or [], key=lambda h: str(h.symbol)):
                sym = str(h.symbol).upper()
                f = facts.get(sym)
                px = _num(h.last) if _ok(_num(h.last)) else (f.price if f else float("nan"))
                cost = _num(h.avg_cost)
                row = HeldRow(sym, int(_num(h.quantity, 0.0)), cost, px, _num(h.market_value),
                              px / cost - 1.0 if _ok(cost) and cost > 0 and _ok(px) else float("nan"),
                              protected=sym in resting, in_book=sym in self.book.positions)
                p = self.book.positions.get(sym)
                if p is not None:
                    venue_stop = _num((resting.get(sym) or {}).get("stop_price"))
                    if _ok(venue_stop) and _ok(p.stop) and venue_stop > p.stop + 1e-9:
                        # the venue is the truth for stops: a stale book copy from
                        # another session undid a raise (MSFT 514.73 → 502, 2026-10-14)
                        pack.warnings.append(f"{sym}: 账本止损 {p.stop:,.2f} 低于交易所挂着的 {venue_stop:,.2f}，已按交易所同步")
                        p.stop = venue_stop
                        self.book.save()
                    if getattr(p, "pending", False):
                        # the venue filled it since the order ran: book the real cost
                        p.pending = False
                        if _ok(cost) and cost > 0:
                            p.entry = cost
                        self.book.save()
                    row.stop, row.target, row.opened = p.stop, p.target, p.opened
                    row.sleeve = getattr(p, "sleeve", "core")
                    row.r_now = p.r_at(px) if _ok(px) else float("nan")
                    row.days_left = (p.deadline() - today).days
                    row.thesis, row.invalidation = p.thesis, p.invalidation
                if f is not None and f.ok:
                    row.verdict, row.news = f.trend.verdict, list(f.news)[:4]
                    if f.earnings is not None:
                        with contextlib.suppress(Exception):
                            row.earnings_in = _num(f.earnings.days_to_next(today))
                pack.held.append(row)

        # what the trader tends to miss: resting buys and sleeve rows past 1R
        for sym, p in self.book.positions.items():
            if getattr(p, "pending", False):
                age = sessions_since(p.opened, today) if p.opened else 0
                f = facts.get(sym)
                px = f.price if f is not None and f.ok else _num(self.market.facts(sym, today).price)
                gap = (f"现价高出限价 {(px / p.entry - 1) * 100:+.1f}%" if _ok(px) and p.entry else "现价未知")
                pack.reminders.append(f"⏳ 挂单 {sym}：限价 {p.entry:,.2f}，第 {age} 天，{gap}；"
                                      f"超过 {PENDING_DAYS} 天未成交自动撤单")
        for h in pack.held:
            p = self.book.positions.get(h.symbol)
            if (p is not None and p.sleeve == "aggressive" and _ok(h.r_now) and h.r_now >= 1.0
                    and not getattr(p, "trimmed", False)):
                pack.reminders.append(f"💡 {h.symbol} 已到 {h.r_now:.1f}R 未减半：手册十·先 trim 一半兑现，"
                                      f"剩余止损移到成本并跟 20 日线")

        # candidates
        recent = self.book.recently_closed(today)
        for s in alive:
            if s in held_syms:
                continue
            n = names.get(s)
            f = facts[s]
            row = row_for(f, pack.tilt.get(f.sector, 0.0), spy_r3, source=n.source if n else "",
                          rank=n.screen_rank if n else 0, today=today)
            if s in recent:
                row.verdict = f"（{pack.limits['reentry_days']} 天内刚止损，不可买）" + row.verdict
            pack.candidates.append(row)
        pack.candidates.sort(key=lambda r: (r.ref_score if _ok(r.ref_score) else -999), reverse=True)
        from .report import breakout_triggers
        for s in alive:
            if s in held_syms:
                continue
            f = facts[s]
            t = breakout_triggers(f)
            rs = _num(getattr(f, "rs_1m", float("nan")))
            if any(x in ("volume", "pattern", "rs") for x in t):
                low = f.bars.lows[-1] if f.bars.lows else float("nan")
                s20 = _num(f.snap.sma20)
                pack.breakouts.append({"symbol": s, "triggers": t, "breakout_low": round(_num(low), 2) if _ok(_num(low)) else None,
                                       "sma20": round(s20, 2) if _ok(s20) else None,
                                       "price": round(f.price, 2), "vol_ratio": round(_num(f.snap.vol_ratio), 2),
                                       "rs_1m": round(rs, 4) if _ok(rs) else None,
                                       "ext_200": round(f.ext_200(), 4) if _ok(f.ext_200()) else None})
        pack.breakouts.sort(key=lambda b: (len(b["triggers"]), b.get("rs_1m") or -1), reverse=True)
        self.save(pack, facts)
        return pack

    def recover(self, account, today: date, pack: TradePack) -> None:
        """Rebuild book rows from the venue's resting orders when the file lost them."""
        if account is None or not hasattr(self.broker(), "open_orders"):
            return
        resting: dict[str, dict] = {}
        targets: dict[str, float] = {}
        for o in self.broker().open_orders():
            sym = str(o.get("symbol", "")).upper()
            if o.get("side") != "sell":
                continue
            if o.get("type") in ("stop", "stop_limit"):
                resting[sym] = o
            elif o.get("type") == "limit":
                targets[sym] = _num(o.get("limit_price"))
        held_syms = {str(h.symbol).upper() for h in (account.holdings or [])}
        for o in self.broker().open_orders():
            sym = str(o.get("symbol", "")).upper()
            if (o.get("side") != "buy" or o.get("parent_id") or sym in self.book.positions
                    or sym in held_syms or str(o.get("id", "")) in self._cancelled):
                continue
            # a buy the book does not know (its row was lost): cancel it once stale
            age = sessions_since(str(o.get("submitted_at", ""))[:10], today)
            if age > PENDING_DAYS and hasattr(self.broker(), "cancel"):
                ok = self.broker().cancel(str(o.get("id", "")))
                pack.warnings.append(f"{sym}: 交易所挂着账本没有的买单（{age} 个交易日），"
                                     f"{'已撤单' if ok else '撤单失败'}")
            else:
                pack.warnings.append(f"{sym}: 交易所挂着账本没有的买单（{age} 个交易日），超过 {PENDING_DAYS} 天会撤")
        for h in account.holdings or []:
            sym = str(h.symbol).upper()
            if sym in self.book.positions or sym not in resting:
                continue
            stop = _num(resting[sym].get("stop_price"))
            if not _ok(stop):
                continue
            entry = _num(h.avg_cost)
            target = targets.get(sym, float("nan"))
            if not _ok(target):
                target = round(entry * 1.10, 2) if _ok(entry) else float("nan")
            opened = str(resting[sym].get("submitted_at", ""))[:10] or today.isoformat()
            rec = last_buy_record(sym)   # the decision log remembers what the book lost
            self.book.positions[sym] = Position(
                symbol=sym, shares=int(_num(h.quantity, 0.0)), entry=entry, stop=stop, target=target,
                opened=str(rec.get("date") or opened), adopted=not rec,
                thesis=str(rec.get("thesis") or "（记录丢失，按交易所挂单恢复；论点未知）"),
                invalidation=str(rec.get("invalidation") or ""),
                principles=list(rec.get("principles") or []), regime=str(rec.get("regime") or ""),
                sleeve=str(rec.get("sleeve") or "core"), triggers=list(rec.get("triggers") or []),
                horizon_days=int(SLEEVE.get(str(rec.get("sleeve") or "core"), SLEEVE["core"])["horizon_days"]),
                stop0=_num(rec.get("stop")) if _ok(_num(rec.get("stop"))) else stop,
                legs={"stop": str(resting[sym].get("id", ""))})
            how = f"按 {rec.get('date')} 的下单记录（{rec.get('sleeve') or 'core'}）" if rec else "按交易所挂单"
            pack.warnings.append(f"{sym}: 账本里没有记录，已{how}恢复，止损 {stop:,.2f}")
        if any("已按" in w and "恢复" in w for w in pack.warnings):
            self.book.save()

    @property
    def _cancelled(self) -> set:
        if not hasattr(self, "_cancelled_ids"):
            self._cancelled_ids = set()
        return self._cancelled_ids

    def _open_buy(self, symbol: str) -> dict | None:
        """The venue's resting buy order for ``symbol``, if any."""
        if not hasattr(self.broker(), "open_orders"):
            return None
        for o in self.broker().open_orders():
            if str(o.get("symbol", "")).upper() == symbol and o.get("side") == "buy" \
                    and not o.get("parent_id"):
                return o
        return None

    def reconcile(self, account, today: date, pack: TradePack) -> None:
        """Book positions the account no longer holds were closed by the venue."""
        if account is None:
            return
        held = {str(h.symbol).upper(): int(_num(h.quantity, 0.0)) for h in (account.holdings or [])}
        gone = [s for s in self.book.positions if held.get(s, 0) <= 0]
        fills = []
        if gone and hasattr(self.broker(), "fills_since"):
            since = datetime.now(clock.ET) - timedelta(days=45)
            fills = self.broker().fills_since(since)
        pack.fills = [f for f in fills if f.get("side") == "sell"][:20]
        for s in gone:
            p = self.book.positions[s]
            if getattr(p, "pending", False):
                # never filled: not a trade, so no closed record and no post-mortem.
                # A pullback limit may rest a couple of sessions; after that it is
                # cancelled at the venue too (PBR 2026-10-09 rested for weeks).
                resting = self._open_buy(s)
                age = sessions_since(p.opened, today) if p.opened else 99
                if resting and age <= PENDING_DAYS:
                    continue
                if resting and hasattr(self.broker(), "cancel"):
                    cancelled = self.broker().cancel(str(resting.get("id") or p.parent_id))
                    self._cancelled.add(str(resting.get("id") or p.parent_id))
                    how = "已撤单并" if cancelled else "撤单失败，仍"
                    pack.warnings.append(f"{s}: 买单挂了 {age} 天未成交，{how}从账本移除（不是交易，不复盘）")
                else:
                    pack.warnings.append(f"{s}: 买单未成交（交易所已无此单），已从账本移除（不是交易，不复盘）")
                del self.book.positions[s]
                self.book.save()
                continue
            sold = [f for f in fills if f["symbol"].upper() == s and f["side"] == "sell"]
            exit_px = _num(sold[-1]["price"]) if sold else _num(self.market.facts(s, today).price)
            kind = (sold[-1].get("type") or "") if sold else ""
            reason = ("stop: 交易所触发止损" if kind.startswith("stop") else
                      "target: 交易所触发止盈" if kind == "limit" else "closed at the venue")
            self.book.close(s, exit_px if _ok(exit_px) else p.stop, today, reason)
        pack.postmortems = [asdict(c) for c in self.book.unreviewed()]
        for c in pack.postmortems:
            try:
                raw, alpha = review.alpha_between(self.market, c["symbol"], date.fromisoformat(c["opened"]),
                                                  date.fromisoformat(c["closed"]), today)
            except Exception:
                raw, alpha = float("nan"), float("nan")
            c["raw"], c["alpha"] = (raw if _ok(raw) else None), (alpha if _ok(alpha) else None)

    def save(self, pack: TradePack, facts: dict) -> None:
        d = task_dir(TASK)
        path = d / f"{pack.date}-pack.md"
        try:
            path.write_text(format_pack(pack, facts), encoding="utf-8")
            (d / f"{pack.date}-pack.json").write_text(
                json.dumps(pack.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            pack.path = str(path)
        except OSError as exc:
            pack.warnings.append(f"the pack could not be saved: {exc}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _f(v, d=2):
    return f"{v:,.{d}f}" if _ok(v) else "—"


def _pct(v, d=1):
    return f"{v * 100:+.{d}f}%" if _ok(v) else "—"


def table(rows: list[Row], zh=True) -> list[str]:
    out = ["| 代码 | 板块 | 现价 | 日 | 月 | 三月 | RSI | 量比 | ATR% | 距200日 | 支撑 | 参考入场 | 参考止损 | 参考目标 | R | 财报 | 参考分 | 来源 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in rows:
        earn = f"{r.earnings_in:.0f}d" if _ok(r.earnings_in) else "—"
        out.append(f"| {r.symbol} | {SECTOR_ZH.get(r.sector, r.sector)} | {_f(r.price)} | {_pct(r.change_pct)} "
                   f"| {_pct(r.ret_1m)} | {_pct(r.ret_3m)} | {_f(r.rsi, 0)} | {_f(r.vol_ratio, 1)} "
                   f"| {_pct(r.atr_pct)} | {_pct(r.ext_200, 0)} | {_f(r.support)} | {_f(r.entry)} | {_f(r.stop)} "
                   f"| {_f(r.target)} | {_f(r.r, 1)} | {earn} | {_f(r.ref_score, 0)} | {r.source}{'#' + str(r.screen_rank) if r.screen_rank else ''} |")
    return out


def format_pack(pack: TradePack, facts: dict | None = None) -> str:
    L = pack.limits
    out = [f"# 交易台数据包 · {pack.date}", "",
           f"数据截至 {pack.data_date} 收盘。市场{'开盘中' if pack.market_open else '未开盘'}。"
           f"净值 ${_f(pack.equity)}（昨收 ${_f(pack.last_equity)}），现金 ${_f(pack.cash)}。", ""]
    if pack.warnings:
        out += [f"- ⚠ {w}" for w in pack.warnings] + [""]
    out += ["## 硬限制（代码执行，见手册 §3）",
            f"单笔风险 {L.get('risk_pct', 0) * 100:.0f}% · 单票 ≤ {L.get('max_name_weight', 0) * 100:.0f}% · "
            f"单日新增风险 ≤ {L.get('max_daily_new_risk', 0) * 100:.0f}% · 当日回撤 {L.get('daily_drawdown_halt', 0) * 100:.0f}% 熔断 · "
            f"最多 {L.get('max_positions', 0)} 仓（主 {L.get('max_positions', 0) - L.get('max_aggressive', 0)} + 进攻 {L.get('max_aggressive', 0)}） · 同板块 ≤ {L.get('max_per_sector', 0)} · R ≥ {L.get('min_r', 0):.1f} · "
            f"止损 ≥ {L.get('min_stop_atrs', 0):.0f} ATR 且 ≤ {L.get('max_stop_pct', 0) * 100:.0f}% · "
            f"财报前 {L.get('earnings_blackout_days', 0)} 天不开仓 · 止损后 {L.get('reentry_days', 0)} 天不回头", ""]

    out.append("## 需要复盘")
    if pack.postmortems:
        for c in pack.postmortems:
            out.append(f"- **{c['symbol']}** {c['opened']} → {c['closed']}，{c['shares']} 股 @{_f(c['entry'])} → {_f(c['exit'])}，"
                       f"盈亏 {_f(c['pnl'])}（{_f(c['r'], 1)}R），{c['reason']}"
                       + (f"；同期相对标普 {_pct(c['alpha'])}" if c.get("alpha") is not None else ""))
            if c.get("thesis"):
                out.append(f"  - 当初论点：{c['thesis']}")
            if c.get("invalidation"):
                out.append(f"  - 失效条件：{c['invalidation']}")
    else:
        out.append("- 没有")
    out.append("")

    out.append(f"## 持仓（{len(pack.held)}/{L.get('max_positions', MAX_POSITIONS)}）")
    if pack.held:
        out += ["| 代码 | 账 | 股数 | 成本 | 现价 | 盈亏 | 止损 | 目标 | 当前R | 剩余天数 | 止损挂着 | 财报 | 图形 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for h in pack.held:
            earn = f"{h.earnings_in:.0f}d" if _ok(h.earnings_in) else "—"
            out.append(f"| {h.symbol} | {'进攻' if h.sleeve == 'aggressive' else '主'} | {h.shares} | {_f(h.avg_cost)} | {_f(h.price)} | {_pct(h.pnl_pct)} | {_f(h.stop)} "
                       f"| {_f(h.target)} | {_f(h.r_now, 1)} | {h.days_left if h.days_left is not None else '—'} "
                       f"| {'是' if h.protected else '**否**'} | {earn} | {h.verdict[:40]} |")
        for h in pack.held:
            if h.thesis or h.invalidation or h.news:
                out.append(f"- **{h.symbol}**" + (f" 论点：{h.thesis}" if h.thesis else "")
                           + (f"；失效条件：{h.invalidation}" if h.invalidation else ""))
                for n in h.news[:3]:
                    out.append(f"  - 新闻：[{n.title}]({n.link})（{n.source}，{n.lean}）")
        unprotected = [h.symbol for h in pack.held if not h.protected]
        if unprotected:
            out.append(f"- ⚠ 没有止损挂在交易所的仓位：{', '.join(unprotected)} — 用 protect 单给它们止损和目标")
    else:
        out.append("空仓。")
    for r in pack.reminders:
        out.append(f"- {r}")
    out.append("")

    out.append("## 环境")
    out += [f"- {r}" for r in pack.regime] or ["- 宏观看板没拿到"]
    if pack.tilt:
        tilted = {k: v for k, v in pack.tilt.items() if abs(v) >= 0.1}
        if tilted:
            out.append("- 政策倾向：" + "，".join(f"{SECTOR_ZH.get(k, k)} {v:+.2f}" for k, v in sorted(tilted.items(), key=lambda kv: -abs(kv[1]))))
    for n in pack.macro_news[:6]:
        out.append(f"- [{n.title}]({n.link})（{n.source}）")
    out.append("")

    out.append("## 进攻仓候选（突破跟踪，手册第十节）")
    if pack.breakouts:
        out += ["| 代码 | 现价 | 量比 | 20 日相对标普 | 距200日 | 图上看到的触发 | 突破日低（止损位） | 20 日线 |", "|---|---:|---:|---:|---:|---|---:|---:|"]
        for b in pack.breakouts[:12]:
            out.append(f"| {b['symbol']} | {_f(b['price'])} | {_f(b['vol_ratio'], 1)} | {_pct(b.get('rs_1m'))} | {_pct(b.get('ext_200'), 0)} "
                       f"| {'、'.join(b['triggers']) or '—'} | {_f(b['breakout_low'])} | {_f(b.get('sma20'))} |")
        out.append("四条触发（volume / catalyst / pattern / rs）至少两条才能进；`catalyst?` 只是说有分量 ≥7 的利好标题，是否算重大催化剂你来判断，并在 intent 里写明。"
                   "止损放在突破日低点或 20 日线，二者取高、不超过 8%。拉伸不是这本账的禁忌（手册第十节）。")
    else:
        out.append("- 今天没有放量创新高或突破平台的候选。")
    out.append("")
    out.append(f"## 候选（{len(pack.candidates)}）")
    out += table(pack.candidates)
    out.append("")
    for r in pack.candidates:
        if r.news:
            out.append(f"- **{r.symbol}**：" + "；".join(f"[{n.title}]({n.link})" for n in r.news[:3]))
    out.append("")
    if facts:
        out.append("## 图")
        for r in pack.candidates[:12]:
            f = facts.get(r.symbol)
            if f is None:
                continue
            chart = chart_for(Idea(symbol=r.symbol, stop=r.stop, target=r.target, facts=f))
            if chart:
                out += ["```"] + chart + ["```", f"{r.symbol}：{r.verdict}", ""]
    return "\n".join(out)


def facts_table(symbols: list[str], *, market: Market | None = None, now: datetime | None = None,
                with_charts: bool = True) -> str:
    """The candidate table for a hand-picked list of names, plus their charts."""
    m = market or Market(task=TASK)
    now = now or datetime.now(clock.ET)
    today, data_day = clock.market_state(now).trading_day, last_completed_session(now)
    syms = [s.strip().upper() for s in symbols if s.strip()]
    spy = m.facts("SPY", data_day, benchmark=False)
    facts = {s: m.facts(s, data_day) for s in syms}
    alive = [s for s in syms if facts[s].ok]
    earnings = m.earnings(alive, data_day)
    fundamentals = m.fundamentals(alive)
    by_symbol, _ = m.headlines(alive, macro=False)
    rows = []
    for s in alive:
        f = facts[s]
        m.attach(f, earnings=earnings, fundamentals=fundamentals, news=by_symbol)
        if not f.sector or f.sector == "Unknown":
            f.sector = universe.sector_of(s)
        rows.append(row_for(f, 0.0, _num(spy.snap.ret_3m), source="claude", today=today))
    out = [f"# 补充名字 · {today.isoformat()}", ""] + table(rows) + [""]
    for s in syms:
        if s not in alive:
            out.append(f"- ⚠ {s}: 拿不到行情")
    for r in rows:
        for n in r.news[:3]:
            out.append(f"- {r.symbol}：[{n.title}]({n.link})（{n.source}，{n.lean}）")
    if with_charts:
        for r in rows:
            chart = chart_for(Idea(symbol=r.symbol, stop=r.stop, target=r.target, facts=facts[r.symbol]))
            if chart:
                out += ["", "```"] + chart + ["```", f"{r.symbol}：{r.verdict}"]
    text = "\n".join(out) + "\n"
    try:
        with open(task_dir(TASK) / f"{today.isoformat()}-extra.md", "a", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass
    return text


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk pack", description="the tables Claude reads before deciding")
    p.add_argument("what", choices=["trade", "facts"])
    p.add_argument("symbols", nargs="?", default="", help="for facts: A,B,C")
    p.add_argument("--add", default="", help="for trade: extra names A,B,C")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from . import state
    print(state.pull())
    if args.what == "facts":
        print(facts_table([s for s in args.symbols.split(",") if s]))
        return 0
    pack = Packer(top=args.top).run([s for s in args.add.split(",") if s])
    print(pack.path if args.quiet else format_pack(pack))
    # The reconciled and recovered book goes to the branch at once. 2026-10-13:
    # the pack rebuilt a lost row, the next command's pull wiped it again.
    print(state.push(message="pack trade"))
    return 0
