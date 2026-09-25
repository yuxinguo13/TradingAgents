"""Task 1 — trade the paper account, every session, and say where it stands.

Rules, not opinions. The run is the same every day:

1. Read the Alpaca account. Any position the desk did not open (or has no
   record of) is *adopted*: given a stop and a target so that from now on it
   is managed like the rest. Nothing in the account is ever left unmanaged.
2. Review every position against its levels: stop hit, target reached (take
   half), out of time, thesis broken by the news, stop raised to breakeven
   once the trade is up one R. Sells go to the venue first.
3. With the slots and cash left, open new positions from the top of the
   momentum screen: in an uptrend on both the 50- and 200-day line, no
   earnings inside the holding period, a stop under the structure, a target
   from the trend, and at least ``min_r`` reward for each unit of risk. Sized
   so a stop-out costs ``risk_pct`` of equity.
4. Every order goes through the risk gate (``live.secretary``) — position
   caps, daily turnover, price sanity — and is recorded in a ledger.
5. Write the note: the portfolio, today's orders, and the plan for each
   position. No analysis; the reasons are one line each.

State lives under ``$TRADINGAGENTS_HOME/desk/trade/``: ``book.json`` (the
positions and their levels), ``ledger.json`` (every order attempted), and one
``<date>.md`` per run. The kill switch ``$TRADINGAGENTS_HOME/STOP`` stops the
run before it reads the account.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timedelta

from tradingagents.live import clock
from tradingagents.live.advisor import last_completed_session, limit_price, project_target
from tradingagents.live.broker import BUY, LIMIT, MARKET, SELL, Account
from tradingagents.live.recommendations import (
    CLOSED,
    RAISE_STOP,
    TRIM,
    UNAVAILABLE,
    ExitRules,
    RecommendationBook,
)
from tradingagents.live.secretary import (
    Order,
    RiskLimits,
    Secretary,
    TradeLedger,
    kill_switch_engaged,
)
from tradingagents.live.sizing import size_position, stop_from_atr, structural_stop

from . import task_dir, universe
from .market import Market, _num, _ok

logger = logging.getLogger(__name__)

TASK = "trade"


@dataclass
class TradeConfig:
    risk_pct: float = 1.0           # % of equity lost if a stop fills exactly
    cap_fraction: float = 0.10      # no single new position above this share of equity
    max_positions: int = 8
    horizon_days: int = 30
    min_r: float = 1.5
    min_price: float = 5.0
    exchange: str = "all"
    top: int = 40                   # screen rows considered
    per_sector: int = 2             # new entries per sector from one screen
    reentry_days: int = 10          # a name just closed is not re-bought for this long
    limit_buffer: float = 0.005     # buy limit this far through the last close
    venue: str = "alpaca"
    dry_run: bool = False
    no_entries: bool = False

    @classmethod
    def from_env(cls) -> TradeConfig:
        cfg = cls()
        for f in fields(cls):
            raw = os.getenv(f"TRADINGAGENTS_DESK_{f.name.upper()}")
            if raw is None:
                continue
            cur = getattr(cfg, f.name)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, f.name, raw.strip().lower() in ("1", "true", "yes"))
                elif isinstance(cur, int):
                    setattr(cfg, f.name, int(raw))
                elif isinstance(cur, float):
                    setattr(cfg, f.name, float(raw))
                else:
                    setattr(cfg, f.name, raw)
            except ValueError:
                logger.warning("ignoring TRADINGAGENTS_DESK_%s=%r", f.name.upper(), raw)
        return cfg


@dataclass
class Placed:
    symbol: str
    action: str
    shares: int
    price: float
    reason: str
    ok: bool = False
    message: str = ""
    order_type: str = MARKET


@dataclass
class PositionLine:
    symbol: str
    shares: float
    avg_cost: float
    last: float
    market_value: float
    unrealized: float
    stop: float = float("nan")
    target: float = float("nan")
    opened: str = ""
    days_left: int | None = None

    @property
    def pnl_pct(self) -> float:
        return (self.last / self.avg_cost - 1.0) if self.avg_cost else float("nan")


@dataclass
class TradeNote:
    date: str
    generated_at: str = ""
    venue: str = ""
    dry_run: bool = False
    account_value: float = float("nan")
    cash: float = float("nan")
    positions: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    refused: str = ""

    @property
    def invested(self) -> float:
        return sum(p.market_value for p in self.positions if _ok(p.market_value))

    @property
    def unrealized(self) -> float:
        return sum(p.unrealized for p in self.positions if _ok(p.unrealized))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["invested"], d["unrealized"] = self.invested, self.unrealized
        return d


# ---------------------------------------------------------------------------
# the trader
# ---------------------------------------------------------------------------

class Trader:
    def __init__(self, cfg: TradeConfig | None = None, *, broker=None, market: Market | None = None,
                 book: RecommendationBook | None = None, secretary: Secretary | None = None,
                 screen=None, now: datetime | None = None, rules: ExitRules | None = None):
        self.cfg = cfg or TradeConfig.from_env()
        self._broker = broker
        self.market = market or Market(task=TASK)
        self.book = book or RecommendationBook(path=task_dir(TASK) / "book.json")
        self.secretary = secretary or Secretary(
            RiskLimits.from_env(), TradeLedger(task_dir(TASK) / "ledger.json"),
            venue=self.cfg.venue)
        self.screen = screen
        self.now = now
        self.today: date | None = None
        self.rules = rules or ExitRules()

    # --- venue --------------------------------------------------------------

    def broker(self):
        if self._broker is None:
            from tradingagents.live.broker import open_broker
            self._broker = open_broker(self.cfg.venue)
        return self._broker

    def quote(self, symbol: str, fallback: float) -> float:
        try:
            px = _num(self.broker().quote(symbol))
        except Exception:
            px = float("nan")
        return px if _ok(px) and px > 0 else fallback

    def market_open(self) -> bool:
        b = self.broker()
        if hasattr(b, "market_open"):
            try:
                return bool(b.market_open())
            except Exception:
                pass
        return clock.market_state(self.now).is_open

    # --- the run --------------------------------------------------------------

    def run(self) -> TradeNote:
        now = self.now or datetime.now(clock.ET)
        today = self.today = clock.market_state(now).trading_day
        note = TradeNote(date=today.isoformat(), generated_at=datetime.now().isoformat(),
                         venue=self.cfg.venue, dry_run=self.cfg.dry_run)
        if kill_switch_engaged():
            note.refused = "kill switch engaged ($TRADINGAGENTS_HOME/STOP exists); nothing read, nothing placed"
            return self.finish(note)
        try:
            account = self.broker().account()
        except Exception as exc:
            note.refused = f"the account could not be read ({type(exc).__name__}: {exc})"
            return self.finish(note)
        note.account_value, note.cash = _num(account.account_value), _num(account.cash)
        data_date = last_completed_session(now)

        self.adopt(account, data_date, note)
        account = self.exits(account, today, data_date, note)
        if not self.cfg.no_entries:
            self.entries(account, today, data_date, note)
        self.describe(account, today, note)
        return self.finish(note)

    # --- 1. adopt -------------------------------------------------------------

    def adopt(self, account: Account, data_date: date, note: TradeNote) -> None:
        managed = {r.symbol for r in self.book.open_recommendations()}
        for h in account.holdings or []:
            sym = str(getattr(h, "symbol", "") or "").upper()
            qty = _num(getattr(h, "quantity", 0), 0.0)
            if not sym or qty <= 0 or sym in managed:
                continue
            if str(getattr(h, "side", "long") or "long").lower() != "long":
                note.notes.append(f"{sym}: 空头仓位，本台只管多头，未接管")
                continue
            f = self.market.facts(sym, data_date)
            # The levels are set from where the name *is*, not from what was
            # paid for it: a stop under the average cost of a position that has
            # since doubled is no stop at all, and one above today's price is
            # an instant sell. The cost stays on the account for the P&L.
            entry = f.price if f.ok else _num(getattr(h, "last", None))
            if not _ok(entry) or entry <= 0:
                note.notes.append(f"{sym}: 账户里有但拿不到行情，无法设止损，暂未接管")
                continue
            stop = self.stop_for(entry, f)
            target = project_target(entry, f.snap, self.cfg.horizon_days) if f.ok else None
            if not _ok(stop) or stop <= 0 or stop >= entry:
                note.notes.append(f"{sym}: 账户里有但算不出止损，暂未接管")
                continue
            if target is None or target <= entry:
                target = round(entry * 1.10, 2)
            self.book.add(sym, "BUY", int(qty), entry, stop, target,
                          horizon_days=self.cfg.horizon_days, sector=f.sector,
                          issued_date=self.today,
                          rationale="接管账户中已有的仓位，按本台规则设止损和目标")
            managed.add(sym)
            note.notes.append(f"{sym}: 接管账户中已有的 {qty:g} 股，止损 {stop:,.2f}，目标 {target:,.2f}")

    def stop_for(self, entry: float, f) -> float:
        atr_pct = _num(getattr(f.snap, "atr_pct", None))
        support = f.support()
        stop = None
        if _ok(atr_pct) and atr_pct > 0:
            stop = structural_stop(entry, support if _ok(support) else None, atr_pct=atr_pct)
            if stop is None:
                stop = stop_from_atr(entry, atr_pct=atr_pct)
        return _num(stop)

    # --- 2. exits -------------------------------------------------------------

    def exits(self, account: Account, today: date, data_date: date, note: TradeNote) -> Account:
        open_recs = self.book.open_recommendations()
        if not open_recs:
            return account
        held = {str(h.symbol).upper(): _num(h.quantity, 0.0) for h in (account.holdings or [])}
        prices = {r.symbol: self.quote(r.symbol, self.market.facts(r.symbol, data_date).price)
                  for r in open_recs}
        news_by_symbol, _ = self.market.headlines([r.symbol for r in open_recs], macro=False)
        signals = self.book.review(prices, news_by_symbol, as_of=today, rules=self.rules)
        market_open = self.market_open()
        for sig in signals:
            if sig.action == UNAVAILABLE:
                note.notes.append(f"{sig.symbol}: {sig.reason}")
                continue
            if sig.action == RAISE_STOP:
                note.notes.append(f"{sig.symbol}: 止损上移到 {sig.new_stop:,.2f}（{sig.reason}）")
                continue
            have = held.get(sig.symbol, 0.0)
            shares = int(min(sig.shares, have)) if have > 0 else 0
            if shares <= 0:
                note.notes.append(f"{sig.symbol}: 账本要卖 {sig.shares} 股，账户里没有，按已离场处理")
                if sig.closes_position:
                    self.book.close_from(sig)
                continue
            label = "清仓" if sig.closes_position else f"减仓 {shares} 股"
            placed = self.place(Order(sig.symbol, SELL, shares, MARKET, None,
                                      rationale=sig.reason, source="desk"),
                                account, prices.get(sig.symbol, 0.0), market_open,
                                reason=f"{label}：{sig.reason}")
            note.orders.append(placed)
            if placed.ok and sig.closes_position:
                self.book.close_from(sig)
                account = self.after_sell(account, sig.symbol, shares, placed.price)
            elif placed.ok and sig.action == TRIM:
                account = self.after_sell(account, sig.symbol, shares, placed.price)
        return account

    @staticmethod
    def after_sell(account: Account, symbol: str, shares: int, price: float) -> Account:
        """The account as the venue will report it once the sell fills."""
        holdings = []
        for h in account.holdings or []:
            if str(h.symbol).upper() == symbol:
                left = _num(h.quantity, 0.0) - shares
                if left > 0:
                    h.quantity = left
                    holdings.append(h)
            else:
                holdings.append(h)
        cash = _num(account.cash, 0.0) + shares * price
        return Account(account_value=account.account_value, cash=cash,
                       buying_power=max(_num(account.buying_power, 0.0), cash),
                       holdings=holdings, fetched_at=getattr(account, "fetched_at", ""))

    # --- 3. entries -----------------------------------------------------------

    def entries(self, account: Account, today: date, data_date: date, note: TradeNote) -> None:
        open_syms = {r.symbol for r in self.book.open_recommendations()}
        slots = self.cfg.max_positions - len(open_syms)
        if slots <= 0:
            note.plan.append(f"仓位已满（{len(open_syms)}/{self.cfg.max_positions}），今天不开新仓")
            return
        spendable = [v for v in (_num(account.cash), _num(account.buying_power)) if _ok(v) and v > 0]
        budget = min(spendable) if spendable else 0.0
        if budget < 500:
            note.plan.append(f"可用现金 ${budget:,.0f}，不够开新仓")
            return
        recent = self.recently_closed(today)
        leaders = universe.screen_leaders(data_date, exchange=self.cfg.exchange, top=self.cfg.top,
                                         per_sector=self.cfg.per_sector, screen=self.screen)
        if not leaders:
            note.plan.append("筛选没有跑出结果，今天不开新仓")
            return
        held = {str(h.symbol).upper() for h in (account.holdings or [])}
        picks = []
        earnings = self.market.earnings([n.symbol for n in leaders], data_date)
        for n in leaders:
            sym = n.symbol
            if sym in open_syms or sym in held or sym in recent:
                continue
            f = self.market.facts(sym, data_date)
            f.sector = f.sector or n.sector
            why = self.qualify(f, earnings.get(sym), today)
            if why:
                logger.debug("%s skipped: %s", sym, why)
                continue
            entry = f.price
            stop = self.stop_for(entry, f)
            target = project_target(entry, f.snap, self.cfg.horizon_days)
            if not _ok(stop) or stop <= 0 or stop >= entry or target is None:
                continue
            r = (target - entry) / (entry - stop)
            if r < self.cfg.min_r:
                continue
            sized = size_position(_num(account.account_value, 0.0), entry, stop,
                                  risk_pct=self.cfg.risk_pct, cap_fraction=self.cfg.cap_fraction)
            if not sized:
                continue
            picks.append((r, n, f, entry, stop, target, sized.quantity))
        picks.sort(key=lambda p: p[0], reverse=True)

        market_open = self.market_open()
        opened = 0
        for r, n, f, entry, stop, target, qty in picks:
            if opened >= slots:
                break
            limit = limit_price(entry, self.cfg.limit_buffer) or entry
            if qty * limit > budget:
                qty = int(budget // limit)
            if qty <= 0:
                continue
            reason = f"新开仓：{f.trend.ma_stack or '趋势向上'}，止损 {stop:,.2f}，目标 {target:,.2f}（{r:.1f}R）"
            placed = self.place(Order(n.symbol, BUY, qty, LIMIT, limit,
                                      rationale=reason, source="desk"),
                                account, entry, market_open, reason=reason)
            note.orders.append(placed)
            if not placed.ok:
                continue
            self.book.add(n.symbol, "BUY", placed.shares, entry, stop, target,
                          limit_price=limit, horizon_days=self.cfg.horizon_days,
                          sector=f.sector or n.sector, rationale=reason,
                          issued_date=today)
            budget -= placed.shares * limit
            opened += 1
        if opened == 0 and not any(o.action == BUY for o in note.orders):
            note.plan.append(f"筛选前 {len(leaders)} 名里没有同时满足趋势、止损、{self.cfg.min_r:.1f}R 的名字，今天不开新仓")

    def qualify(self, f, earnings, today: date) -> str:
        if not f.ok or f.price < self.cfg.min_price:
            return "no data or too cheap"
        if f.above("sma50") is not True or f.above("sma200") is not True:
            return "not above both moving averages"
        rsi = _num(f.snap.rsi14)
        if _ok(rsi) and rsi > 80:
            return "overbought"
        ext = f.ext_200()
        if _ok(ext) and ext > 0.60:
            return "too far above the 200-day line"
        if earnings is not None:
            try:
                if earnings.reports_within(self.cfg.horizon_days, today):
                    return "earnings inside the horizon"
            except Exception:
                pass
        return ""

    def recently_closed(self, today: date) -> set[str]:
        out = set()
        for r in self.book.recommendations:
            if r.status != CLOSED or not r.exit_date:
                continue
            try:
                exited = date.fromisoformat(str(r.exit_date))
            except ValueError:
                continue
            if (today - exited).days < self.cfg.reentry_days:
                out.add(r.symbol)
        return out

    # --- the venue, through the gate ------------------------------------------

    def place(self, order: Order, account: Account, price: float, market_open: bool,
              *, reason: str) -> Placed:
        placed = Placed(order.symbol, order.action, order.quantity,
                        order.limit_price or price, reason, order_type=order.order_type)
        verdict = self.secretary.check(order, account, price,
                                       market_open=market_open or self.cfg.dry_run)
        if not verdict.ok:
            placed.message = f"风控拒绝：{verdict.reason}"
            return placed
        order = verdict.order or order
        placed.shares = order.quantity
        if self.cfg.dry_run:
            placed.ok, placed.message = True, "dry run（未发单）"
            return placed
        try:
            result = self.broker().place_order(order.symbol, order.action, order.quantity,
                                               order_type=order.order_type,
                                               limit_price=order.limit_price)
        except Exception as exc:
            placed.message = f"下单失败：{type(exc).__name__}: {exc}"
            return placed
        placed.ok = bool(getattr(result, "ok", False))
        placed.message = str(getattr(result, "message", "") or "")
        fill = _num(getattr(result, "filled_avg_price", None))
        if _ok(fill) and fill > 0:
            placed.price = fill
        try:
            self.secretary.ledger.record(order, price, placed.ok, placed.message,
                                         venue=self.cfg.venue)
        except Exception as exc:
            logger.warning("the ledger was not written: %s", exc)
        return placed

    # --- 4. the note ------------------------------------------------------------

    def describe(self, account: Account, today: date, note: TradeNote) -> None:
        recs = {r.symbol: r for r in self.book.open_recommendations()}
        pending = {o.symbol for o in note.orders if o.ok and o.action == BUY}
        for h in sorted(account.holdings or [], key=lambda h: str(h.symbol)):
            sym = str(h.symbol).upper()
            r = recs.get(sym)
            line = PositionLine(sym, _num(h.quantity, 0.0), _num(h.avg_cost), _num(h.last),
                                _num(h.market_value), _num(h.unrealized))
            if r is not None:
                line.stop, line.target = _num(r.stop_price), _num(r.target_price)
                line.opened = str(r.issued_date or "")
                try:
                    deadline = date.fromisoformat(str(r.issued_date)) + timedelta(days=int(r.horizon_days))
                    line.days_left = (deadline - today).days
                except (TypeError, ValueError):
                    pass
            note.positions.append(line)
        for sym in sorted(pending):
            r = recs.get(sym)
            if r is None:
                continue
            note.positions.append(PositionLine(sym, r.shares, _num(r.limit_price or r.reference_price),
                                               _num(r.reference_price), r.shares * _num(r.reference_price),
                                               0.0, _num(r.stop_price), _num(r.target_price),
                                               str(r.issued_date), int(r.horizon_days)))
        for p in note.positions:
            if not _ok(p.stop):
                continue
            left = f"，还剩 {p.days_left} 天" if p.days_left is not None else ""
            note.plan.append(f"{p.symbol}: 跌破 {p.stop:,.2f} 清仓，到 {p.target:,.2f} 先卖一半{left}")
        free = self.cfg.max_positions - len(recs)
        if free > 0:
            note.plan.append(f"还有 {free} 个仓位空着，明天继续从筛选榜上找")

    def finish(self, note: TradeNote) -> TradeNote:
        try:
            d = task_dir(TASK)
            (d / f"{note.date}.md").write_text(format_note(note), encoding="utf-8")
            with open(d / "runs.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(note.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            note.notes.append(f"the note could not be saved: {exc}")
        return note


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _money(v: float) -> str:
    return f"${v:,.2f}" if _ok(v) else "—"


def format_note(note: TradeNote) -> str:
    out = [f"# 交易台 · {note.date}" + ("（dry run）" if note.dry_run else ""), ""]
    if note.refused:
        out += [f"**没有交易：** {note.refused}", ""]
        return "\n".join(out)
    total = note.account_value
    out += [f"账户净值 {_money(total)} · 现金 {_money(note.cash)} · 持仓市值 {_money(note.invested)}"
            f" · 浮动盈亏 {_money(note.unrealized)}", ""]
    if note.positions:
        out += ["| 代码 | 股数 | 成本 | 现价 | 盈亏 | 止损 | 目标 | 剩余天数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for p in note.positions:
            pnl = f"{p.pnl_pct * 100:+.1f}%" if _ok(p.pnl_pct) else "—"
            out.append(f"| {p.symbol} | {p.shares:g} | {_money(p.avg_cost)} | {_money(p.last)} | {pnl} "
                       f"| {_money(p.stop)} | {_money(p.target)} | {p.days_left if p.days_left is not None else '—'} |")
        out.append("")
    else:
        out += ["目前空仓。", ""]
    out.append("## 今天的单")
    if note.orders:
        for o in note.orders:
            mark = "✓" if o.ok else "✗"
            kind = "限价" if o.order_type == LIMIT else "市价"
            out.append(f"- {mark} {o.action} {o.symbol} {o.shares} 股 @{kind} {_money(o.price)} — {o.reason}"
                       + (f"（{o.message}）" if o.message and not o.ok else ""))
    else:
        out.append("- 没有下单")
    out += ["", "## 计划"]
    out += [f"- {p}" for p in note.plan] or ["- 按兵不动"]
    if note.notes:
        out += ["", "## 备注"] + [f"- {n}" for n in note.notes]
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk trade",
                                description="paper-trade the account by rule, then say where it stands")
    p.add_argument("--dry-run", action="store_true", help="decide and report, place nothing")
    p.add_argument("--no-entries", action="store_true", help="manage exits only")
    p.add_argument("--venue", default=None, help="alpaca (default) | paper")
    p.add_argument("--json", action="store_true", help="print the note as JSON too")
    args = p.parse_args(argv)
    cfg = TradeConfig.from_env()
    cfg.dry_run = cfg.dry_run or args.dry_run
    cfg.no_entries = cfg.no_entries or args.no_entries
    if args.venue:
        cfg.venue = args.venue
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    note = Trader(cfg).run()
    print(format_note(note))
    if args.json:
        print(json.dumps(note.to_dict(), ensure_ascii=False, indent=2))
    return 2 if note.refused else 0
