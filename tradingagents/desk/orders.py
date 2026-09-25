"""The hands and the gate: what the desk's Claude may ask the venue to do.

Claude writes an *intent* file — symbols, levels, reasons — and this module
turns it into orders, or refuses. The division is deliberate and the manual
(``MANUAL.md``, §3) states it: Claude decides *where* (entry, stop, target)
and *why*; this file decides *how many* and *whether at all*.

    python -m tradingagents.desk order intent.json [--dry-run]
    python -m tradingagents.desk log entry.json

The intent file::

    {"date": "2026-09-25", "orders": [
      {"action": "buy", "symbol": "NVDA", "entry": 141.7, "stop": 139.4, "target": 150.3,
       "thesis": "...", "invalidation": "...", "principles": ["一·多头排列", "二·回调到20日线"]},
      {"action": "sell", "symbol": "XOM", "reason": "论点失效：..."},
      {"action": "trim", "symbol": "AAPL", "fraction": 0.5, "reason": "到目标先卖一半"},
      {"action": "raise_stop", "symbol": "AAPL", "stop": 130.0, "reason": "1R，移到成本"},
      {"action": "protect", "symbol": "MSFT", "stop": 400.0, "target": 460.0, "reason": "接管账户里没止损的仓位"}
    ]}

Every buy is a bracket order (stop and take-profit attached, GTC) so the exit
lives at the venue. A stop can be raised, never lowered, never cancelled; the
only cancel in this module is inside ``sell``, where the position it protected
is flattened in the same step. Every order, placed or refused, is appended to
``desk/trade/decisions.jsonl`` with the reasons Claude gave and the verdict the
gate returned.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from tradingagents.live import clock
from tradingagents.live.secretary import kill_switch_engaged

from . import task_dir
from .market import Market, _num, _ok

logger = logging.getLogger(__name__)

TASK = "trade"

# --- the limits (MANUAL.md §3). Change them here and in the manual together. --
RISK_PCT = 0.01                 # a stop-out costs this share of equity
MAX_NAME_WEIGHT = 0.10          # one symbol's share of equity, after the fill
MAX_DAILY_NEW_RISK = 0.03       # sum of today's new positions' risk, share of equity
DAILY_DRAWDOWN_HALT = 0.03      # equity down this much on the day → no new buys
MAX_POSITIONS = 8
MAX_PER_SECTOR = 3
MIN_R = 2.0
MIN_STOP_ATRS = 1.0             # stop no nearer than this many ATRs
MAX_STOP_PCT = 0.10             # stop no farther than this below entry
MAX_LIMIT_DEVIATION = 0.03      # entry no farther than this from the last price
EARNINGS_BLACKOUT_DAYS = 1      # no new position with a report inside this
REENTRY_DAYS = 10
MIN_PRICE = 5.0
MIN_CASH_LEFT = 0.0


# ---------------------------------------------------------------------------
# the book
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    shares: int
    entry: float
    stop: float
    target: float
    opened: str
    horizon_days: int = 30
    sector: str = ""
    thesis: str = ""
    invalidation: str = ""
    principles: list = field(default_factory=list)
    regime: str = ""
    parent_id: str = ""
    legs: dict = field(default_factory=dict)
    adopted: bool = False
    stop_raised: bool = False

    def risk(self) -> float:
        return self.shares * max(0.0, self.entry - self.stop)

    def r_at(self, price: float) -> float:
        d = self.entry - self.stop
        return (price - self.entry) / d if d > 0 else float("nan")

    def deadline(self) -> date:
        return date.fromisoformat(self.opened) + timedelta(days=self.horizon_days)


@dataclass
class Closed:
    symbol: str
    shares: int
    entry: float
    exit: float
    opened: str
    closed: str
    reason: str
    thesis: str = ""
    invalidation: str = ""
    principles: list = field(default_factory=list)
    pnl: float = 0.0
    r: float = float("nan")
    reviewed: bool = False          # a post-mortem has been logged


class DeskBook:
    """Positions with their levels and their reasons. One file, plain JSON."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else task_dir(TASK) / "book.json"
        self.positions: dict[str, Position] = {}
        self.closed: list[Closed] = []
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for sym, d in (data.get("positions") or {}).items():
            self.positions[sym] = Position(**d)
        self.closed = [Closed(**d) for d in data.get("closed") or []]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "closed": [asdict(c) for c in self.closed],
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    def close(self, symbol: str, exit_price: float, when: date, reason: str,
              shares: int | None = None) -> Closed | None:
        p = self.positions.get(symbol)
        if p is None:
            return None
        n = shares if shares is not None else p.shares
        c = Closed(symbol=symbol, shares=n, entry=p.entry, exit=exit_price, opened=p.opened,
                   closed=when.isoformat(), reason=reason, thesis=p.thesis,
                   invalidation=p.invalidation, principles=list(p.principles),
                   pnl=round(n * (exit_price - p.entry), 2), r=p.r_at(exit_price))
        self.closed.append(c)
        if n >= p.shares:
            del self.positions[symbol]
        else:
            p.shares -= n
        self.save()
        return c

    def recently_closed(self, today: date, days: int = REENTRY_DAYS) -> set[str]:
        out = set()
        for c in self.closed:
            try:
                if (today - date.fromisoformat(c.closed)).days < days and c.reason.startswith("stop"):
                    out.add(c.symbol)
            except ValueError:
                continue
        return out

    def unreviewed(self) -> list[Closed]:
        return [c for c in self.closed if not c.reviewed]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    ok: bool
    reason: str
    shares: int = 0
    risk: float = 0.0
    notional: float = 0.0


def size(equity: float, entry: float, stop: float, cash: float,
         held_value: float = 0.0) -> Verdict:
    """The manual's formula, exactly: min(risk budget, name cap, cash)."""
    per_share = entry - stop
    if per_share <= 0:
        return Verdict(False, "stop is not below the entry")
    by_risk = int(equity * RISK_PCT // per_share)
    by_cap = int(max(0.0, equity * MAX_NAME_WEIGHT - held_value) // entry)
    by_cash = int(max(0.0, cash - MIN_CASH_LEFT) // entry)
    n = max(0, min(by_risk, by_cap, by_cash))
    if n <= 0:
        which = ("the name cap" if by_cap <= 0 else "cash" if by_cash <= 0 else "the risk budget")
        return Verdict(False, f"{which} allows no shares (risk {by_risk}, cap {by_cap}, cash {by_cash})")
    return Verdict(True, f"{n} shares = min(risk {by_risk}, cap {by_cap}, cash {by_cash})",
                   shares=n, risk=round(n * per_share, 2), notional=round(n * entry, 2))


class Gate:
    """Refuses what the manual forbids. Holds no opinion about anything else."""

    def __init__(self, *, equity: float, cash: float, last_equity: float,
                 holdings: dict[str, float], book: DeskBook, today: date,
                 risk_committed_today: float = 0.0, market_open: bool = True):
        self.equity, self.cash, self.last_equity = equity, cash, last_equity
        self.holdings = holdings                     # symbol → market value
        self.book, self.today = book, today
        self.risk_today = risk_committed_today
        self.market_open = market_open

    def drawdown(self) -> float:
        if _ok(self.last_equity) and self.last_equity > 0 and _ok(self.equity):
            return 1.0 - self.equity / self.last_equity
        return 0.0

    def buy(self, symbol: str, entry: float, stop: float, target: float, *,
            price: float, atr_pct: float, sector: str, earnings_days: float,
            sma200: float = float("nan")) -> Verdict:
        if not self.market_open:
            return Verdict(False, "the market is closed; buys only in the session")
        if _ok(sma200) and _ok(price) and price > 0 and price < sma200:
            # The manual's first hard negative, enforced here too so that no
            # headline, however good, buys a name under its 200-day line.
            return Verdict(False, f"price {price:.2f} is under the 200-day line {sma200:.2f}")
        if symbol in self.book.positions:
            return Verdict(False, f"{symbol} is already a position; raise its stop or leave it")
        if symbol in self.book.recently_closed(self.today):
            return Verdict(False, f"{symbol} was stopped out within {REENTRY_DAYS} days")
        if len(self.book.positions) >= MAX_POSITIONS:
            return Verdict(False, f"{MAX_POSITIONS} positions already")
        same = sum(1 for p in self.book.positions.values() if p.sector and p.sector == sector)
        if sector and same >= MAX_PER_SECTOR:
            return Verdict(False, f"{MAX_PER_SECTOR} positions in {sector} already")
        if self.drawdown() >= DAILY_DRAWDOWN_HALT:
            return Verdict(False, f"equity is down {self.drawdown() * 100:.1f}% today; no new buys")
        if not all(_ok(v) and v > 0 for v in (entry, stop, target)):
            return Verdict(False, "entry, stop and target must all be positive numbers")
        if not (stop < entry < target):
            return Verdict(False, f"levels do not bracket the entry (stop {stop}, entry {entry}, target {target})")
        if entry < MIN_PRICE:
            return Verdict(False, f"price under ${MIN_PRICE:.0f}")
        if _ok(price) and price > 0 and abs(entry - price) / price > MAX_LIMIT_DEVIATION:
            return Verdict(False, f"entry {entry:.2f} is {abs(entry - price) / price * 100:.1f}% from the last price {price:.2f} (max {MAX_LIMIT_DEVIATION * 100:.0f}%)")
        dist = (entry - stop) / entry
        if dist > MAX_STOP_PCT:
            return Verdict(False, f"stop is {dist * 100:.1f}% below the entry (max {MAX_STOP_PCT * 100:.0f}%)")
        if _ok(atr_pct) and atr_pct > 0 and dist < MIN_STOP_ATRS * atr_pct:
            return Verdict(False, f"stop is {dist / atr_pct:.2f} ATR from the entry (min {MIN_STOP_ATRS:.0f} ATR)")
        r = (target - entry) / (entry - stop)
        if r < MIN_R:
            return Verdict(False, f"R {r:.2f} is under {MIN_R:.1f}")
        if _ok(earnings_days) and 0 <= earnings_days <= EARNINGS_BLACKOUT_DAYS:
            return Verdict(False, f"earnings in {earnings_days:.0f} day(s)")
        v = size(self.equity, entry, stop, self.cash, self.holdings.get(symbol, 0.0))
        if not v.ok:
            return v
        if self.risk_today + v.risk > self.equity * MAX_DAILY_NEW_RISK:
            return Verdict(False, f"today's new risk would reach ${self.risk_today + v.risk:,.0f}, over {MAX_DAILY_NEW_RISK * 100:.0f}% of equity")
        return v

    def stop_change(self, symbol: str, new_stop: float) -> Verdict:
        p = self.book.positions.get(symbol)
        if p is None:
            return Verdict(False, f"{symbol} is not a position in the book")
        if not _ok(new_stop) or new_stop <= p.stop:
            return Verdict(False, f"stops only move up: {new_stop} is not above {p.stop:.2f}")
        return Verdict(True, f"stop {p.stop:.2f} → {new_stop:.2f}")

    def protect(self, symbol: str, stop: float, target: float, *, price: float,
                atr_pct: float) -> Verdict:
        held = self.holdings.get(symbol, 0.0)
        if held <= 0:
            return Verdict(False, f"{symbol} is not held")
        if not (_ok(stop) and _ok(target) and stop < price < target):
            return Verdict(False, f"stop {stop} and target {target} must bracket the price {price:.2f}")
        if (price - stop) / price > MAX_STOP_PCT:
            return Verdict(False, f"stop is more than {MAX_STOP_PCT * 100:.0f}% below the price")
        if _ok(atr_pct) and atr_pct > 0 and (price - stop) / price < MIN_STOP_ATRS * atr_pct:
            return Verdict(False, f"stop is under {MIN_STOP_ATRS:.0f} ATR from the price")
        return Verdict(True, "protected")


# ---------------------------------------------------------------------------
# the executor
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    action: str
    symbol: str
    ok: bool
    reason: str
    shares: int = 0
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    venue: str = ""


def log_entry(entry: dict, path: Path | None = None) -> Path:
    path = path or task_dir(TASK) / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"logged_at": datetime.now().isoformat(), **entry}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if entry.get("kind") == "postmortem" and entry.get("symbol"):
        book = DeskBook()
        for c in book.closed:
            if c.symbol == entry["symbol"] and not c.reviewed:
                c.reviewed = True
        book.save()
    return path


def risk_committed(today: date, path: Path | None = None) -> float:
    path = path or task_dir(TASK) / "decisions.jsonl"
    total = 0.0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d.get("kind") == "order" and d.get("action") == "buy" and d.get("ok") \
                    and str(d.get("date", "")) == today.isoformat():
                total += _num(d.get("risk"), 0.0)
    except (OSError, ValueError):
        pass
    return total


class Executor:
    def __init__(self, *, broker, market: Market | None = None, book: DeskBook | None = None,
                 now: datetime | None = None, dry_run: bool = False, log_path: Path | None = None):
        self.broker = broker
        self.market = market or Market(task=TASK)
        self.book = book or DeskBook()
        self.now = now or datetime.now(clock.ET)
        self.dry_run = dry_run
        self.log_path = log_path
        self.outcomes: list[Outcome] = []

    def run(self, intent: dict) -> list[Outcome]:
        today = clock.market_state(self.now).trading_day
        data_day = today if clock.market_state(self.now).is_open else today
        orders = list(intent.get("orders") or [])
        if kill_switch_engaged():
            for o in orders:
                self.record(Outcome(o.get("action", "?"), str(o.get("symbol", "")).upper(), False,
                                    "kill switch engaged"), o, today)
            return self.outcomes
        try:
            account = self.broker.account()
        except Exception as exc:
            for o in orders:
                self.record(Outcome(o.get("action", "?"), str(o.get("symbol", "")).upper(), False,
                                    f"account unreadable: {exc}"), o, today)
            return self.outcomes
        holdings = {str(h.symbol).upper(): _num(h.market_value, 0.0) for h in (account.holdings or [])}
        shares_held = {str(h.symbol).upper(): int(_num(h.quantity, 0.0)) for h in (account.holdings or [])}
        equity, last_equity = (self.broker.equity_today() if hasattr(self.broker, "equity_today")
                               else (_num(account.account_value), float("nan")))
        equity = equity if _ok(equity) else _num(account.account_value)
        market_open = self.broker.market_open() if hasattr(self.broker, "market_open") else True
        gate = Gate(equity=equity, cash=_num(account.cash, 0.0), last_equity=last_equity,
                    holdings=holdings, book=self.book, today=today,
                    risk_committed_today=risk_committed(today, self.log_path),
                    market_open=market_open or self.dry_run)

        rank = {"sell": 0, "trim": 1, "raise_stop": 2, "protect": 3, "buy": 4}
        for o in sorted(orders, key=lambda o: rank.get(str(o.get("action")), 9)):
            action = str(o.get("action", "")).lower()
            sym = str(o.get("symbol", "")).upper()
            if not sym:
                self.record(Outcome(action, sym, False, "no symbol"), o, today)
                continue
            f = self.market.facts(sym, data_day)
            try:
                price = _num(self.broker.quote(sym))
            except Exception:
                price = float("nan")
            if not (_ok(price) and price > 0):
                price = f.price
            if action == "buy":
                out = self.buy(o, sym, f, price, gate, today)
            elif action == "sell":
                out = self.sell(o, sym, price, shares_held, today)
            elif action == "trim":
                out = self.trim(o, sym, price, shares_held, today, gate, f)
            elif action == "raise_stop":
                out = self.raise_stop(o, sym, gate)
            elif action == "protect":
                out = self.protect(o, sym, price, shares_held, gate, f, today)
            else:
                out = Outcome(action, sym, False, f"unknown action {action!r}")
            if out.ok and action == "buy":
                gate.risk_today += out.shares * (out.entry - out.stop)
                gate.cash -= out.shares * out.entry
            self.record(out, o, today)
        self.write_note(today)
        return self.outcomes

    # --- each action ------------------------------------------------------

    def buy(self, o, sym, f, price, gate: Gate, today: date) -> Outcome:
        entry, stop, target = _num(o.get("entry")), _num(o.get("stop")), _num(o.get("target"))
        if not _ok(entry) and _ok(price):
            entry = price
        earnings_days = float("nan")
        e = self.market.earnings([sym], today).get(sym)
        if e is not None:
            with contextlib.suppress(Exception):
                earnings_days = _num(e.days_to_next(today))
        sector = str(o.get("sector") or f.sector or "")
        v = gate.buy(sym, entry, stop, target, price=price, atr_pct=_num(f.snap.atr_pct),
                     sector=sector, earnings_days=earnings_days, sma200=_num(f.snap.sma200))
        out = Outcome("buy", sym, v.ok, v.reason, v.shares, entry, stop, target)
        if not v.ok:
            return out
        res = self.broker.place_bracket(sym, v.shares, entry, stop, target, dry_run=self.dry_run)
        out.ok, out.venue = bool(res.ok), str(res.message)
        if not res.ok:
            out.reason = f"venue refused: {res.message}"
            return out
        legs = {}
        with contextlib.suppress(ValueError):
            legs = json.loads(res.artifact) if res.artifact else {}
        if not self.dry_run:
            self.book.positions[sym] = Position(
                symbol=sym, shares=v.shares, entry=entry, stop=stop, target=target,
                opened=today.isoformat(), horizon_days=int(o.get("horizon_days") or 30),
                sector=sector, thesis=str(o.get("thesis") or ""),
                invalidation=str(o.get("invalidation") or ""),
                principles=list(o.get("principles") or []), regime=str(o.get("regime") or ""),
                parent_id=res.broker_order_id or "", legs=legs)
            self.book.save()
        return out

    def sell(self, o, sym, price, held, today) -> Outcome:
        if held.get(sym, 0) <= 0:
            return Outcome("sell", sym, False, f"{sym} is not held")
        if self.dry_run:
            return Outcome("sell", sym, True, "dry run", held[sym], venue="not sent")
        res = self.broker.close_now(sym)
        out = Outcome("sell", sym, bool(res.ok), str(o.get("reason") or ""), held[sym], venue=str(res.message))
        if res.ok:
            self.book.close(sym, price if _ok(price) else 0.0, today, f"manual: {o.get('reason', '')}")
        else:
            out.reason = f"venue refused: {res.message}"
        return out

    def trim(self, o, sym, price, held, today, gate: Gate, f) -> Outcome:
        n_held = held.get(sym, 0)
        if n_held <= 0:
            return Outcome("trim", sym, False, f"{sym} is not held")
        frac = _num(o.get("fraction"), 0.5)
        n = int(o.get("shares") or max(1, int(n_held * frac)))
        n = min(n, n_held)
        if n >= n_held:
            return self.sell(o, sym, price, held, today)
        p = self.book.positions.get(sym)
        stop = _num(o.get("stop"), p.stop if p else float("nan"))
        target = _num(o.get("target"), p.target if p else float("nan"))
        if self.dry_run:
            return Outcome("trim", sym, True, "dry run", n, venue="not sent")
        # cancel the legs, sell n at market, re-protect the remainder
        for leg in self.broker.open_orders():
            if leg["symbol"].upper() == sym and leg["side"] == "sell":
                self.broker.cancel(leg["id"])
        res = self.broker.place_order(sym, "Sell", n, order_type="Market")
        out = Outcome("trim", sym, bool(res.ok), str(o.get("reason") or ""), n, venue=str(res.message))
        if not res.ok:
            out.reason = f"venue refused: {res.message}"
            return out
        self.book.close(sym, price if _ok(price) else 0.0, today, f"trim: {o.get('reason', '')}", shares=n)
        left = n_held - n
        if _ok(stop) and _ok(target) and stop < price < target:
            pr = self.broker.protect(sym, left, stop, target)
            out.venue += f"; remainder {left} re-protected: {pr.message}"
        else:
            out.venue += f"; remainder {left} is UNPROTECTED — send a protect order"
        return out

    def raise_stop(self, o, sym, gate: Gate) -> Outcome:
        new_stop = _num(o.get("stop"))
        v = gate.stop_change(sym, new_stop)
        out = Outcome("raise_stop", sym, v.ok, v.reason, stop=new_stop)
        if not v.ok:
            return out
        if self.dry_run:
            out.venue = "not sent"
            return out
        ok, msg = self.broker.raise_stop(sym, new_stop)
        out.ok, out.venue = ok, msg
        if ok:
            p = self.book.positions[sym]
            p.stop, p.stop_raised = new_stop, True
            self.book.save()
        else:
            out.reason = f"venue refused: {msg}"
        return out

    def protect(self, o, sym, price, held, gate: Gate, f, today) -> Outcome:
        stop, target = _num(o.get("stop")), _num(o.get("target"))
        v = gate.protect(sym, stop, target, price=price, atr_pct=_num(f.snap.atr_pct))
        out = Outcome("protect", sym, v.ok, v.reason, held.get(sym, 0), price, stop, target)
        if not v.ok:
            return out
        if hasattr(self.broker, "stop_leg") and self.broker.stop_leg(sym) is not None:
            out.ok, out.reason = False, f"{sym} already has a resting stop; raise it instead"
            return out
        res = self.broker.protect(sym, held[sym], stop, target, dry_run=self.dry_run)
        out.ok, out.venue = bool(res.ok), str(res.message)
        if res.ok and not self.dry_run:
            self.book.positions[sym] = Position(
                symbol=sym, shares=held[sym], entry=price, stop=stop, target=target,
                opened=today.isoformat(), sector=str(o.get("sector") or f.sector or ""),
                thesis=str(o.get("thesis") or o.get("reason") or "接管账户里已有的仓位"),
                invalidation=str(o.get("invalidation") or ""), adopted=True,
                parent_id=res.broker_order_id or "")
            self.book.save()
        elif not res.ok:
            out.reason = f"venue refused: {res.message}"
        return out

    # --- the record ---------------------------------------------------------

    def record(self, out: Outcome, o: dict, today: date) -> None:
        self.outcomes.append(out)
        entry = {"kind": "order", "date": today.isoformat(), "action": out.action, "symbol": out.symbol,
                 "ok": out.ok, "gate": out.reason, "venue": out.venue, "shares": out.shares,
                 "entry": out.entry, "stop": out.stop, "target": out.target,
                 "risk": round(out.shares * (out.entry - out.stop), 2)
                 if out.action == "buy" and _ok(out.entry) and _ok(out.stop) else 0.0,
                 "dry_run": self.dry_run,
                 "thesis": o.get("thesis") or o.get("reason") or "",
                 "invalidation": o.get("invalidation") or "",
                 "principles": o.get("principles") or [], "regime": o.get("regime") or ""}
        try:
            log_entry(entry, self.log_path)
        except OSError as exc:
            logger.warning("decision not logged: %s", exc)

    def write_note(self, today: date) -> None:
        try:
            with open(task_dir(TASK) / f"{today.isoformat()}-orders.md", "a", encoding="utf-8") as fh:
                fh.write(f"\n### {datetime.now():%H:%M}" + ("（dry run）" if self.dry_run else "") + "\n")
                fh.write(format_outcomes(self.outcomes) + "\n")
        except OSError:
            pass


def format_outcomes(outcomes: list[Outcome]) -> str:
    if not outcomes:
        return "- 没有单"
    lines = []
    for o in outcomes:
        mark = "✓" if o.ok else "✗"
        levels = ""
        if o.action in ("buy", "protect") and _ok(o.stop):
            levels = f" 入场 {o.entry:,.2f} 止损 {o.stop:,.2f} 目标 {o.target:,.2f}"
        elif o.action == "raise_stop":
            levels = f" 止损 → {o.stop:,.2f}"
        lines.append(f"- {mark} {o.action} {o.symbol} {o.shares or ''}{levels} — {o.reason}"
                     + (f"（{o.venue}）" if o.venue else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk order",
                                description="place what an intent file asks for, through the gate")
    p.add_argument("intent", help="the intent json file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--venue", default="alpaca")
    args = p.parse_args(argv)
    intent = json.loads(Path(args.intent).read_text(encoding="utf-8"))
    from tradingagents.live.broker import open_broker
    broker = open_broker(args.venue)
    from . import state
    print(state.pull())
    ex = Executor(broker=broker, dry_run=args.dry_run)
    outcomes = ex.run(intent)
    print(format_outcomes(outcomes))
    print(state.push(message=f"orders {intent.get('date', '')}"))
    return 0 if all(o.ok for o in outcomes) else 1


def main_log(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk log",
                                description="append a structured entry (decision, postmortem, note) to the decision log")
    p.add_argument("entry", help="a json file, or '-' for stdin")
    args = p.parse_args(argv)
    import sys
    text = sys.stdin.read() if args.entry == "-" else Path(args.entry).read_text(encoding="utf-8")
    data = json.loads(text)
    entries = data if isinstance(data, list) else [data]
    for e in entries:
        path = log_entry(e)
    print(f"logged {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} to {path}")
    from . import state
    print(state.push(message="decision log"))
    return 0
