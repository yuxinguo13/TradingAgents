"""The bridge between what was decided and what is actually held.

There are two decision systems in this package and, until this module, no way
for either of them to reach the account.

:mod:`advisor` produces decisions without a model — screen, levels, sizing,
exits — and writes them into ``recommendations.json``. It says so at the top of
its own docstring: *nothing here places an order.* :mod:`monitor` can place
orders, but only after a persona panel votes, and the panel needs an LLM; run
with ``--no-llm`` it is a sentinel that never decides. So a desk configured the
way this one is — advisor daily, monitor with ``--no-llm`` — produces a full
book of decisions and executes none of them, forever, without ever erroring.

That gap is not visible from either side. The advisor's track record scores
ideas as though they were taken; the venue quietly holds something else. On
2026-09-01 the three books had *no symbol in common*: six open recommendations,
five local paper positions seeded as a demo on day one, and an empty Alpaca
account.

This module closes it, and the shape is chosen so it cannot cause the failure
it exists to prevent:

* **Reporting is the default; submitting takes a flag.** An execution bridge
  that trades by default is one you learn about after it has traded.
* **Every order goes through the same Secretary** the panel's orders go
  through. A second path to the venue would be a second set of risk limits.
* **Positions the book does not know about are reported, never touched.** The
  venue may hold things this desk did not choose — a hand trade, an older
  strategy, the long-term book on its own monthly clock, a demo seed. Selling
  them because they are absent from one book is the bridge deciding it owns
  the whole account. *Absent* has to mean absent, though: a position the book
  already closed is not unrecognised, it is an exit that never reached the
  venue, and reading only the open rows files that under "never touch" — which
  leaves the record showing the loss cut while the account keeps riding it.
* **It sells as readily as it buys.** Entries come out of the book on their
  own; exits have to be computed. Behind a flag, they never were, and the only
  thing the bridge could do was open positions.
* **It never writes to the book.** The advisor owns those records; a bridge
  that edited them could make the track record agree with the account by
  changing the wrong one.

    python -m tradingagents.live.execute            # what it would do
    python -m tradingagents.live.execute --submit   # do it
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import date as _date

from . import clock
from .broker import BUY, SELL, Account, Holding, LIMIT, MARKET, open_broker
from .recommendations import (
    CLOSED, EXPIRED, TRIM, Recommendation, RecommendationBook,
)
from .secretary import Order, RiskLimits, Secretary, TradeLedger, kill_switch_engaged

logger = logging.getLogger(__name__)

# How far a venue position may differ from the book before it is called drift
# rather than rounding. One share: fractional fills and dividend reinvestment
# both produce sub-share differences that are not a reconciliation problem.
SHARE_TOLERANCE = 1.0

# How many days past the session it was issued FOR an unfilled entry stays
# actionable. Zero: an idea is priced off one close and meant for one open, so
# once that open has gone by, its limit, its stop and its R all refer to a
# price that has moved.
#
# The clock this counts against is the session being planned, not the session
# the data came from — those are always one apart, and counting from the data
# day quietly granted every entry an extra session of life it was never
# supposed to have.
#
# This is the failure the method document records happening for real: NRIX and
# NTRA were issued at 2.42R and 2.41R, both names rose ~2.6% before the open,
# and taking them at the old stop would have bought the same downside for less
# upside. A bridge that reads the book literally would place those orders six
# days later at a six-day-old limit and never mention it — so stale entries are
# reported in their own section with R recomputed at the current price, and are
# never submitted.
ENTRY_FRESH_DAYS = 0

# How long the book's own exit stays an order rather than a remark, in calendar
# days. A position the book has already closed while the account still holds it
# is not an unrecognised position — the decision was taken and never reached
# the venue, which is the precise failure this module exists for.
#
# It is bounded because the opposite reading is also a real account: a name
# exited months ago and since bought back by hand is not this bridge's to sell.
# Past the window the holding is still named, with the exit and its date
# attached, so the bounded case degrades into a printed remark and never into
# silence.
EXIT_WINDOW_DAYS = 10


def _num(v, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _age_days(rec: Recommendation, as_of: _date | None) -> int | None:
    """Calendar days from the session the idea was issued for. None if unreadable."""
    try:
        issued = _date.fromisoformat(str(rec.issued_date))
    except (TypeError, ValueError):
        return None
    return max(0, ((as_of or _date.today()) - issued).days)


@dataclass
class Intent:
    """One order the book implies. Not yet vetted, not yet placed."""

    action: str                      # Buy | Sell
    symbol: str
    shares: int
    limit: float | None = None
    reason: str = ""
    rec_id: str = ""
    urgency: int = 1                 # same 1/2/3 scale as ExitSignal

    @property
    def order_type(self) -> str:
        return LIMIT if self.limit else MARKET

    def order(self) -> Order:
        return Order(symbol=self.symbol, action=self.action, quantity=int(self.shares),
                     order_type=self.order_type, limit_price=self.limit,
                     rationale=self.reason, source="book")


@dataclass
class Reconciliation:
    """What the book says the account should hold, against what it holds."""

    to_open: list = field(default_factory=list)      # Intent
    to_close: list = field(default_factory=list)     # Intent
    # Partial sells the book has already booked against the idea. Separate from
    # to_close because they are a different instruction: conflating the two is
    # how the trim went missing — the book shrank the position, the venue did
    # not, and the difference was filed as an unexplained share gap.
    to_trim: list = field(default_factory=list)      # Intent
    # Entries whose levels have gone stale: (Intent, days_old, R_now).
    # Reported, never submitted — see ENTRY_FRESH_DAYS.
    stale: list = field(default_factory=list)
    drift: list = field(default_factory=list)        # (symbol, book_shares, venue_shares)
    matched: list = field(default_factory=list)      # (symbol, shares)
    unmanaged: list = field(default_factory=list)    # Holding
    # Held by the long-term book, which runs on its own monthly clock. Not
    # unmanaged: naming a core position "unrecognised" every single day is how
    # that section stops being read, and the one day it says something new goes
    # by unnoticed with it.
    core_held: list = field(default_factory=list)    # Holding
    # The account holds the opposite of what the book says. Its own bucket
    # because it is the one disagreement that must never be resolved by this
    # module: buying to "close the gap" against a short would cover it, and
    # selling would open one.
    conflicts: list = field(default_factory=list)    # (Holding, why)
    notes: list = field(default_factory=list)

    @property
    def intents(self) -> list:
        """Exits first: the proceeds of a sale are what fund a purchase, and a
        bridge that buys before it sells can be rejected for cash it is about
        to have."""
        return (sorted(self.to_close, key=lambda i: -i.urgency)
                + sorted(self.to_trim, key=lambda i: -i.urgency)
                + self.to_open)
        # 过期未成交刻意不在这里：它们要重新定量，不是补单

    @property
    def clean(self) -> bool:
        return not (self.to_open or self.to_close or self.to_trim
                    or self.drift or self.stale or self.conflicts)


def _r_now(rec: Recommendation, price: float) -> float:
    """The reward/risk an entry would have if taken at ``price`` today.

    Against the stop as issued, because that is the stop the idea carries: a
    later entry with an unchanged stop buys less upside for the same downside,
    and this number is how much less.
    """
    stop = _num(rec.initial_stop_price, _num(rec.stop_price))
    target = _num(rec.target_price)
    if not all(math.isfinite(x) for x in (price, stop, target)) or price <= stop:
        return float("nan")
    return (target - price) / (price - stop)


def _to_stop(rec: Recommendation, price: float) -> float:
    """How far the current price sits above the stop, as a fraction.

    The number that stops a stale entry's R from being read as good news. When
    a name falls toward its stop the R *rises* — the target is unchanged and
    the risk per share shrank — but what actually shrank is the room the trade
    has to breathe. An 8R entry bought 1% above its own stop is not an 8R
    trade; it is a coin flip with a flattering ratio.
    """
    stop = _num(rec.initial_stop_price, _num(rec.stop_price))
    if not (math.isfinite(price) and math.isfinite(stop)) or price <= 0:
        return float("nan")
    return price / stop - 1.0


def _sell_intent(sym: str, have: int, sig, rec_id: str, *,
                 whole: bool = True) -> Intent | None:
    """A sell of what is actually held, never of what the signal imagined.

    ``whole`` decides what an unusable share count means, and the two answers
    are not interchangeable. For a close it means everything held: the
    instruction is "get out", and the number is only how the book remembers
    the size. For a trim it means nothing can be placed — a partial sell whose
    size did not survive is not a licence to sell the lot, and treating it as
    one turns a take-some-profit into a full exit nobody asked for.
    """
    n = int(_num(getattr(sig, "shares", 0), 0))
    if n <= 0:
        if not whole:
            return None
        n = have
    return Intent(SELL, sym, min(have, n), None,
                  f"{getattr(sig, 'action', 'SELL')}：{getattr(sig, 'reason', '')}",
                  rec_id, int(_num(getattr(sig, "urgency", 1), 1)))


def _recent_exits(book, as_of: _date | None, window_days: int) -> dict:
    """Symbols the book has exited, newest exit per symbol.

    A holding whose book record is a *closed* recommendation is the one case a
    reader of open rows alone gets exactly backwards. The book closed it, the
    order was never placed, and from then on the symbol is absent from
    ``open_recommendations`` — so a bridge that matches only open rows files it
    under "not recognised, never touched" and never sells it again. The record
    shows the loss cut; the account keeps riding it. That is this module's own
    failure mode, printed as reassurance.

    Returns every exit, dated. The caller decides which are recent enough to be
    orders — see :data:`EXIT_WINDOW_DAYS`.
    """
    out: dict = {}
    for rec in list(getattr(book, "recommendations", None) or []):
        if getattr(rec, "status", "") not in (CLOSED, EXPIRED):
            continue
        sym = str(getattr(rec, "symbol", "") or "").upper()
        if not sym:
            continue
        try:
            exited = _date.fromisoformat(str(rec.exit_date))
        except (TypeError, ValueError):
            continue
        prev = out.get(sym)
        if prev is None or exited >= prev[1]:
            out[sym] = (rec, exited)
    return out


def _positions(account: Account) -> tuple[dict, dict]:
    """The account's long and short books, keyed by symbol.

    Three things a venue does that a dict comprehension over ``holdings`` gets
    wrong, all of them silently:

    * **Two rows for one symbol.** Keyed naively the later row wins and the
      earlier size vanishes, so a 150-share position reads as 50 and the
      reconciliation invents a drift that is not there. Summed instead.
    * **A row with no shares.** A closed position some venues still return. It
      is not a holding, and printed as one it says the account holds something
      the book does not recognise — of nothing.
    * **A short.** Held apart, because as a positive quantity it matches a long
      of the same size and the reconciliation reports agreement while the
      account is positioned the opposite way.
    """
    long_: dict = {}
    short: dict = {}
    for h in (account.holdings or []):
        sym = str(getattr(h, "symbol", "") or "").upper()
        qty = _num(getattr(h, "quantity", 0.0), 0.0)
        if not sym or not (abs(qty) > 0):
            continue
        book = short if str(getattr(h, "side", "long")).lower() == "short" else long_
        prev = book.get(sym)
        book[sym] = h if prev is None else replace(
            prev, quantity=_num(prev.quantity, 0.0) + qty,
            market_value=_num(getattr(prev, "market_value", 0.0), 0.0)
            + _num(getattr(h, "market_value", 0.0), 0.0))
    return long_, short


def plan(book: RecommendationBook, account: Account, *,
         exits=None, as_of: _date | None = None, quote=None,
         fresh_days: int = ENTRY_FRESH_DAYS, core=None,
         exit_window_days: int = EXIT_WINDOW_DAYS) -> Reconciliation:
    """Compare the book to the account and say what would close the gap.

    ``as_of`` is the session being planned — the one the orders are FOR, not
    the one the data came from. Every age below counts against it.

    ``exits`` is the exit-signal list the advisor already computed for today;
    passing it keeps one exit engine rather than two. Without it, the plan
    covers entries and drift only, and says so.

    ``core`` is the long-term book's symbols. They are held deliberately, on a
    monthly clock this module has no part in, and listing them as unrecognised
    every day is how a warning section becomes wallpaper.
    """
    out = Reconciliation()
    out._book = {}
    when = as_of or _date.today()
    held, short = _positions(account)
    open_recs = [r for r in book.open_recommendations() if r.symbol]
    booked = {r.symbol.upper(): r for r in open_recs}
    core_syms = {str(s).upper() for s in (core or ()) if str(s).strip()}

    closing, trimming = {}, {}
    for sig in (exits or []):
        sym = str(getattr(sig, "symbol", "") or "").upper()
        if not sym:
            continue
        if getattr(sig, "closes_position", False):
            closing[sym] = sig
        elif str(getattr(sig, "action", "")) == TRIM:
            trimming[sym] = sig
    if exits is None:
        out.notes.append("没有传入离场信号，本次只对账入场与股数差异；"
                         "离场要由 advisor 的规则算出来，不能在这里另起一套")

    for sym, rec in booked.items():
        out._book[rec.id] = rec
        want = int(_num(rec.shares, 0.0))
        if sym in short:
            # Never an order. Buying to close the gap would cover the short;
            # selling would deepen it. Either way this module would be taking a
            # position on a trade it has no record of.
            out.conflicts.append((short[sym],
                                  f"账户是空头 {_num(short[sym].quantity, 0):,.0f} 股，"
                                  f"账本却是多头 {want:,} 股——这里不下任何单"))
            continue
        h = held.get(sym)
        have = int(_num(getattr(h, "quantity", 0.0), 0.0)) if h else 0
        sig = closing.get(sym)
        if sig is not None:
            if have > 0:
                out.to_close.append(_sell_intent(sym, have, sig, rec.id))
            else:
                out.notes.append(f"{sym}：账本要离场，但账户里本来就没有仓位")
            continue
        tsig = trimming.get(sym)
        if tsig is not None:
            # The book already took these shares off the idea, so ``want`` is
            # the post-trim size and the venue still holds the pre-trim one.
            # Read as drift the difference reads as "someone traded this by
            # hand"; it is this morning's own instruction.
            if have <= 0:
                out.notes.append(f"{sym}：账本要减仓，但账户里本来就没有仓位")
                continue
            trim = _sell_intent(sym, have, tsig, rec.id, whole=False)
            if trim is not None:
                out.to_trim.append(trim)
                continue
            # 说不出减多少就不下单，也绝不当成「全卖」。差额照样往下走，
            # 按股数差异报出来——不能因为定不了量就连差额也不提。
            out.notes.append(f"{sym}：账本说要减仓却没给股数，这里只报差额、不下单")
        if want <= 0:
            # Silence here reads as "nothing to do". It is a recommendation
            # whose sizing produced no shares, which is a different thing.
            out.notes.append(f"{sym}：账本里这条没有股数，无法对账，也不会下单")
            continue
        if have == 0:
            intent = Intent(BUY, sym, want, _num(rec.limit_price, None) or None,
                            f"账本 {rec.issued_date} 发出，尚未建仓", rec.id)
            age = _age_days(rec, when)
            if age is not None and age > fresh_days:
                px = float("nan")
                if quote is not None:
                    try:
                        px = _num(quote(sym), float("nan"))
                    except Exception:
                        px = float("nan")
                out.stale.append((intent, age, _r_now(rec, px), px,
                                  _to_stop(rec, px)))
            else:
                out.to_open.append(intent)
        elif abs(have - want) > SHARE_TOLERANCE:
            out.drift.append((sym, want, have))
        else:
            out.matched.append((sym, have))

    exited = _recent_exits(book, when, exit_window_days)
    for sym, h in short.items():
        if sym not in booked:
            out.unmanaged.append(h)
    for sym, h in held.items():
        if sym in booked:
            continue
        have = int(_num(getattr(h, "quantity", 0.0), 0.0))
        sig = closing.get(sym)
        gone = exited.get(sym)
        if sig is not None and have > 0:
            # The advisor writes its exits back to the book before this runs,
            # so today's sell is already closed there and its symbol is no
            # longer open. The signal is the instruction; the book row is only
            # where it was recorded.
            rec_id = str(getattr(sig, "rec_id", "") or "")
            out.to_close.append(_sell_intent(sym, have, sig, rec_id))
        elif gone is not None and have > 0 and (when - gone[1]).days <= exit_window_days:
            rec, day = gone
            out._book[rec.id] = rec
            out.to_close.append(Intent(
                SELL, sym, have, None,
                f"账本 {day.isoformat()} 已记为离场"
                f"（{getattr(rec, 'exit_reason', '') or 'exit'}），但仓位还在账户里",
                rec.id, 3))
        elif sym in core_syms:
            out.core_held.append(h)
        else:
            if gone is not None:
                out.notes.append(
                    f"{sym}：账本 {gone[1].isoformat()} 就记为离场了，超过 "
                    f"{exit_window_days} 天，这里只提不下单——它可能是后来手工买回的")
            out.unmanaged.append(h)
    return out


def market_is_open(log=logger.info) -> bool:
    """Whether the regular session is open — the reading :mod:`monitor` takes.

    Fails closed. The guard this replaces asked ``clock`` for an
    ``is_market_open`` that has never existed, got ``False`` from ``hasattr``
    every single time, and passed ``market_open=True`` to the gate — which
    switched ``require_market_open`` off for every order this bridge placed,
    while reading like a careful check.
    """
    try:
        return bool(clock.market_state().is_tradeable)
    except Exception as exc:
        log(f"读不出市场状态（{type(exc).__name__}: {exc}）；按休市处理")
        return False


def _applied(account: Account, order: Order, price: float) -> Account:
    """``account`` with one fill applied, erring tight.

    A buy debits cash and adds the shares. A sell removes the shares and
    credits nothing: unsettled proceeds are not room to buy with, and the only
    error that matters here is the one that lets the next order through.
    """
    px, qty = _num(price, 0.0), int(_num(order.quantity, 0))
    if not (px > 0) or qty <= 0:
        return account
    notional, sym = qty * px, order.symbol.upper()
    holdings = list(account.holdings or [])
    i = next((n for n, h in enumerate(holdings) if h.symbol.upper() == sym), None)
    cash, power = account.cash, account.buying_power
    if order.action == BUY:
        if i is None:
            holdings.append(Holding(symbol=sym, quantity=float(qty), avg_cost=px,
                                    last=px, market_value=notional))
        else:
            h = holdings[i]
            holdings[i] = replace(h, quantity=h.quantity + qty,
                                  market_value=h.market_value + notional)
        # Read off the pre-fill pair. A venue that reports no buying power
        # falls back to cash, the same reading the Secretary takes — but taken
        # after cash was already debited it charges the same fill twice.
        room = power or cash
        cash = max(0.0, cash - notional)
        power = max(0.0, room - notional)
    elif i is not None:
        h = holdings[i]
        left = max(0.0, h.quantity - qty)
        if left <= 0:
            holdings.pop(i)
        else:
            holdings[i] = replace(h, quantity=left,
                                  market_value=max(0.0, h.market_value - notional))
    return replace(account, cash=cash, buying_power=power, holdings=holdings)


def _after(broker, account: Account, order: Order, price: float, log) -> Account:
    """The account the *next* order is checked against.

    One snapshot vetting a whole basket is a gate that reads as strict and
    approves more than the account can pay for: six buys each measured against
    the same untouched cash and the same untouched gross exposure. The venue's
    own books are the truth, so they are asked first; when they cannot be
    reached the fill is applied locally, and only in the direction that
    tightens what comes next.
    """
    local = _applied(account, order, price)
    try:
        fresh = broker.account()
    except Exception as exc:
        log(f"下单后读不回账户（{type(exc).__name__}: {exc}）；按本地估算继续")
        return local
    if fresh is None or _num(getattr(fresh, "account_value", 0.0), 0.0) <= 0:
        return local
    # The venue is the truth about what is held, and the estimate is the floor
    # under what is left to spend. A venue that has not yet registered the
    # order it just accepted answers with room that is already committed — and
    # that answer arriving one order too late is exactly the case this guard
    # exists for, so it is never allowed to read looser than the estimate.
    return replace(fresh,
                   cash=min(_num(fresh.cash, 0.0), _num(local.cash, 0.0)),
                   buying_power=min(_num(fresh.buying_power, 0.0),
                                    _num(local.buying_power, 0.0)),
                   holdings=_wider(fresh.holdings, local.holdings))


def _wider(venue: list, local: list) -> list:
    """Per symbol, whichever book shows the larger position.

    Gross exposure is a limit, and a venue that has not yet booked the fill
    reports less of it than the account carries.
    """
    out = {str(h.symbol).upper(): h for h in (venue or [])}
    for h in (local or []):
        sym = str(h.symbol).upper()
        cur = out.get(sym)
        if cur is None or _num(h.quantity, 0.0) > _num(cur.quantity, 0.0):
            out[sym] = h
    return list(out.values())


def _as_filled(order: Order, result) -> Order:
    """The order as the venue actually executed it.

    The daily trade and turnover budgets are spent from this row, so a
    partial fill booked at the size that was *asked for* spends budget on
    shares nobody owns — 30 of 100 filled charges the day for all 100. An
    order accepted without a fill keeps its full size: it is live at the
    venue, and the exposure is committed whether or not it has printed.
    """
    filled = _num(getattr(result, "filled_quantity", 0.0), 0.0)
    if filled <= 0 or int(filled) == int(order.quantity):
        return order
    return replace(order, quantity=int(filled))


def submit(rec: Reconciliation, broker, secretary: Secretary, account: Account,
           *, market_open: bool | None = None, log=logger.info) -> list:
    """Vet every intent through the Secretary, then place what survives.

    Returns ``(Intent, Verdict, OrderResult | None)`` triples so the caller can
    print exactly what happened to each one — including the ones the gate
    resized or refused, which are the interesting rows.

    The account is re-read after every fill. This function places a whole book
    at once, which is the case where a stale snapshot stops being a rounding
    error and becomes a basket the account cannot fund.
    """
    if market_open is None:
        market_open = market_is_open(log)
    out = []
    for intent in rec.intents:
        price = 0.0
        try:
            price = _num(broker.quote(intent.symbol), 0.0)
        except Exception as exc:
            log(f"{intent.symbol}: 取价失败 ({exc})")
        verdict = secretary.check(intent.order(), account, price, market_open=market_open)
        if not verdict.ok:
            out.append((intent, verdict, None))
            continue
        order = verdict.order or intent.order()
        try:
            result = broker.place_order(
                order.symbol, order.action, order.quantity,
                order_type=order.order_type, limit_price=order.limit_price)
        except Exception as exc:                       # adapters promise not to
            log(f"{intent.symbol}: 下单抛异常 ({type(exc).__name__}: {exc})")
            out.append((intent, verdict, None))
            continue
        try:
            # record() persists on its own; a second save() here wrote the
            # whole ledger twice per fill.
            secretary.ledger.record(_as_filled(order, result), price,
                                    bool(getattr(result, "ok", False)),
                                    str(getattr(result, "message", "")))
        except Exception as exc:
            log(f"成交流水没写进去 ({exc})")
        if getattr(result, "ok", False):
            account = _after(broker, account, order, price, log)
        out.append((intent, verdict, result))
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

# 高于这个距离，R 才是赔率；低于它，R 只是止损贴脸的算术结果。
STOP_TOO_CLOSE = 0.02
MIN_R = 1.5


def _stale_verdict(r_now: float, to_stop: float) -> str:
    # Through the stop is not "close to" the stop. The level was named as the
    # price at which the idea is wrong, and it has been passed: there is no R
    # left to quote and nothing to re-size. Folded into the "too close" branch
    # it read as a ratio worth a second look.
    if math.isfinite(to_stop) and to_stop <= 0:
        return "已经跌破当初的止损：这条作废，不是按新价再发一次"
    if math.isfinite(to_stop) and to_stop < STOP_TOO_CLOSE:
        return "离止损太近：R 高是止损贴脸算出来的，不是赔率好"
    if not math.isfinite(r_now):
        return "定不了价"
    if r_now < MIN_R:
        return f"已跌到 {MIN_R:.1f}R 之下，应作废而不是补单"
    return "R 仍在门槛之上，可按新价重新定量后再发"


def _lookup(rec: Reconciliation, rec_id: str):
    return getattr(rec, "_book", {}).get(rec_id)


def format_plan(rec: Reconciliation, account: Account, venue: str) -> str:
    W = 92
    out = ["=" * W, f"  账本 → 账户对账（{venue}）", "-" * W,
           f"  净值 ${account.account_value:,.2f}   现金 ${account.cash:,.2f}   "
           f"持仓 {len(account.holdings or [])} 个"]

    if rec.clean and not rec.unmanaged:
        out += ["", "  账本与账户一致，没有要下的单。"]
    if rec.to_close:
        out += ["", f"离场 ({len(rec.to_close)})", "-" * W]
        for i in rec.to_close:
            flag = "!" if i.urgency >= 3 else " "
            out.append(f" {flag}卖出 {i.symbol:<7}{i.shares:>7} 股   {i.reason[:58]}")
    if rec.to_trim:
        out += ["", f"减仓 ({len(rec.to_trim)})", "-" * W,
                "  账本已经把这些股数从这笔建议上扣掉了，账户还没有。它不是股数对不上，"
                "是今早自己发出的减仓指令。"]
        for i in rec.to_trim:
            out.append(f"  卖出 {i.symbol:<7}{i.shares:>7} 股   {i.reason[:58]}")
    if rec.to_open:
        out += ["", f"建仓 ({len(rec.to_open)})", "-" * W]
        for i in rec.to_open:
            px = f"限价 {i.limit:,.2f}" if i.limit else "市价"
            out.append(f"  买入 {i.symbol:<7}{i.shares:>7} 股   {px:<12} {i.reason[:44]}")
    if rec.stale:
        out += ["", f"过期未成交 ({len(rec.stale)})", "-" * W,
                "  这些不会下单。它们是按发出当天的收盘定的价：限价、止损、R 都指向一个",
                "  已经移动过的价格。止损不动而入场价抬高，等于用同样的下行空间买更少的",
                "  上行空间——这正是方法文件里记下的那次真实失效。",
                f"  {'代码':<6}{'放了':>5}{'发出时R':>9}{'现价':>10}{'现在R':>8}"
                f"{'距止损':>9}  处理"]
        for intent, age, r_now, px, to_stop in rec.stale:
            r0 = _num(getattr(_lookup(rec, intent.rec_id), "planned_r",
                              lambda: float("nan"))())
            rn = f"{r_now:.2f}" if math.isfinite(r_now) else "—"
            pxs = f"{px:,.2f}" if math.isfinite(px) else "—"
            ts = f"{to_stop * 100:+.1f}%" if math.isfinite(to_stop) else "—"
            out.append(f"  {intent.symbol:<6}{age:>4}天{r0:>9.2f}{pxs:>10}{rn:>8}"
                       f"{ts:>9}  {_stale_verdict(r_now, to_stop)}")
    if rec.drift:
        out += ["", f"股数对不上 ({len(rec.drift)})", "-" * W,
                "  这一节不自动改：股数差异可能是部分成交，也可能是手工动过，"
                "两者要用不同的方式处理。"]
        for sym, want, have in rec.drift:
            out.append(f"  {sym:<7} 账本 {want:>7} 股   账户 {have:>7} 股   "
                       f"差 {have - want:+}")
    if rec.matched:
        out += ["", f"一致 ({len(rec.matched)})", "-" * W,
                "  " + "  ".join(f"{s}×{n}" for s, n in rec.matched)]
    if rec.unmanaged:
        out += ["", f"账户里有、账本不认识 ({len(rec.unmanaged)})", "-" * W,
                "  这些不会被动。它们可能是手工下的单、别的策略、或者早先的演示数据——",
                "  仅仅因为不在这本账里就卖掉，等于这座桥认为整个账户都归它管。"]
        for h in rec.unmanaged:
            val = _num(getattr(h, "market_value", None), 0.0)
            out.append(f"  {h.symbol:<7}{_num(h.quantity, 0):>9,.0f} 股   "
                       f"成本 {_num(h.avg_cost, 0):>9,.2f}   市值 ${val:>11,.2f}")
    if rec.conflicts:
        out += ["", f"方向相反 ({len(rec.conflicts)})", "-" * W,
                "  账户和账本在同一只票上方向相反。这一节永远不下单：买进去是平掉空头，",
                "  卖出去是加深它，两个都是这座桥在替一笔它没有记录的交易做决定。"]
        for h, why in rec.conflicts:
            out.append(f"  {h.symbol:<7} {why}")
    if rec.core_held:
        out += ["", f"核心长仓 ({len(rec.core_held)})", "-" * W,
                "  这些在 core.json 里，按月复核，不归这本波段账管，也不会在这里下单。"]
        out.append("  " + "  ".join(
            f"{h.symbol}×{_num(h.quantity, 0):,.0f}" for h in rec.core_held))
    if rec.notes:
        out += ["", "说明", "-" * W] + [f"  - {n}" for n in rec.notes]
    out += ["", "=" * W]
    return "\n".join(out)


def format_results(results: list) -> str:
    if not results:
        return "  没有要提交的单。"
    out = [f"提交结果 ({len(results)})", "-" * 92]
    for intent, verdict, result in results:
        if not verdict.ok:
            out.append(f"  ✗ {intent.action} {intent.symbol:<7} 被风控拒绝：{verdict.reason}")
            continue
        qty = verdict.order.quantity if verdict.order else intent.shares
        trimmed = " (被改小)" if qty != intent.shares else ""
        if result is None:
            out.append(f"  ✗ {intent.action} {intent.symbol:<7} 没能提交")
        elif getattr(result, "ok", False):
            out.append(f"  ✓ {intent.action} {intent.symbol:<7}{qty:>7} 股{trimmed}   "
                       f"{getattr(result, 'message', '')}")
        else:
            out.append(f"  ✗ {intent.action} {intent.symbol:<7} 场所拒绝："
                       f"{getattr(result, 'message', '')}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _sessions() -> tuple[_date, _date]:
    """(the session being planned, the session the data comes from).

    Imported inside the call because :mod:`advisor` imports this module.
    """
    try:
        from .advisor import sessions_for
        return sessions_for(None, None)
    except Exception:
        today = _date.today()
        return today, today


def _core_symbols() -> list:
    """The long-term book's names, or none. Never raises."""
    try:
        from . import horizons
        return [h.symbol for h in horizons.load_core()]
    except Exception as exc:
        logger.info("读不到核心长仓名单（%s）；这次不区分它们", exc)
        return []


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="execute",
        description="把推荐账本和券商账户对账，并（可选地）下出补齐差异的单。")
    p.add_argument("--submit", action="store_true",
                   help="真的下单。不给这个参数就只打印会下什么")
    p.add_argument("--venue", default=None,
                   help="覆盖 TRADINGAGENTS_BROKER（alpaca / paper）")
    p.add_argument("--no-exits", action="store_true",
                   help="不算离场信号，只对账入场与股数差异（默认是算的）")
    p.add_argument("--with-exits", action="store_true",
                   help=argparse.SUPPRESS)          # 现在是默认行为，留着不报错
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if kill_switch_engaged():
        print("熔断开关已按下（~/.tradingagents/STOP），不做任何事。")
        return 3

    book = RecommendationBook()
    venue = (a.venue or os.getenv("TRADINGAGENTS_BROKER") or "alpaca").lower()
    try:
        broker = open_broker(venue)
    except Exception as exc:
        print(f"连不上 {venue}：{type(exc).__name__}: {exc}")
        return 2

    order_day, data_day = _sessions()

    with broker if hasattr(broker, "__enter__") else _null(broker) as b:
        try:
            account = b.account()
        except Exception as exc:
            print(f"读不到账户：{type(exc).__name__}: {exc}")
            return 2

        # Exits are computed by default. Off by default, the only thing this
        # bridge could ever do was open positions: the entries were read from
        # the book and the sells needed a flag nobody passed. A one-way bridge
        # is worse than none, because it looks like both.
        exits = None
        if not a.no_exits:
            prices = {}
            for rec in book.open_recommendations():
                px = _num(b.quote(rec.symbol), 0.0)
                if px > 0:
                    prices[rec.symbol] = px
            try:
                # persist=False: the advisor owns these records. Nothing here
                # may write to them.
                exits = book.review(prices, {}, as_of=data_day, persist=False)
            except Exception as exc:
                print(f"离场信号算不出来（{type(exc).__name__}: {exc}）；本次只对账入场")

        rec = plan(book, account, exits=exits, as_of=order_day, quote=b.quote,
                   core=_core_symbols())
        print()
        print(format_plan(rec, account, venue))

        if not a.submit:
            n = len(rec.intents)
            print(f"\n  这是对账，不是下单。要真的下这 {n} 笔，加 --submit。")
            return 0
        if not rec.intents:
            print("\n  没有要下的单。")
            return 0

        if not market_is_open(print):
            print("\n  现在不是常规交易时段。风控的 require_market_open 会逐笔拒掉，"
                  "\n  这是它该做的事——要在盘前排队下单，把它显式关掉"
                  "（TRADINGAGENTS_RISK_REQUIRE_MARKET_OPEN=false）。")

        secretary = Secretary(limits=RiskLimits.from_env(), ledger=TradeLedger())
        results = submit(rec, b, secretary, account)
        print()
        print(format_results(results))
    return 0


class _null:
    """Adapters that are not context managers still need `with`."""

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    sys.exit(main())
