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
  strategy, a demo seed. Selling them because they are absent from one book is
  the bridge deciding it owns the whole account.
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
from dataclasses import dataclass, field
from datetime import date as _date

from . import clock
from .broker import BUY, SELL, Account, Holding, LIMIT, MARKET, open_broker
from .recommendations import Recommendation, RecommendationBook
from .secretary import Order, RiskLimits, Secretary, TradeLedger, kill_switch_engaged

logger = logging.getLogger(__name__)

# How far a venue position may differ from the book before it is called drift
# rather than rounding. One share: fractional fills and dividend reinvestment
# both produce sub-share differences that are not a reconciliation problem.
SHARE_TOLERANCE = 1.0

# How long an unfilled entry stays actionable, in calendar days from the date it
# was issued for. An idea is priced off one close and meant for the next open;
# after that its limit, its stop and its R all refer to a price that has moved.
#
# This is the failure the method document records happening for real: NRIX and
# NTRA were issued at 2.42R and 2.41R, both names rose ~2.6% before the open,
# and taking them at the old stop would have bought the same downside for less
# upside. A bridge that reads the book literally would place those orders six
# days later at a six-day-old limit and never mention it — so stale entries are
# reported in their own section with R recomputed at the current price, and are
# never submitted.
ENTRY_FRESH_DAYS = 1


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
    # Entries whose levels have gone stale: (Intent, days_old, R_now).
    # Reported, never submitted — see ENTRY_FRESH_DAYS.
    stale: list = field(default_factory=list)
    drift: list = field(default_factory=list)        # (symbol, book_shares, venue_shares)
    matched: list = field(default_factory=list)      # (symbol, shares)
    unmanaged: list = field(default_factory=list)    # Holding
    notes: list = field(default_factory=list)

    @property
    def intents(self) -> list:
        """Exits first: the proceeds of a sale are what fund a purchase, and a
        bridge that buys before it sells can be rejected for cash it is about
        to have."""
        return sorted(self.to_close, key=lambda i: -i.urgency) + self.to_open
        # 过期未成交刻意不在这里：它们要重新定量，不是补单

    @property
    def clean(self) -> bool:
        return not (self.to_open or self.to_close or self.drift or self.stale)


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


def plan(book: RecommendationBook, account: Account, *,
         exits=None, as_of: _date | None = None, quote=None,
         fresh_days: int = ENTRY_FRESH_DAYS) -> Reconciliation:
    """Compare the open book to the account and say what would close the gap.

    ``exits`` is the exit-signal list the advisor already computed for today;
    passing it keeps one exit engine rather than two. Without it, the plan
    covers entries and drift only, and says so.
    """
    out = Reconciliation()
    out._book = {}
    held = {h.symbol.upper(): h for h in (account.holdings or [])}
    open_recs = [r for r in book.open_recommendations() if r.symbol]
    booked = {r.symbol.upper(): r for r in open_recs}

    closing = {}
    for sig in (exits or []):
        if getattr(sig, "closes_position", False):
            closing[sig.symbol.upper()] = sig
    if exits is None:
        out.notes.append("没有传入离场信号，本次只对账入场与股数差异；"
                         "离场要由 advisor 的规则算出来，不能在这里另起一套")

    for sym, rec in booked.items():
        out._book[rec.id] = rec
        want = int(_num(rec.shares, 0.0))
        h = held.get(sym)
        have = int(_num(getattr(h, "quantity", 0.0), 0.0)) if h else 0
        sig = closing.get(sym)
        if sig is not None:
            if have > 0:
                out.to_close.append(Intent(
                    SELL, sym, min(have, int(_num(sig.shares, have))),
                    None, f"{sig.action}：{sig.reason}", rec.id,
                    int(_num(getattr(sig, "urgency", 1), 1))))
            else:
                out.notes.append(f"{sym}：账本要离场，但账户里本来就没有仓位")
            continue
        if want <= 0:
            continue
        if have == 0:
            intent = Intent(BUY, sym, want, _num(rec.limit_price, None) or None,
                            f"账本 {rec.issued_date} 发出，尚未建仓", rec.id)
            age = _age_days(rec, as_of)
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

    for sym, h in held.items():
        if sym not in booked:
            out.unmanaged.append(h)
    return out


def submit(rec: Reconciliation, broker, secretary: Secretary, account: Account,
           *, market_open: bool | None = None, log=logger.info) -> list:
    """Vet every intent through the Secretary, then place what survives.

    Returns ``(Intent, Verdict, OrderResult | None)`` triples so the caller can
    print exactly what happened to each one — including the ones the gate
    resized or refused, which are the interesting rows.
    """
    if market_open is None:
        market_open = clock.is_market_open() if hasattr(clock, "is_market_open") else True
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
            secretary.ledger.record(order, price, bool(getattr(result, "ok", False)),
                                    str(getattr(result, "message", "")))
            secretary.ledger.save()
        except Exception as exc:
            log(f"成交流水没写进去 ({exc})")
        out.append((intent, verdict, result))
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

# 高于这个距离，R 才是赔率；低于它，R 只是止损贴脸的算术结果。
STOP_TOO_CLOSE = 0.02
MIN_R = 1.5


def _stale_verdict(r_now: float, to_stop: float) -> str:
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

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="execute",
        description="把推荐账本和券商账户对账，并（可选地）下出补齐差异的单。")
    p.add_argument("--submit", action="store_true",
                   help="真的下单。不给这个参数就只打印会下什么")
    p.add_argument("--venue", default=None,
                   help="覆盖 TRADINGAGENTS_BROKER（alpaca / paper）")
    p.add_argument("--with-exits", action="store_true",
                   help="同时算今天的离场信号（会按上一收盘给账本定价）")
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

    with broker if hasattr(broker, "__enter__") else _null(broker) as b:
        try:
            account = b.account()
        except Exception as exc:
            print(f"读不到账户：{type(exc).__name__}: {exc}")
            return 2

        exits = None
        if a.with_exits:
            try:
                data_day = clock.last_completed_session() if hasattr(
                    clock, "last_completed_session") else _date.today()
            except Exception:
                data_day = _date.today()
            try:
                from .advisor import last_completed_session
                data_day = last_completed_session()
            except Exception:
                pass
            prices = {}
            for rec in book.open_recommendations():
                px = _num(b.quote(rec.symbol), 0.0)
                if px > 0:
                    prices[rec.symbol] = px
            try:
                exits = book.review(prices, {}, as_of=data_day, persist=False)
            except Exception as exc:
                print(f"离场信号算不出来（{type(exc).__name__}: {exc}）；本次只对账入场")

        try:
            from .advisor import last_completed_session
            today = last_completed_session()
        except Exception:
            today = _date.today()
        rec = plan(book, account, exits=exits, as_of=today, quote=b.quote)
        print()
        print(format_plan(rec, account, venue))

        if not a.submit:
            n = len(rec.intents)
            print(f"\n  这是对账，不是下单。要真的下这 {n} 笔，加 --submit。")
            return 0
        if not rec.intents:
            print("\n  没有要下的单。")
            return 0

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
