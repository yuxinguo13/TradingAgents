"""Turn a rating into a sized order, scaled by holding horizon and volatility.

Two sleeves, because a position's right size depends on why you hold it:

* **core** — long-term convictions. Sized by conviction alone, held through
  drawdowns, reviewed occasionally.
* **tactical** — short-term trades. Sized by *risk*, not by dollars: the target
  weight is set so a one-ATR move costs roughly the same fraction of equity
  regardless of which name it is. A 10%-ATR name and a 2%-ATR name given the
  same dollar weight are not the same bet, and sizing them identically is the
  most common way a two-horizon book quietly becomes a volatility bet.

``Hold`` maps to ``None`` in both sleeves — deliberately distinct from a 0%
target — because "no strong view" must never be read as "liquidate".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv

from .portfolio import Fill, Portfolio

CORE = "core"
TACTICAL = "tactical"
SLEEVES = (CORE, TACTICAL)

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")

# --- core sleeve: conviction -> fixed target weight -------------------------
CORE_TARGETS: dict[str, float | None] = {
    "Buy":         0.25,
    "Overweight":  0.15,
    "Hold":        None,
    "Underweight": 0.05,
    "Sell":        0.0,
}

# --- tactical sleeve: conviction scales a volatility-targeted base ----------
TACTICAL_CONVICTION: dict[str, float | None] = {
    "Buy":         1.00,
    "Overweight":  0.60,
    "Hold":        None,
    "Underweight": 0.25,
    "Sell":        0.00,
}
# Equity risked per 1x ATR daily move on a full-conviction tactical position.
TACTICAL_DAILY_RISK = 0.004      # 0.4% of equity
# Per-name cap. Sized so the sleeve holds a basket (~8-10 names at 4-6%)
# rather than three concentrated bets — a screen-sourced tactical book is a
# diversified momentum tilt, and its edge comes from breadth, not conviction
# in any single name.
TACTICAL_MAX_WEIGHT = 0.06

# --- book-level guard rails -------------------------------------------------
MAX_GROSS_EXPOSURE = 0.95        # always keep some cash
MAX_TACTICAL_SLEEVE = 0.40       # tactical can never dominate the book

# Skip rebalances smaller than this; without a deadband the book churns on
# rounding drift every run and pays spread for nothing.
MIN_TRADE_FRACTION = 0.01
MIN_TRADE_VALUE = 50.0


def latest_close(ticker: str, date: str) -> float:
    """Last verified close on or before ``date``.

    Uses the same look-ahead-filtered loader the evidence brief is built from,
    so the fill price and the analysis agree by construction.
    """
    df = load_ohlcv(ticker, date)
    if df is None or df.empty:
        raise ValueError(f"No OHLCV data for {ticker} on or before {date}")
    return float(df.iloc[-1]["Close"])


def atr_pct(ticker: str, date: str, window: int = 14) -> float:
    """14-day Average True Range as a fraction of price.

    True range (not just high-low) so overnight gaps count — for a tactical
    book, gap risk is the risk that actually hurts.
    """
    df = load_ohlcv(ticker, date)
    if df is None or len(df) < window + 1:
        raise ValueError(f"Not enough history for ATR on {ticker}")
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(window).mean() / c.iloc[-1])


def target_weight(rating: str, sleeve: str, atr: float | None = None) -> float | None:
    """Target weight of total equity, or ``None`` for 'leave it alone'."""
    if sleeve == CORE:
        return CORE_TARGETS.get(rating)

    conviction = TACTICAL_CONVICTION.get(rating)
    if conviction is None or conviction == 0.0:
        return conviction  # None (Hold) or 0.0 (Sell) pass straight through
    if not atr or atr <= 0:
        # No volatility estimate means no risk-based size. Fall back to the
        # tightest allowed size rather than guessing large.
        return TACTICAL_MAX_WEIGHT * 0.25 * conviction
    return min(TACTICAL_MAX_WEIGHT, TACTICAL_DAILY_RISK / atr) * conviction


class Broker(Protocol):
    """Execution surface, implemented here only by :class:`PaperBroker`.

    Sizing lives in :func:`plan_trade` rather than inside the broker so that
    routing to a real venue is a new adapter, never an edit to the risk rules.
    """

    def execute(self, portfolio: Portfolio, ticker: str, rating: str,
                price: float, date: str, **kw) -> Fill | None: ...


@dataclass
class TradePlan:
    action: str            # BUY | SELL | NONE
    shares: float
    reason: str
    target: float | None = None


def _sleeve_exposure(portfolio: Portfolio, sleeve: str, prices: dict[str, float],
                     equity: float, exclude: str | None = None) -> float:
    if equity <= 0:
        return 0.0
    total = 0.0
    for tkr, pos in portfolio.positions.items():
        if tkr == exclude or getattr(pos, "sleeve", CORE) != sleeve:
            continue
        total += pos.market_value(prices.get(tkr, pos.avg_cost))
    return total / equity


def plan_trade(
    portfolio: Portfolio,
    ticker: str,
    rating: str,
    price: float,
    prices: dict[str, float],
    sleeve: str = CORE,
    atr: float | None = None,
) -> TradePlan:
    """Compute the order moving ``ticker`` to its sleeve-appropriate target."""
    if rating not in CORE_TARGETS:
        raise ValueError(f"unknown rating {rating!r}; expected one of {RATINGS}")
    if sleeve not in SLEEVES:
        raise ValueError(f"unknown sleeve {sleeve!r}; expected one of {SLEEVES}")

    target = target_weight(rating, sleeve, atr)
    if target is None:
        return TradePlan("NONE", 0.0, f"{rating} ({sleeve}): position unchanged")

    equity = portfolio.equity(prices)
    if equity <= 0:
        return TradePlan("NONE", 0.0, "no equity")

    # Guard rails: clamp the target so this trade cannot breach the tactical
    # sleeve cap or overall gross exposure. Clamping beats rejecting — a
    # partial move toward the target is still the right direction.
    if sleeve == TACTICAL:
        room = MAX_TACTICAL_SLEEVE - _sleeve_exposure(
            portfolio, TACTICAL, prices, equity, exclude=ticker)
        target = max(0.0, min(target, room))
    gross_other = sum(
        p.market_value(prices.get(t, p.avg_cost))
        for t, p in portfolio.positions.items() if t != ticker
    ) / equity
    target = max(0.0, min(target, MAX_GROSS_EXPOSURE - gross_other))

    pos = portfolio.position(ticker)
    delta_value = (equity * target) - (pos.shares * price)

    threshold = max(MIN_TRADE_VALUE, equity * MIN_TRADE_FRACTION)
    if abs(delta_value) < threshold:
        return TradePlan(
            "NONE", 0.0,
            f"{rating} ({sleeve}): within {MIN_TRADE_FRACTION:.0%} of "
            f"{target:.1%} target", target,
        )

    if delta_value > 0:
        spend = min(delta_value, portfolio.cash)
        shares = spend / price
        if shares * price < threshold:
            return TradePlan("NONE", 0.0, f"{rating} ({sleeve}): insufficient cash", target)
        return TradePlan("BUY", shares, f"{rating} ({sleeve}) → {target:.1%}", target)

    shares = min(abs(delta_value) / price, pos.shares)
    if shares <= 0:
        return TradePlan("NONE", 0.0, f"{rating} ({sleeve}): nothing held", target)
    return TradePlan("SELL", shares, f"{rating} ({sleeve}) → {target:.1%}", target)


class PaperBroker:
    """Fills at the verified close. No slippage, no commission, no real money."""

    def execute(
        self,
        portfolio: Portfolio,
        ticker: str,
        rating: str,
        price: float,
        date: str,
        prices: dict[str, float] | None = None,
        sleeve: str = CORE,
        atr: float | None = None,
        note: str = "",
    ) -> Fill | None:
        marks = dict(prices or {})
        marks[ticker] = price
        for tkr, pos in portfolio.positions.items():
            marks.setdefault(tkr, pos.avg_cost)

        plan = plan_trade(portfolio, ticker, rating, price, marks, sleeve=sleeve, atr=atr)
        if plan.action == "NONE":
            return None

        if plan.action == "BUY":
            portfolio.buy(ticker, plan.shares, price, sleeve=sleeve)
        else:
            portfolio.sell(ticker, plan.shares, price)

        reason = f"{plan.reason} | {note}" if note else plan.reason
        fill = Fill(date=date, ticker=ticker, action=plan.action,
                    shares=round(plan.shares, 6), price=price,
                    rating=rating, note=reason, sleeve=sleeve)
        portfolio.history.append(fill)
        return fill
