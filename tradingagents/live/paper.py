"""Local paper broker: the whole desk, real prices, no account anywhere.

This exists because a venue you have to be admitted to is a dependency, and
dependencies fail. Investopedia's terms turned out to forbid automated access;
Alpaca's onboarding can strand a new user on a "select an account type" screen
that blocks the dashboard (alpaca-py#659, open). Neither is a reason to be
unable to watch the agent trade.

So this adapter satisfies the same :class:`~.broker.Broker` contract against a
book kept on disk. Prices are real — the same look-ahead-filtered loader the
rest of the framework uses, with a live intraday quote when yfinance can serve
one — so the P&L is a genuine read on the strategy. What is simulated is only
the venue: fills are immediate and complete at the quoted price.

That last point is the honest limitation and it cuts one way. There is no
slippage, no partial fill, no queue position, no borrow check. A strategy that
depends on getting filled at the touch will look better here than it would
anywhere real. For judging whether the agent's *reasoning* is any good, that is
an acceptable trade; for judging execution, it is not.

State reuses :class:`tradingagents.trading.portfolio.Portfolio` — the same
atomic-write, long-only book the offline desk already runs on, so a run here
and a run there are directly comparable.

    TRADINGAGENTS_BROKER=paper python -m tradingagents.live.cli portfolio
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from .broker import (
    ACTIONS, BUY, COVER, LIMIT, MARKET, SELL, SHORT, STOP,
    Account, Holding, OrderResult,
)

logger = logging.getLogger(__name__)

DEFAULT_STARTING_CASH = 100_000.0


def portfolio_path() -> Path:
    """Separate from the offline desk's book by default.

    The two are the same shape but not the same experiment; sharing one file
    would silently merge an LLM-driven run with a hand-rated one and make the
    resulting track record mean nothing.
    """
    env = os.getenv("TRADINGAGENTS_LIVE_PORTFOLIO_PATH")
    if env:
        return Path(env).expanduser()
    home = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
    return home / "live_portfolio.json"


class LocalPaperBroker:
    """Fills against real quotes, books into a local portfolio file."""

    def __init__(self, starting_cash: float = DEFAULT_STARTING_CASH,
                 path: Path | None = None, **_ignored):
        from tradingagents.trading.portfolio import Portfolio
        self.path = Path(path) if path else portfolio_path()
        self.portfolio = Portfolio.load(self.path, starting_cash=starting_cash)
        self._quotes: dict[str, float] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Persist on the way out so an interrupted session keeps its fills.
        try:
            self.portfolio.save(self.path)
        except Exception as exc:
            logger.error("could not save the book: %s", exc)
        return False

    # --- account ------------------------------------------------------------

    def is_logged_in(self) -> bool:
        """Always true — there is nothing to authenticate against."""
        return True

    def account(self) -> Account:
        pf = self.portfolio
        holdings: list[Holding] = []
        marks: dict[str, float] = {}
        for tkr, pos in pf.positions.items():
            px = self.quote(tkr) or pos.avg_cost
            marks[tkr] = px
            holdings.append(Holding(
                symbol=tkr, quantity=pos.shares, avg_cost=pos.avg_cost,
                last=px, market_value=pos.market_value(px),
                unrealized=pos.unrealized(px), side="long",
            ))
        equity = pf.equity(marks)
        return Account(
            account_value=equity, cash=pf.cash,
            # Long-only and unlevered, so cash is the whole of buying power.
            buying_power=pf.cash,
            holdings=holdings, fetched_at=datetime.now().isoformat(),
        )

    # --- pricing ------------------------------------------------------------

    def quote(self, symbol: str) -> float:
        """Live intraday price if yfinance will serve one, else the last close.

        Cached per instance: one cycle prices the same symbol several times
        (sizing, marking, the fill) and each of those hitting the network would
        make a sweep crawl and risk a rate limit.
        """
        sym = symbol.upper()
        if sym in self._quotes:
            return self._quotes[sym]

        px = 0.0
        try:
            import yfinance as yf
            fi = yf.Ticker(sym).fast_info
            px = float(getattr(fi, "last_price", 0) or 0)
        except Exception:
            px = 0.0

        if px <= 0:
            try:
                from tradingagents.dataflows.stockstats_utils import load_ohlcv
                from . import clock
                df = load_ohlcv(sym, clock.last_trading_day().isoformat())
                if df is not None and not df.empty:
                    px = float(df.iloc[-1]["Close"])
            except Exception as exc:
                logger.debug("could not price %s: %s", sym, exc)

        if px > 0:
            self._quotes[sym] = px
        return px

    def refresh_quotes(self) -> None:
        """Drop the price cache. The loop calls this once per cycle."""
        self._quotes.clear()

    # --- orders -------------------------------------------------------------

    def place_order(self, symbol: str, action: str, quantity: float,
                    order_type: str = MARKET, limit_price: float | None = None,
                    dry_run: bool = False) -> OrderResult:
        symbol = symbol.upper()
        res = OrderResult(ok=False, symbol=symbol, action=action, quantity=quantity,
                          order_type=order_type, limit_price=limit_price,
                          submitted_at=datetime.now().isoformat())
        if action not in ACTIONS:
            res.message = f"unknown action {action!r}; expected one of {ACTIONS}"
            return res
        if action in (SHORT, COVER):
            # The underlying Portfolio cannot represent a negative position, so
            # a short would have to be faked. Refusing is the honest answer.
            res.message = "local paper book is long-only; shorting is unsupported"
            return res
        qty = int(quantity)
        if qty <= 0:
            res.message = f"quantity rounds to zero ({quantity})"
            return res

        price = self.quote(symbol)
        if price <= 0:
            res.message = f"no usable price for {symbol}"
            return res
        # A limit better than the market is treated as marketable and fills at
        # the limit; anything else would need a resting-order book this does
        # not have, and pretending otherwise would invent fills.
        if order_type in (LIMIT, STOP) and limit_price:
            if action == BUY and limit_price < price:
                res.message = (f"limit {limit_price:,.2f} is below the market "
                               f"{price:,.2f}; no resting orders in the local book")
                return res
            if action == SELL and limit_price > price:
                res.message = (f"limit {limit_price:,.2f} is above the market "
                               f"{price:,.2f}; no resting orders in the local book")
                return res
            price = limit_price

        if dry_run:
            res.ok = True
            res.message = f"DRY RUN — would {action} {qty} {symbol} @ {price:,.2f}"
            return res

        try:
            if action == BUY:
                cost = qty * price
                if cost > self.portfolio.cash + 1e-9:
                    # Trim to what the cash actually buys rather than failing:
                    # the risk gate already approved the direction.
                    qty = int(self.portfolio.cash // price)
                    if qty <= 0:
                        res.message = (f"insufficient cash "
                                       f"(${self.portfolio.cash:,.2f} < ${price:,.2f})")
                        return res
                self.portfolio.buy(symbol, qty, price)
            else:
                held = self.portfolio.position(symbol).shares
                if held <= 0:
                    res.message = f"no {symbol} position to sell"
                    return res
                qty = min(qty, int(held))
                if qty <= 0:
                    res.message = f"{symbol} position too small to sell a whole share"
                    return res
                self.portfolio.sell(symbol, qty, price)
        except Exception as exc:
            res.message = f"{type(exc).__name__}: {exc}"
            return res

        from tradingagents.trading.portfolio import Fill
        self.portfolio.history.append(Fill(
            date=datetime.now().strftime("%Y-%m-%d"), ticker=symbol,
            action="BUY" if action == BUY else "SELL", shares=qty, price=price,
            rating="", note="live desk", sleeve="core",
        ))
        self.portfolio.save(self.path)

        res.ok = True
        res.quantity = qty
        res.status = "filled"
        res.filled_quantity = qty
        res.filled_avg_price = price
        res.message = f"filled {qty} @ {price:,.2f}"
        return res

    def open_orders(self) -> list[dict]:
        """Always empty: every fill here is immediate."""
        return []

    # --- reporting ----------------------------------------------------------

    def pnl(self) -> dict:
        acct = self.account()
        start = self.portfolio.starting_cash
        pnl = acct.account_value - start
        return {
            "starting_cash": start,
            "equity": acct.account_value,
            "cash": acct.cash,
            "pnl": pnl,
            "pnl_pct": (pnl / start) if start else 0.0,
            "positions": len(acct.holdings),
            "fills": len(self.portfolio.history),
        }
