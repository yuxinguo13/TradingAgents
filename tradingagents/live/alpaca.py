"""Alpaca paper-trading adapter — the sanctioned venue.

Alpaca exists to be traded by software: paper accounts are free, the keys are
issued instantly, and algorithmic access is the product rather than something
tolerated. That is the whole reason this adapter replaced the browser-driven
Investopedia one, whose terms prohibit automated access outright.

Practically it is also a better venue to reason against. The browser adapter
had to infer an order's fate from whatever text the page rendered, and its
honest failure mode was "submitted but no confirmation text found — verify
manually". Here an order comes back with an id, a status and a fill quantity,
so the book and the venue cannot silently disagree.

Two details are load-bearing and easy to get wrong:

**Every numeric field Alpaca returns is a string**, not a number — ``cash``,
``qty``, ``avg_entry_price``, all of them. Arithmetic on them silently
concatenates or raises depending on the operator, so everything goes through
:func:`_f`.

**Alpaca's ``OrderSide`` has only BUY and SELL.** Mapping "Sell Short" onto
``SELL`` and stopping there means that shorting a name you already hold long
would *sell the long instead* — the position flips to flat and no short is
opened, with the order reporting success. ``PositionIntent`` is what
disambiguates, and both fields must be sent: the SDK's own validator rejects an
intent without a side.

    export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
    python -m tradingagents.live.cli portfolio
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from datetime import datetime

from .broker import (
    ACTIONS, BUY, COVER, LIMIT, MARKET, SELL, SHORT, STOP,
    Account, Holding, OrderResult,
)

logger = logging.getLogger(__name__)

PAPER_SIGNUP_URL = "https://app.alpaca.markets/signup"

# Alpaca speaks buy/sell; the desk speaks four verbs. position_intent carries
# the half that a side cannot express. See the module docstring.
_INTENT: dict[str, tuple[str, str]] = {
    BUY:   ("BUY",  "BUY_TO_OPEN"),
    SELL:  ("SELL", "SELL_TO_CLOSE"),
    SHORT: ("SELL", "SELL_TO_OPEN"),
    COVER: ("BUY",  "BUY_TO_CLOSE"),
}


def _f(x, default: float = 0.0) -> float:
    """Alpaca returns numbers as strings; nothing may skip this."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class MissingCredentials(RuntimeError):
    pass


def credentials() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise MissingCredentials(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set.\n"
            f"Create a free paper account at {PAPER_SIGNUP_URL}, generate paper "
            "keys, then add them to your .env:\n"
            "  ALPACA_API_KEY=PK...\n"
            "  ALPACA_SECRET_KEY=..."
        )
    return key, sec


class AlpacaBroker:
    """Satisfies :class:`~.broker.Broker` against Alpaca's paper endpoint.

    ``paper`` defaults to True and there is no code path that flips it from
    configuration. Trading real money should require editing this file, which
    is a decision, rather than setting an environment variable, which is an
    accident waiting to happen.
    """

    def __init__(self, paper: bool = True, api_key: str | None = None,
                 secret_key: str | None = None, **_ignored):
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        if api_key and secret_key:
            key, sec = api_key, secret_key
        else:
            key, sec = credentials()
        self.paper = paper
        self.trading = TradingClient(key, sec, paper=paper)
        self.data = StockHistoricalDataClient(key, sec)

    # Context-manager shape so the monitor can hold either adapter identically.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # --- account ------------------------------------------------------------

    def is_logged_in(self) -> bool:
        """Reachable, authenticated, and permitted to trade.

        A blocked account is reported as not-logged-in rather than as an error:
        from the loop's point of view "cannot trade here" is one condition, and
        splitting it would only add a branch that does the same thing.
        """
        try:
            acct = self.trading.get_account()
        except Exception as exc:
            logger.error("Alpaca account check failed: %s", exc)
            return False
        if getattr(acct, "account_blocked", False) or getattr(acct, "trading_blocked", False):
            logger.error("Alpaca account is blocked from trading")
            return False
        return True

    def account(self) -> Account:
        from alpaca.trading.enums import PositionSide
        a = self.trading.get_account()
        positions = self.trading.get_all_positions()
        return Account(
            # equity is the marked value including positions; portfolio_value is
            # its documented alias and is used only if equity is absent.
            account_value=_f(a.equity) or _f(a.portfolio_value),
            cash=_f(a.cash),
            buying_power=_f(a.buying_power),
            fetched_at=datetime.now().isoformat(),
            holdings=[
                Holding(
                    symbol=p.symbol,
                    # Alpaca signs both qty and market_value negative for a
                    # short. The desk keeps size positive and puts direction in
                    # `side`, so a short does not subtract from gross exposure.
                    quantity=abs(_f(p.qty)),
                    avg_cost=_f(p.avg_entry_price),
                    last=_f(p.current_price),
                    market_value=abs(_f(p.market_value)),
                    unrealized=_f(p.unrealized_pl),
                    side="short" if p.side == PositionSide.SHORT else "long",
                )
                for p in positions
            ],
        )

    # --- pricing ------------------------------------------------------------

    def quote(self, symbol: str) -> float:
        """Last price, with a fallback chain, or 0.0.

        The free data plan serves IEX only, which is a few percent of US volume.
        For a liquid name that is fine; for a thin one the last IEX print can be
        hours old or missing entirely, and a 0.0 here makes the risk gate reject
        the trade with "no usable price". So: last trade, then the bid/ask
        midpoint, then the daily bar the rest of the framework already caches.
        """
        from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
        sym = symbol.upper()

        with suppress(Exception):
            r = self.data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=sym))
            px = _f(r[sym].price)
            if px > 0:
                return px

        with suppress(Exception):
            q = self.data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
            bid, ask = _f(q.bid_price), _f(q.ask_price)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            if ask > 0 or bid > 0:
                return ask or bid

        # Last resort: the look-ahead-filtered daily loader. Stale by up to a
        # session, but a stale price the gate can sanity-check beats no price.
        with suppress(Exception):
            from tradingagents.dataflows.stockstats_utils import load_ohlcv
            from . import clock
            df = load_ohlcv(sym, clock.last_trading_day().isoformat())
            if df is not None and not df.empty:
                logger.debug("%s priced from the daily bar (no live IEX print)", sym)
                return float(df.iloc[-1]["Close"])

        logger.warning("no price available for %s", sym)
        return 0.0

    # --- orders -------------------------------------------------------------

    def _build_request(self, symbol: str, action: str, quantity: float,
                       order_type: str, limit_price: float | None):
        from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest, MarketOrderRequest, StopOrderRequest,
        )
        side_name, intent_name = _INTENT[action]
        common = dict(
            symbol=symbol,
            qty=quantity,
            side=OrderSide[side_name],
            position_intent=PositionIntent[intent_name],
            # DAY, not GTC: an order the desk placed on a thesis it formed this
            # morning should not quietly fill days later on a different one.
            time_in_force=TimeInForce.DAY,
        )
        if order_type == LIMIT:
            return LimitOrderRequest(limit_price=limit_price, **common)
        if order_type == STOP:
            return StopOrderRequest(stop_price=limit_price, **common)
        return MarketOrderRequest(**common)

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
        if quantity <= 0:
            res.message = f"non-positive quantity ({quantity})"
            return res
        if order_type in (LIMIT, STOP) and not limit_price:
            res.message = f"{order_type} order needs a price"
            return res

        try:
            req = self._build_request(symbol, action, quantity, order_type, limit_price)
        except Exception as exc:
            res.message = f"could not build order: {type(exc).__name__}: {exc}"
            return res

        if dry_run:
            res.ok = True
            with suppress(Exception):
                res.message = f"DRY RUN — {req.model_dump(exclude_none=True)}"
            res.message = res.message or "DRY RUN — order built, not submitted"
            return res

        try:
            o = self.trading.submit_order(req)
        except Exception as exc:
            # Insufficient buying power, a halted or non-tradeable symbol, and
            # wash-trade rejections all land here. None of them should stop the
            # sweep over the other names.
            res.message = f"rejected by Alpaca: {exc}"
            logger.warning("order rejected for %s: %s", symbol, exc)
            return res

        status = getattr(o.status, "value", str(o.status))
        res.broker_order_id = str(o.id)
        res.status = status
        res.filled_quantity = _f(o.filled_qty)
        res.filled_avg_price = _f(o.filled_avg_price)
        # An accepted-but-unfilled market order is a success: it is working, and
        # a fill notification is a separate event. Only a terminal negative
        # status is a failure.
        res.ok = status not in ("rejected", "canceled", "expired")
        res.message = (f"{status} ({res.filled_quantity:g}/{_f(o.qty):g} filled"
                       + (f" @ {res.filled_avg_price:,.2f}" if res.filled_avg_price else "")
                       + ")")
        return res

    def open_orders(self) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        try:
            orders = self.trading.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except Exception as exc:
            logger.warning("could not list open orders: %s", exc)
            return []
        out = []
        for o in orders:
            out.append({
                "id": str(o.id), "symbol": o.symbol,
                "side": getattr(o.side, "value", str(o.side)),
                "qty": _f(o.qty), "filled": _f(o.filled_qty),
                "type": getattr(o.order_type, "value", str(o.order_type)),
                "status": getattr(o.status, "value", str(o.status)),
                "submitted_at": str(getattr(o, "submitted_at", "")),
            })
        return out

    # --- extras the browser adapter could not offer -------------------------

    def market_open(self) -> bool:
        """The venue's own clock — authoritative over a local calendar."""
        try:
            return bool(self.trading.get_clock().is_open)
        except Exception:
            return False

    def cancel(self, order_id: str) -> bool:
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.warning("could not cancel %s: %s", order_id, exc)
            return False

    def cancel_all(self) -> int:
        try:
            return len(self.trading.cancel_orders() or [])
        except Exception:
            return 0

    def history(self, period: str = "1M", timeframe: str = "1D") -> list[dict]:
        """Equity curve straight from the venue — what the account actually did.

        Worth preferring over a locally reconstructed curve: it already accounts
        for fills the desk did not make, and it is the number the user sees on
        Alpaca's own dashboard, so the two cannot disagree.
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            h = self.trading.get_portfolio_history(
                GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
        except Exception as exc:
            logger.warning("portfolio history unavailable: %s", exc)
            return []
        ts = getattr(h, "timestamp", None) or []
        eq = getattr(h, "equity", None) or []
        pl = getattr(h, "profit_loss", None) or []
        out = []
        for i, t in enumerate(ts):
            out.append({
                "timestamp": t,
                "equity": _f(eq[i]) if i < len(eq) else 0.0,
                "profit_loss": _f(pl[i]) if i < len(pl) else 0.0,
            })
        return out
