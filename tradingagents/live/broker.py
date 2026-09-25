"""The venue contract: what the desk needs from a broker, and nothing more.

Everything above this line — triggers, the persona panel, the risk gate, the
news monitor, the screener — is venue-agnostic. It reasons about accounts,
positions and orders, not about browsers or REST clients. Pulling those types
out of any one adapter is what makes the venue a swappable detail rather than
an assumption baked through the stack.

That mattered immediately. The first adapter drove the Investopedia simulator
through a real browser, because Investopedia has no API and blocks scripted
clients. Its terms then turned out to prohibit exactly that (People Inc. ToS
§3.3(e), automated access; §3.3(f), using site data to develop software or
train a model), so the venue was replaced with Alpaca's paper API — which is
built for algorithmic trading and permits it explicitly. Nothing above this
module changed.

Two vocabularies are deliberately fixed here:

* **Actions** use the four-verb form (Buy / Sell / Sell Short / Buy to Cover)
  rather than a bare buy/sell pair. The distinction between "sell what I hold"
  and "open a short" cannot be recovered from a side alone, and an adapter that
  collapses them will one day sell a long position when it meant to short.
* **Quantities are shares**, and sizing happens above the broker. An adapter
  that sized its own orders would put the risk limits somewhere the Secretary
  could not enforce them.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# --- order vocabulary -------------------------------------------------------
BUY = "Buy"
SELL = "Sell"
SHORT = "Sell Short"
COVER = "Buy to Cover"
ACTIONS = (BUY, SELL, SHORT, COVER)

MARKET = "Market"
LIMIT = "Limit"
STOP = "Stop"
ORDER_TYPES = (MARKET, LIMIT, STOP)


@dataclass
class Holding:
    symbol: str
    quantity: float                 # always positive; direction is in `side`
    avg_cost: float = 0.0
    last: float = 0.0
    market_value: float = 0.0
    unrealized: float = 0.0
    side: str = "long"              # long | short


@dataclass
class Account:
    account_value: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    holdings: list[Holding] = field(default_factory=list)
    fetched_at: str = ""

    def position(self, symbol: str) -> Holding | None:
        return next((h for h in self.holdings if h.symbol == symbol.upper()), None)

    def to_dict(self) -> dict:
        return {
            "account_value": self.account_value, "cash": self.cash,
            "buying_power": self.buying_power, "fetched_at": self.fetched_at,
            "holdings": [h.__dict__ for h in self.holdings],
        }


@dataclass
class OrderResult:
    ok: bool
    symbol: str
    action: str
    quantity: float
    order_type: str = MARKET
    limit_price: float | None = None
    message: str = ""
    submitted_at: str = ""
    artifact: str = ""              # browser adapters: screenshot on failure

    # Populated by API-backed venues, which can answer "what happened to this
    # order" directly. A browser adapter has to infer it from page text and
    # leaves these empty — the difference is exactly why an API venue is
    # preferable, so it is worth representing rather than flattening away.
    broker_order_id: str = ""
    status: str = ""
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0

    @property
    def is_filled(self) -> bool:
        return self.filled_quantity > 0


@runtime_checkable
class Broker(Protocol):
    """What :mod:`monitor` requires. Adapters may offer more.

    Deliberately small. Every method here is one the loop actually calls; an
    adapter that satisfies these four can run the desk.
    """

    def is_logged_in(self) -> bool:
        """True when the venue is reachable and the account is tradeable."""

    def account(self) -> Account:
        """Cash, equity, buying power, and open positions."""

    def quote(self, symbol: str) -> float:
        """Last price, or 0.0 when none is available.

        Returning 0.0 rather than raising is intentional: an unpriceable symbol
        is a normal condition that the risk gate already rejects cleanly, and a
        raise here would abort a sweep over unrelated names.
        """

    def place_order(self, symbol: str, action: str, quantity: float,
                    order_type: str = MARKET, limit_price: float | None = None,
                    dry_run: bool = False) -> OrderResult:
        """Submit one order. Never raises; failure is reported in the result."""


# --- selection --------------------------------------------------------------

ALPACA = "alpaca"
INVESTOPEDIA = "investopedia"
PAPER = "paper"


def configured_venue() -> str:
    """Which venue the desk trades. Alpaca unless explicitly overridden.

    Alpaca is the default because it is the only one of the two whose terms
    permit an agent to trade it. The Investopedia adapter is kept because it
    works and because losing it would erase the reason the abstraction exists,
    but choosing it has to be a deliberate act.
    """
    return (os.getenv("TRADINGAGENTS_BROKER") or ALPACA).strip().lower()


def open_broker(venue: str | None = None, **kw):
    """Construct the configured adapter. Caller uses it as a context manager."""
    venue = (venue or configured_venue()).lower()
    if venue == ALPACA:
        from .alpaca import AlpacaBroker
        return AlpacaBroker(**{k: v for k, v in kw.items()
                               if k in ("paper", "api_key", "secret_key")})
    if venue == PAPER:
        from .paper import LocalPaperBroker
        return LocalPaperBroker(**{k: v for k, v in kw.items()
                                   if k in ("starting_cash", "path")})
    if venue == INVESTOPEDIA:
        from .investopedia import InvestopediaBroker
        return InvestopediaBroker(**{k: v for k, v in kw.items()
                                     if k in ("headless", "slow_mo", "timeout")})
    raise ValueError(f"unknown venue {venue!r}; expected one of "
                     f"{ALPACA}, {PAPER}, {INVESTOPEDIA}")
