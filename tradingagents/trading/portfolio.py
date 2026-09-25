"""Persistent paper portfolio: cash, positions, and a fill history.

The graph emits a *rating*, not an order. This module supplies the state an
order needs to exist against — what is already held, at what cost, and how much
cash is free — so a rating can be turned into a sized trade.

State is a single JSON document so a run is inspectable and hand-editable
between sessions. Path defaults to ``~/.tradingagents/portfolio.json`` and is
overridable with ``TRADINGAGENTS_PORTFOLIO_PATH``, mirroring how the decision
log takes ``TRADINGAGENTS_MEMORY_LOG_PATH``.

Long-only by construction: shares never go negative and cash never goes below
zero, so no code path can silently open a short or lever the book.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_STARTING_CASH = 100_000.0


def default_portfolio_path() -> Path:
    env = os.getenv("TRADINGAGENTS_PORTFOLIO_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".tradingagents" / "portfolio.json"


@dataclass
class Position:
    shares: float = 0.0
    avg_cost: float = 0.0
    # Which horizon this position is held on. Defaults to "core" so portfolios
    # written before sleeves existed still load.
    sleeve: str = "core"

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized(self, price: float) -> float:
        return (price - self.avg_cost) * self.shares


@dataclass
class Fill:
    date: str
    ticker: str
    action: str          # BUY | SELL
    shares: float
    price: float
    rating: str
    note: str = ""
    sleeve: str = "core"


@dataclass
class Portfolio:
    cash: float = DEFAULT_STARTING_CASH
    starting_cash: float = DEFAULT_STARTING_CASH
    positions: dict[str, Position] = field(default_factory=dict)
    history: list[Fill] = field(default_factory=list)

    # --- valuation ---

    def position(self, ticker: str) -> Position:
        return self.positions.get(ticker, Position())

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus marked-to-market positions.

        A ticker missing from ``prices`` is held at cost rather than dropped —
        silently valuing it at zero would understate equity and cause the
        sizer to over-allocate everything else.
        """
        held = 0.0
        for tkr, pos in self.positions.items():
            held += pos.market_value(prices.get(tkr, pos.avg_cost))
        return self.cash + held

    def weight(self, ticker: str, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        pos = self.position(ticker)
        return pos.market_value(prices.get(ticker, pos.avg_cost)) / eq

    # --- mutation ---

    def buy(self, ticker: str, shares: float, price: float, sleeve: str | None = None) -> None:
        cost = shares * price
        if shares <= 0:
            raise ValueError(f"buy shares must be positive, got {shares}")
        if cost > self.cash + 1e-9:
            raise ValueError(f"insufficient cash: need {cost:.2f}, have {self.cash:.2f}")
        pos = self.positions.setdefault(ticker, Position())
        if sleeve:
            # A name re-rated into a different sleeve adopts the new horizon;
            # its risk limits should follow the reason it is now held.
            pos.sleeve = sleeve
        total = pos.shares + shares
        # Weighted-average cost basis, so unrealized P&L stays meaningful
        # across multiple adds rather than resetting to the latest price.
        pos.avg_cost = ((pos.avg_cost * pos.shares) + cost) / total
        pos.shares = total
        self.cash -= cost

    def sell(self, ticker: str, shares: float, price: float) -> float:
        """Sell ``shares`` and return realized P&L. Cost basis is unchanged."""
        pos = self.positions.get(ticker)
        if pos is None or shares > pos.shares + 1e-9:
            have = 0.0 if pos is None else pos.shares
            raise ValueError(f"cannot sell {shares} of {ticker}; hold {have}")
        realized = (price - pos.avg_cost) * shares
        pos.shares -= shares
        self.cash += shares * price
        if pos.shares <= 1e-9:
            del self.positions[ticker]
        return realized

    # --- persistence ---

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "starting_cash": self.starting_cash,
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "history": [asdict(f) for f in self.history],
        }

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else default_portfolio_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Temp-file + replace so an interrupted write can't truncate the book,
        # matching the atomic-write discipline in the decision log.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None, starting_cash: float = DEFAULT_STARTING_CASH):
        path = Path(path) if path else default_portfolio_path()
        if not path.exists():
            return cls(cash=starting_cash, starting_cash=starting_cash)
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            cash=d.get("cash", starting_cash),
            starting_cash=d.get("starting_cash", starting_cash),
            positions={t: Position(**p) for t, p in d.get("positions", {}).items()},
            history=[Fill(**f) for f in d.get("history", [])],
        )
