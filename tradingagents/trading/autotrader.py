"""Autonomous trading loop: analyse a watchlist, size each call, execute, persist.

This is the piece that makes the framework self-directed rather than a
question-answering tool. One ``run_once`` pass walks the watchlist, runs the
full agent graph per ticker, converts each rating into a trade against the
persistent paper portfolio, and marks the book to market.

The learning loop comes for free: ``propagate`` resolves that ticker's pending
decision-log entries before analysing, so every pass scores the previous call
against realised alpha and feeds the reflection into the Portfolio Manager's
prompt. Run it repeatedly on the same watchlist and it accumulates a track
record it actually reads.

Paper trading only. See :mod:`tradingagents.trading.execution` — routing to a
real venue means writing a new :class:`~.execution.Broker` adapter, which is a
deliberate act, not a config flag.

    python -m tradingagents.trading.autotrader --watchlist QQQM,SPY
    python -m tradingagents.trading.autotrader --status
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date as _date

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .execution import RATINGS, PaperBroker, latest_close
from .portfolio import Fill, Portfolio

logger = logging.getLogger(__name__)


@dataclass
class TickerResult:
    ticker: str
    rating: str | None = None
    price: float | None = None
    fill: Fill | None = None
    error: str | None = None


class AutoTrader:
    def __init__(
        self,
        watchlist: list[str],
        config: dict | None = None,
        analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals"),
        portfolio_path=None,
        starting_cash: float = 100_000.0,
    ):
        self.watchlist = [t.strip().upper() for t in watchlist if t.strip()]
        self.config = config or DEFAULT_CONFIG.copy()
        self.analysts = analysts
        self.portfolio_path = portfolio_path
        self.portfolio = Portfolio.load(portfolio_path, starting_cash=starting_cash)
        self.broker = PaperBroker()

    # --- pricing ---

    def mark_prices(self, date: str) -> dict[str, float]:
        """Latest verified close for everything held or watched.

        A ticker that fails to price is simply absent; ``Portfolio.equity``
        falls back to cost basis for those rather than treating them as zero.
        """
        prices: dict[str, float] = {}
        for tkr in set(self.watchlist) | set(self.portfolio.positions):
            try:
                prices[tkr] = latest_close(tkr, date)
            except Exception as exc:
                logger.warning("Could not price %s: %s", tkr, exc)
        return prices

    # --- main loop ---

    def run_once(self, date: str, dry_run: bool = False) -> list[TickerResult]:
        results: list[TickerResult] = []
        # One mark pass up front so every ticker in the sweep is sized against
        # the same, live valuation of the book.
        marks = self.mark_prices(date)
        for tkr in self.watchlist:
            res = TickerResult(ticker=tkr)
            try:
                # A fresh graph per ticker: TradingAgentsGraph carries per-run
                # state (self.ticker, checkpoint context), so reusing one
                # instance across tickers would leak state between them.
                graph = TradingAgentsGraph(
                    selected_analysts=list(self.analysts),
                    debug=False,
                    config=self.config,
                )
                _state, decision = graph.propagate(tkr, date)
                res.rating = parse_rating(decision)
                res.price = latest_close(tkr, date)
                marks[tkr] = res.price

                if not dry_run:
                    res.fill = self.broker.execute(
                        self.portfolio, tkr, res.rating, res.price, date, prices=marks
                    )
                    # Persist after each ticker so an interrupted sweep keeps
                    # the fills it already made rather than losing the batch.
                    self.portfolio.save(self.portfolio_path)
            except Exception as exc:
                # One bad ticker must not abort the sweep.
                logger.exception("Analysis failed for %s", tkr)
                res.error = str(exc)
            results.append(res)
        return results

    # --- externally-supplied ratings ---

    def apply_rating(
        self, ticker: str, rating: str, date: str, note: str = ""
    ) -> TickerResult:
        """Execute a rating produced outside the graph.

        The sizing and portfolio layers never cared where a rating came from,
        so an analyst reading the evidence brief directly drives the same book,
        with the same risk limits, as the LLM pipeline would. This is the seam
        that makes the reasoning engine swappable.
        """
        res = TickerResult(ticker=ticker, rating=rating)
        try:
            marks = self.mark_prices(date)
            res.price = latest_close(ticker, date)
            marks[ticker] = res.price
            res.fill = self.broker.execute(
                self.portfolio, ticker, rating, res.price, date, prices=marks
            )
            if res.fill is not None and note:
                res.fill.note = f"{res.fill.note} | {note}"
            self.portfolio.save(self.portfolio_path)
        except Exception as exc:
            logger.exception("Could not apply rating for %s", ticker)
            res.error = str(exc)
        return res

    # --- reporting ---

    def summary(self, date: str) -> str:
        prices = self.mark_prices(date)
        eq = self.portfolio.equity(prices)
        pnl = eq - self.portfolio.starting_cash
        pct = pnl / self.portfolio.starting_cash if self.portfolio.starting_cash else 0.0

        lines = [
            f"Portfolio as of {date}",
            "=" * 62,
            f"{'Cash':<12}{self.portfolio.cash:>14,.2f}",
            f"{'Equity':<12}{eq:>14,.2f}",
            f"{'P&L':<12}{pnl:>+14,.2f}  ({pct:+.2%})",
        ]
        if self.portfolio.positions:
            lines += ["", f"{'Ticker':<10}{'Shares':>10}{'Cost':>10}{'Price':>10}"
                          f"{'Value':>12}{'Unreal':>12}{'Wgt':>7}"]
            for tkr, pos in sorted(self.portfolio.positions.items()):
                px = prices.get(tkr, pos.avg_cost)
                lines.append(
                    f"{tkr:<10}{pos.shares:>10.3f}{pos.avg_cost:>10.2f}{px:>10.2f}"
                    f"{pos.market_value(px):>12,.2f}{pos.unrealized(px):>+12,.2f}"
                    f"{(pos.market_value(px)/eq if eq else 0):>7.1%}"
                )
        else:
            lines += ["", "(no open positions)"]
        return "\n".join(lines)


def _print_results(results: list[TickerResult]) -> None:
    print(f"\n{'Ticker':<10}{'Rating':<14}{'Price':>10}  Action")
    print("-" * 62)
    for r in results:
        if r.error:
            print(f"{r.ticker:<10}{'ERROR':<14}{'—':>10}  {r.error[:30]}")
        elif r.fill:
            print(f"{r.ticker:<10}{r.rating:<14}{r.price:>10.2f}  "
                  f"{r.fill.action} {r.fill.shares:.3f}")
        else:
            print(f"{r.ticker:<10}{r.rating or '—':<14}"
                  f"{r.price if r.price else 0:>10.2f}  no trade")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Autonomous paper-trading agent.")
    p.add_argument("--watchlist", default="SPY", help="Comma-separated tickers")
    p.add_argument("--date", default=str(_date.today()), help="Analysis date YYYY-MM-DD")
    p.add_argument("--analysts", default="market,social,news,fundamentals")
    p.add_argument("--portfolio", default=None, help="Path to portfolio JSON")
    p.add_argument("--starting-cash", type=float, default=100_000.0)
    p.add_argument("--dry-run", action="store_true", help="Analyse but do not trade")
    p.add_argument("--status", action="store_true", help="Show the book and exit")
    p.add_argument(
        "--rate", action="append", default=None, metavar="TICKER=RATING",
        help="Apply an externally-produced rating (skips the LLM graph entirely). "
             "Repeatable, e.g. --rate QQQM=Buy --rate SPY=Hold",
    )
    p.add_argument("--note", default="", help="Rationale recorded with --rate fills")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    trader = AutoTrader(
        watchlist=args.watchlist.split(","),
        analysts=tuple(a.strip() for a in args.analysts.split(",") if a.strip()),
        portfolio_path=args.portfolio,
        starting_cash=args.starting_cash,
    )

    if args.status:
        print(trader.summary(args.date))
        return 0

    if args.rate:
        results = []
        for spec in args.rate:
            if "=" not in spec:
                p.error(f"--rate expects TICKER=RATING, got {spec!r}")
            tkr, _, rating = spec.partition("=")
            rating = rating.strip().title()
            if rating not in RATINGS:
                p.error(f"unknown rating {rating!r}; expected one of "
                        f"{', '.join(RATINGS)}")
            results.append(
                trader.apply_rating(tkr.strip().upper(), rating, args.date, args.note)
            )
        _print_results(results)
        print()
        print(trader.summary(args.date))
        return 0

    print(f"Watchlist: {', '.join(trader.watchlist)}   date: {args.date}"
          f"{'   [DRY RUN]' if args.dry_run else ''}\n")

    results = trader.run_once(args.date, dry_run=args.dry_run)
    _print_results(results)
    print()
    print(trader.summary(args.date))
    return 0


if __name__ == "__main__":
    sys.exit(main())
