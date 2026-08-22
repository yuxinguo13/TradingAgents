"""Assemble every free data source for a ticker into one document.

The expensive part of the original pipeline was never the data — yfinance,
StockTwits, Reddit and Polymarket are all keyless. It was the ~20 LLM calls
that read that data and wrote prose about it.

This module does the first half only: gather, verify, and lay out the evidence,
with zero LLM calls and zero cost. The reasoning half is then done by whatever
analyst you point at the result — including Claude reading it directly.

Every section is independently fault-tolerant: a source that fails is reported
as unavailable rather than aborting the brief, because a partial evidence set
is still worth reasoning over and a silent omission is not.

    python -m tradingagents.trading.brief QQQM --date 2026-08-21
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as _date, datetime, timedelta

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages

from .execution import latest_close
from .portfolio import Portfolio
from .technicals import technical_structure


def _section(title: str, fn, *args, **kwargs) -> str:
    """Run one source, never letting its failure escape."""
    try:
        body = fn(*args, **kwargs)
        body = (body or "").strip() or "_(empty)_"
    except Exception as exc:
        body = f"_UNAVAILABLE: {type(exc).__name__}: {exc}_"
    return f"\n\n## {title}\n\n{body}"


def portfolio_context(ticker: str, date: str, portfolio_path=None) -> str:
    """Where this ticker already sits in the book.

    Included because a rating is a *change* decision: 'Buy' means something
    different when you already hold 25% than when you hold nothing.
    """
    pf = Portfolio.load(portfolio_path)
    try:
        px = latest_close(ticker, date)
    except Exception:
        px = None

    marks = {}
    for tkr, pos in pf.positions.items():
        try:
            marks[tkr] = latest_close(tkr, date)
        except Exception:
            marks[tkr] = pos.avg_cost
    if px is not None:
        marks[ticker] = px

    eq = pf.equity(marks)
    pos = pf.position(ticker)
    lines = [
        f"- Total equity: **${eq:,.2f}**  (started ${pf.starting_cash:,.2f}, "
        f"P&L {eq - pf.starting_cash:+,.2f})",
        f"- Cash available: **${pf.cash:,.2f}**",
        f"- Current {ticker} position: **{pos.shares:.4f} shares** @ avg cost "
        f"${pos.avg_cost:,.2f} → weight **{pf.weight(ticker, marks):.2%}**",
    ]
    if px is not None:
        lines.append(f"- Latest verified close: **${px:,.2f}** "
                     f"(unrealized {pos.unrealized(px):+,.2f})")
    if pf.positions:
        others = ", ".join(
            f"{t} {p.shares:.3f}@{p.avg_cost:,.2f}" for t, p in sorted(pf.positions.items())
        )
        lines.append(f"- Full book: {others}")
    lines.append(
        "\nTarget weights by rating: Buy 25% · Overweight 15% · "
        "Hold = leave unchanged · Underweight 5% · Sell 0%."
    )
    return "\n".join(lines)


def build_brief(
    ticker: str,
    date: str,
    lookback_days: int = 30,
    news_days: int = 14,
    portfolio_path=None,
) -> str:
    start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=news_days)).strftime("%Y-%m-%d")

    out = [f"# Evidence brief — {ticker} @ {date}",
           "\n_All sources below are keyless and free. No LLM calls were made "
           "to produce this document._"]

    out.append(_section("Portfolio context", portfolio_context, ticker, date, portfolio_path))
    out.append(_section("Verified market snapshot (ground truth OHLCV + indicators)",
                        build_verified_market_snapshot, ticker, date, lookback_days))
    # Full-history structure: multi-horizon returns vs the index, MA stack and
    # slopes, swing structure, volume accumulation/distribution, volatility
    # regime, key levels, and two ASCII charts. The snapshot above says where
    # price is; this says how it got there and on what volume.
    out.append(_section("Technical structure (full history, volume, charts)",
                        technical_structure, ticker, date))
    out.append(_section(f"Ticker news ({start} → {date})",
                        route_to_vendor, "get_news", ticker, start, date))
    out.append(_section("Macro / global news",
                        route_to_vendor, "get_global_news", date, 7, 10))
    out.append(_section("StockTwits sentiment",
                        fetch_stocktwits_messages, ticker, 30))
    out.append(_section("Reddit sentiment", fetch_reddit_posts, ticker))
    out.append(_section("Fundamentals",
                        route_to_vendor, "get_fundamentals", ticker, date))
    out.append(_section("Insider transactions",
                        route_to_vendor, "get_insider_transactions", ticker))

    return "".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build a keyless evidence brief for a ticker.")
    p.add_argument("ticker")
    p.add_argument("--date", default=str(_date.today()))
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--news-days", type=int, default=14)
    p.add_argument("--portfolio", default=None)
    p.add_argument("-o", "--out", default=None, help="Write to file instead of stdout")
    a = p.parse_args(argv)

    text = build_brief(a.ticker.upper(), a.date, a.lookback, a.news_days, a.portfolio)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(text):,} chars -> {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
