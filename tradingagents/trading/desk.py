"""The trading desk: one command surface, no LLM anywhere in the loop.

This replaces the API-driven pipeline as the primary way to run the book. The
division of labor is explicit:

* **This code** does everything deterministic — fetch free data, build
  evidence briefs, compute ATR, size positions, enforce risk limits, persist
  the book, track P&L.
* **The analyst** (Claude, a human, anything that can read) reads the briefs
  and produces ratings. Reasoning is the only step that ever needed
  intelligence, and it never needed an API key.

The watchlist carries each ticker's sleeve, so a rating is all the analyst
ever supplies — sizing style follows from configuration, not from prompts.

    python -m tradingagents.trading.desk brief            # briefs for the watchlist
    python -m tradingagents.trading.desk rate QQQM=Buy MU=Sell --note "..."
    python -m tradingagents.trading.desk status
    python -m tradingagents.trading.desk watchlist        # show / edit the list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as _date
from pathlib import Path

from .execution import (
    CORE, RATINGS, SLEEVES, TACTICAL, PaperBroker, atr_pct, latest_close,
)
from .portfolio import Portfolio

# Default two-horizon watchlist. Chosen 2026-08-21 on measured ATR%, dollar
# volume, and thesis diversity; edit with `desk watchlist --add/--remove`
# or by hand in the JSON file.
DEFAULT_WATCHLIST: dict[str, str] = {
    # core: long-term convictions, conviction-sized
    "SPY":  CORE, "QQQM": CORE, "NVDA": CORE, "MSFT": CORE, "LLY": CORE,
    # tactical: short-term, volatility-sized
    "MU": TACTICAL, "AMD": TACTICAL, "CRWV": TACTICAL,
    "META": TACTICAL, "XOM": TACTICAL,
}


def watchlist_path() -> Path:
    env = os.getenv("TRADINGAGENTS_WATCHLIST_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".tradingagents" / "watchlist.json"


def load_watchlist() -> dict[str, str]:
    p = watchlist_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return dict(DEFAULT_WATCHLIST)


def save_watchlist(wl: dict[str, str]) -> None:
    p = watchlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wl, indent=2), encoding="utf-8")


class Desk:
    """Facade over the portfolio, broker, and evidence brief."""

    def __init__(self, portfolio_path=None, starting_cash: float = 100_000.0):
        self.portfolio_path = portfolio_path
        self.portfolio = Portfolio.load(portfolio_path, starting_cash=starting_cash)
        self.broker = PaperBroker()
        self.watchlist = load_watchlist()

    # --- pricing -----------------------------------------------------------

    def marks(self, date: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for tkr in set(self.watchlist) | set(self.portfolio.positions):
            try:
                out[tkr] = latest_close(tkr, date)
            except Exception:
                pass
        return out

    # --- the two verbs -----------------------------------------------------

    def brief(self, date: str, tickers: list[str] | None = None,
              out_dir: str | None = None) -> list[Path]:
        """Write one evidence brief per ticker; returns the paths."""
        from .brief import build_brief  # local import: keeps `rate` LLM-free AND yfinance-light
        targets = [t.upper() for t in tickers] if tickers else list(self.watchlist)
        out = Path(out_dir) if out_dir else Path.home() / ".tradingagents" / "briefs"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for tkr in targets:
            text = build_brief(tkr, date, portfolio_path=self.portfolio_path)
            sleeve = self.watchlist.get(tkr, CORE)
            try:
                vol = atr_pct(tkr, date)
                risk = f"\n- 14d ATR: **{vol:.2%}** of price"
            except Exception:
                risk = ""
            text = (f"{text}\n\n## Sleeve\n\n- This ticker is rated on the "
                    f"**{sleeve}** horizon{risk}\n")
            p = out / f"{tkr}_{date}.md"
            p.write_text(text, encoding="utf-8")
            paths.append(p)
        return paths

    def rate(self, ratings: dict[str, str], date: str, note: str = "") -> list[str]:
        """Apply analyst ratings; returns human-readable result lines."""
        lines = []
        marks = self.marks(date)
        for tkr, rating in ratings.items():
            tkr = tkr.upper()
            rating = rating.strip().title()
            if rating not in RATINGS:
                lines.append(f"{tkr:<7} REJECTED: unknown rating {rating!r}")
                continue
            sleeve = self.watchlist.get(tkr) or getattr(
                self.portfolio.position(tkr), "sleeve", CORE)
            try:
                price = latest_close(tkr, date)
                marks[tkr] = price
                atr = atr_pct(tkr, date) if sleeve == TACTICAL else None
                fill = self.broker.execute(
                    self.portfolio, tkr, rating, price, date,
                    prices=marks, sleeve=sleeve, atr=atr, note=note,
                )
                self.portfolio.save(self.portfolio_path)
                if fill:
                    lines.append(f"{tkr:<7} {rating:<12} {fill.action} "
                                 f"{fill.shares:.3f} @ {price:,.2f}  [{sleeve}]")
                else:
                    lines.append(f"{tkr:<7} {rating:<12} no trade  [{sleeve}]")
            except Exception as exc:
                lines.append(f"{tkr:<7} ERROR: {exc}")
        return lines

    # --- reporting ---------------------------------------------------------

    def status(self, date: str) -> str:
        marks = self.marks(date)
        pf = self.portfolio
        eq = pf.equity(marks)
        pnl = eq - pf.starting_cash
        lines = [
            f"Book as of {date}",
            "=" * 74,
            f"Cash {pf.cash:>14,.2f}   Equity {eq:>14,.2f}   "
            f"P&L {pnl:>+12,.2f} ({(pnl / pf.starting_cash if pf.starting_cash else 0):+.2%})",
        ]
        for sleeve in SLEEVES:
            held = {t: p for t, p in sorted(pf.positions.items())
                    if getattr(p, "sleeve", CORE) == sleeve}
            lines.append(f"\n[{sleeve}]")
            if not held:
                lines.append("  (empty)")
                continue
            lines.append(f"  {'Ticker':<8}{'Shares':>10}{'Cost':>10}{'Price':>10}"
                         f"{'Value':>12}{'Unreal':>12}{'Wgt':>7}")
            for tkr, pos in held.items():
                px = marks.get(tkr, pos.avg_cost)
                lines.append(
                    f"  {tkr:<8}{pos.shares:>10.3f}{pos.avg_cost:>10.2f}{px:>10.2f}"
                    f"{pos.market_value(px):>12,.2f}{pos.unrealized(px):>+12,.2f}"
                    f"{(pos.market_value(px) / eq if eq else 0):>7.1%}"
                )
        watch_only = sorted(set(self.watchlist) - set(pf.positions))
        if watch_only:
            lines.append(f"\nWatching (no position): {', '.join(watch_only)}")
        return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="desk", description=__doc__.split("\n")[0])
    p.add_argument("--date", default=str(_date.today()))
    p.add_argument("--portfolio", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="Write evidence briefs (free, no LLM)")
    b.add_argument("tickers", nargs="*", help="Default: whole watchlist")
    b.add_argument("-o", "--out-dir", default=None)

    r = sub.add_parser("rate", help="Apply ratings: TICKER=RATING ...")
    r.add_argument("pairs", nargs="+", metavar="TICKER=RATING")
    r.add_argument("--note", default="")

    sub.add_parser("status", help="Show the book")

    c = sub.add_parser("chart", help="Full-history technical structure + ASCII charts")
    c.add_argument("tickers", nargs="+")
    c.add_argument("--benchmark", default="SPY")

    s = sub.add_parser("screen", help="Scan the whole exchange, rank by evidence, keep top N")
    s.add_argument("--exchange", choices=["nasdaq", "all"], default="nasdaq")
    s.add_argument("--top", type=int, default=50)
    s.add_argument("--min-price", type=float, default=5.0)
    s.add_argument("--min-dollar-vol", type=float, default=20.0, help="millions, 50d avg")
    s.add_argument("--max-vol", type=float, default=1.20, help="max 20d realized vol (1.2 = 120%%)")
    s.add_argument("--allow-below-200", action="store_true",
                   help="do not require price above the 200-day SMA")
    s.add_argument("--max-per-sector", type=int, default=8,
                   help="cap per sector in the final list; 0 disables the cap")
    s.add_argument("--refresh", action="store_true", help="ignore the cached price panel")

    w = sub.add_parser("watchlist", help="Show or edit the watchlist")
    w.add_argument("--add", action="append", default=[],
                   metavar="TICKER[=SLEEVE]", help="sleeve: core|tactical")
    w.add_argument("--remove", action="append", default=[], metavar="TICKER")

    a = p.parse_args(argv)
    desk = Desk(portfolio_path=a.portfolio)

    if a.cmd == "brief":
        for path in desk.brief(a.date, a.tickers or None, a.out_dir):
            print(path)
    elif a.cmd == "rate":
        ratings = {}
        for pair in a.pairs:
            if "=" not in pair:
                p.error(f"expected TICKER=RATING, got {pair!r}")
            t, _, rt = pair.partition("=")
            ratings[t] = rt
        for line in desk.rate(ratings, a.date, a.note):
            print(line)
        print()
        print(desk.status(a.date))
    elif a.cmd == "status":
        print(desk.status(a.date))
    elif a.cmd == "chart":
        from .technicals import technical_structure
        for tkr in a.tickers:
            print(f"\n# {tkr.upper()} — technical structure @ {a.date}\n")
            print(technical_structure(tkr.upper(), a.date, benchmark=a.benchmark))
    elif a.cmd == "screen":
        from .screener import format_table, save_results, screen
        g, stats = screen(
            a.date, exchange=a.exchange, top=a.top, min_price=a.min_price,
            min_dollar_vol=a.min_dollar_vol * 1e6, max_rvol=a.max_vol,
            require_above_200=not a.allow_below_200,
            max_per_sector=(a.max_per_sector or None), refresh=a.refresh,
            log=lambda m: print(m, file=sys.stderr),
        )
        print(f"\nUniverse {stats['universe']} → priced {stats['priced']} → "
              f"passed filters {stats['passed_filters']} → top {len(g)}"
              + (f"  (sector cap {a.max_per_sector}/sector over a pool of "
                 f"{stats['candidate_pool']})" if "candidate_pool" in stats else "") + "\n")
        if "sector_counts_in_top" in stats:
            sc = stats["sector_counts_in_top"]
            print("Sectors in top list: " + ", ".join(f"{k} {v}" for k, v in
                  sorted(sc.items(), key=lambda kv: -kv[1])) + "\n")
        print(format_table(g))
        if not g.empty:
            print(f"\nsaved: {save_results(g, a.date, a.exchange)}")
    elif a.cmd == "watchlist":
        wl = desk.watchlist
        for spec in a.add:
            t, _, s = spec.partition("=")
            sleeve = (s or CORE).lower()
            if sleeve not in SLEEVES:
                p.error(f"sleeve must be one of {SLEEVES}, got {sleeve!r}")
            wl[t.upper()] = sleeve
        for t in a.remove:
            wl.pop(t.upper(), None)
        if a.add or a.remove:
            save_watchlist(wl)
        for tkr, sleeve in sorted(wl.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"{tkr:<8}{sleeve}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
