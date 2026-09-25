"""Which names the report looks at: the leaders, sector by sector.

Two sources, merged and deduplicated:

- **Bellwethers** — a fixed table of the largest names in every sector. These
  are on the page every day whether or not they rank, because a market report
  that does not mention Apple, JPMorgan or Exxon on the day they move is not a
  market report.
- **Screen leaders** — the top of the momentum screen over every exchange,
  capped per sector, so the names the market is currently rewarding show up
  next to the names it always watches.

No account is consulted. Whatever the reader holds is deliberately unknown
here; the report describes the market, and task 3 describes the portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# yfinance sector names, so they line up with the fundamentals cache and the
# policy monitor's sector_impact keys.
BELLWETHERS: dict[str, tuple[str, ...]] = {
    "Technology": ("AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM"),
    "Communication Services": ("GOOGL", "META", "NFLX", "DIS"),
    "Consumer Cyclical": ("AMZN", "TSLA", "HD", "NKE", "MCD"),
    "Consumer Defensive": ("WMT", "COST", "PG", "KO", "PEP"),
    "Healthcare": ("LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"),
    "Financial Services": ("JPM", "BRK-B", "V", "MA", "GS", "BAC"),
    "Industrials": ("CAT", "GE", "BA", "UNP", "HON", "LMT"),
    "Energy": ("XOM", "CVX", "COP", "SLB"),
    "Basic Materials": ("LIN", "FCX", "NEM"),
    "Utilities": ("NEE", "SO", "DUK"),
    "Real Estate": ("PLD", "AMT", "O"),
}


@dataclass
class Name:
    symbol: str
    sector: str = "Unknown"
    name: str = ""
    source: str = "bellwether"      # bellwether | screen
    screen_rank: int = 0
    score: float = float("nan")


def bellwethers() -> list[Name]:
    return [Name(symbol=s, sector=sector) for sector, syms in BELLWETHERS.items()
            for s in syms]


def sector_of(symbol: str) -> str:
    for sector, syms in BELLWETHERS.items():
        if symbol.upper() in syms:
            return sector
    return "Unknown"


def screen_leaders(when: date, *, exchange: str = "all", top: int = 60,
                   per_sector: int = 4, screen=None, log=logger.debug) -> list[Name]:
    """The screen's top rows as Names, capped per sector. Empty when it fails.

    ``screen(when, exchange, top)`` is the seam: the default runs the real
    universe scan through :func:`live.advisor.run_screen`, which also files
    the CSV, and a test hands in a list of rows instead.
    """
    from tradingagents.live.advisor import candidates_from_frame, run_screen
    try:
        if screen is None:
            frame, _ = run_screen(when.isoformat(), exchange, top, log=log)
        else:
            frame = screen(when, exchange, top)
        cands = candidates_from_frame(frame)
    except Exception as exc:
        logger.warning("the screen did not run: %s", exc)
        return []
    out: list[Name] = []
    per: dict[str, int] = {}
    for c in cands:
        sector = c.sector or "Unknown"
        if per.get(sector, 0) >= per_sector:
            continue
        per[sector] = per.get(sector, 0) + 1
        out.append(Name(symbol=c.symbol, sector=sector, name=c.name, source="screen",
                        screen_rank=c.rank, score=c.score))
    return out


def saved_screen(when: date) -> list[Name]:
    """Yesterday's screen from disk, when a fresh scan is unwanted or failed."""
    from tradingagents.live.advisor import candidates_from_csv, screens_dirs
    for d in screens_dirs():
        for exchange in ("all", "nasdaq"):
            p = Path(d) / f"screen_{exchange}_{when.isoformat()}.csv"
            if p.exists():
                return [Name(symbol=c.symbol, sector=c.sector or "Unknown", name=c.name,
                             source="screen", screen_rank=c.rank, score=c.score)
                        for c in candidates_from_csv(p)]
    return []


def prominent(when: date, *, exchange: str = "all", top: int = 60, per_sector: int = 4,
              screen=None, use_cache: bool = False, log=logger.debug) -> list[Name]:
    """Bellwethers plus screen leaders, one entry per symbol, bellwethers first."""
    names = bellwethers()
    seen = {n.symbol for n in names}
    leaders = saved_screen(when) if use_cache else []
    if not leaders:
        leaders = screen_leaders(when, exchange=exchange, top=top, per_sector=per_sector,
                                 screen=screen, log=log)
    for n in leaders:
        if n.symbol not in seen:
            seen.add(n.symbol)
            names.append(n)
        else:
            # A bellwether that also ranks keeps its fixed seat and gains the rank.
            for b in names:
                if b.symbol == n.symbol:
                    b.screen_rank, b.score = n.screen_rank, n.score
                    if b.sector == "Unknown":
                        b.sector = n.sector
    return names
