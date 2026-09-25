"""Insider transactions: who inside the company bought or sold, and when.

Ported in spirit from upstream's fundamentals analyst, which was given an
``get_insider_transactions`` tool. The reading here is deliberately narrow,
because the literature is: open-market **purchases** carry information (an
insider has one reason to buy with their own money), **sales** mostly do not
(diversification, taxes, 10b5-1 plans). So a cluster of buyers is reported
as a supporting fact; selling is counted and printed but never used as a
caution, because at a large company several insiders sell every quarter on
10b5-1 plans. Grants, gifts, option exercises and tax withholding are
ignored: they say nothing about what the insider expects.

Yahoo serves recent filings only, so an empty frame is "none reported", not
"no insiders". Rows are dated by the transaction, not the filing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_TTL_HOURS = 20.0
WINDOW_DAYS = 90            # transactions this far back count
CLUSTER_BUYERS = 2          # distinct open-market buyers that make a cluster
CLUSTER_SELLERS = 3         # distinct open-market sellers that make a caution


def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


def cache_path() -> Path:
    return _home() / "insiders.json"


@dataclass
class InsiderRead:
    symbol: str
    as_of: str = ""
    window_days: int = WINDOW_DAYS
    buys: int = 0                       # open-market purchases (rows)
    sells: int = 0                      # open-market sales (rows)
    buy_value: float = 0.0
    sell_value: float = 0.0
    buyers: list = field(default_factory=list)     # distinct insiders who bought
    sellers: list = field(default_factory=list)    # distinct insiders who sold
    last_buy: str = ""
    last_sell: str = ""
    fetched_at: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def cluster_buying(self) -> bool:
        return len(self.buyers) >= CLUSTER_BUYERS

    @property
    def cluster_selling(self) -> bool:
        """Broad one-sided selling. Recorded, not acted on: routine at large caps."""
        return len(self.sellers) >= CLUSTER_SELLERS and self.sell_value > 5 * max(self.buy_value, 1.0)

    def read(self) -> str:
        """One sentence, or an empty string when there is nothing to say."""
        if not self.ok:
            return ""
        if self.cluster_buying:
            who = "、".join(self.buyers[:3])
            return (f"内部人买入：近 {self.window_days} 天 {len(self.buyers)} 位内部人公开市场买入"
                    f"（{who}，合计 ${self.buy_value / 1e6:.1f}M，最近 {self.last_buy}）")
        if self.cluster_selling:
            return (f"内部人集中卖出：近 {self.window_days} 天 {len(self.sellers)} 位内部人卖出"
                    f"合计 ${self.sell_value / 1e6:.1f}M，无人买入")
        if self.buys:
            return (f"内部人买入 {self.buys} 笔（{'、'.join(self.buyers[:2])}，"
                    f"${self.buy_value / 1e6:.1f}M，最近 {self.last_buy}）")
        return ""


def classify(transaction: str) -> str:
    """'buy' | 'sell' | '' for the free text Yahoo puts in the Transaction column."""
    t = (transaction or "").strip().lower()
    if not t:
        return ""
    if t.startswith(("purchase", "buy", "open market purchase")):
        return "buy"
    if t.startswith("sale") and "option" not in t:
        return "sell"
    return ""       # grants, gifts, conversions, option exercises, tax withholding


def summarize(frame, symbol: str, as_of: date | None = None,
              window_days: int = WINDOW_DAYS) -> InsiderRead:
    """Reduce Yahoo's transaction frame to the counts the desk reads."""
    out = InsiderRead(symbol=symbol.upper(), window_days=window_days, fetched_at=time.time())
    as_of = as_of or date.today()
    out.as_of = as_of.isoformat()
    if frame is None or getattr(frame, "empty", True):
        return out
    start = as_of - timedelta(days=window_days)
    buyers: dict[str, None] = {}
    sellers: dict[str, None] = {}
    for _, row in frame.iterrows():
        # Yahoo puts the sentence ("Sale at price ...") in Text; Transaction is
        # usually blank, and is the fallback.
        kind = classify(str(row.get("Text") or row.get("Transaction") or ""))
        if not kind:
            continue
        try:
            when = row.get("Start Date")
            when = when.date() if hasattr(when, "date") else date.fromisoformat(str(when)[:10])
        except (ValueError, TypeError, AttributeError):
            continue
        if when < start or when > as_of:
            continue
        who = str(row.get("Insider", "") or "").strip().title()
        value = row.get("Value", 0)
        try:
            value = float(value) if value == value else 0.0
        except (TypeError, ValueError):
            value = 0.0
        if kind == "buy":
            out.buys += 1
            out.buy_value += value
            buyers.setdefault(who or f"#{out.buys}", None)
            out.last_buy = max(out.last_buy, when.isoformat())
        else:
            out.sells += 1
            out.sell_value += value
            sellers.setdefault(who or f"#{out.sells}", None)
            out.last_sell = max(out.last_sell, when.isoformat())
    out.buyers, out.sellers = list(buyers), list(sellers)
    return out


def fetch(symbol: str, *, as_of: date | None = None, log=logger.debug) -> InsiderRead:
    """One symbol from Yahoo. Never raises; the error rides in the result."""
    try:
        import yfinance as yf
        frame = yf.Ticker(symbol.upper()).insider_transactions
        return summarize(frame, symbol, as_of)
    except Exception as exc:
        log(f"insiders: {symbol} unavailable ({type(exc).__name__}: {exc})")
        return InsiderRead(symbol=symbol.upper(), fetched_at=time.time(),
                           error=f"{type(exc).__name__}: {exc}")


class InsidersBook:
    """Disk-cached insider reads, the same idiom as the fundamentals book."""

    def __init__(self, path: Path | None = None, *, ttl_hours: float = CACHE_TTL_HOURS,
                 fetcher=fetch):
        self.path = path or cache_path()
        self.ttl = ttl_hours * 3600.0
        self._fetch = fetcher
        self._cache: dict[str, InsiderRead] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("insiders cache unreadable (%s); starting empty", exc)
            return
        if not isinstance(raw, dict):
            return
        for sym, entry in raw.items():
            try:
                self._cache[str(sym).upper()] = InsiderRead(**entry)
            except Exception:
                continue

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in self._cache.items()},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as exc:
            logger.warning("could not write the insiders cache: %s", exc)

    def fresh(self, symbol: str) -> bool:
        got = self._cache.get(symbol.upper())
        return bool(got) and got.ok and (time.time() - got.fetched_at) < self.ttl

    def get(self, symbols: list[str], *, refresh: bool = False, as_of: date | None = None,
            log=logger.debug) -> dict[str, InsiderRead]:
        wanted = [s.upper() for s in symbols if s]
        stale = [s for s in wanted if refresh or not self.fresh(s)]
        for sym in stale:
            log(f"insiders: fetching {sym}")
            self._cache[sym] = self._fetch(sym, as_of=as_of, log=log)
        if stale:
            self._save()
        return {s: self._cache[s] for s in wanted if s in self._cache}
