"""Always-on news monitor: poll keyless RSS, surface only what is new.

A monitor that re-reads the same fifty headlines every cycle is a monitor that
tells you nothing. The unit of value here is the *delta* — a headline the desk
has not already reasoned about — so every item is fingerprinted and the seen
set persists across restarts. Without that persistence, restarting the loop
would replay the whole day's news as if it had just broken.

Two keyless sources, deliberately overlapping:

* **Yahoo Finance ticker RSS** — fast, first-party, but shallow (~20 items).
* **Google News RSS** — broad aggregation, catches regional and trade-press
  coverage Yahoo misses, at the cost of more noise and near-duplicates.

Google is queried by **company name**, not ticker. Tickers collide with
ordinary words and acronyms, and the collisions are not harmless: a search for
"AMD stock" returns clinical-trial coverage of age-related macular degeneration,
which scores 9 on the materiality table and would put a trade decision in front
of the panel for entirely the wrong company. Searching for "Advanced Micro
Devices" instead fixes it at the source, which is better than filtering the
noise back out downstream.

Materiality scoring is keyword-based and unashamedly crude. Its job is not to
judge news; it is to decide *which headlines are worth an LLM call*, and for
triage a transparent keyword table beats a model you cannot debug at 06:00.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"

YAHOO_TICKER = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
GOOGLE_QUERY = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

# Market-wide feeds, polled once per cycle rather than once per ticker.
MACRO_QUERIES = (
    "stock market today",
    "Federal Reserve interest rates",
    "inflation CPI report",
)

# --- materiality -----------------------------------------------------------
# Weights are "how much should this headline change a position", not "how
# exciting is it". Regulatory and deal news reprices a stock in one print;
# an analyst note usually does not.
_MATERIAL = (
    (12, r"\bhalted?\b|trading halt|\bbankrupt|chapter 11|\bfraud\b|\bdelist"),
    (10, r"\bacquir(e|es|ed|ing)\b|\bmerger\b|\bbuyout\b|takeover|\bto acquire\b|tender offer"),
    (9,  r"\bFDA\b|approval|phase (?:1|2|3|i{1,3})\b|clinical trial|breakthrough therapy"),
    (9,  r"\bguidance\b|\bcuts? outlook|raises? outlook|profit warning|preannounce"),
    (8,  r"\bearnings\b|\bQ[1-4]\b|quarterly results|beats?\b|misses?\b|\bEPS\b"),
    (7,  r"\bSEC\b|investigation|subpoena|lawsuit|antitrust|probe\b|settlement"),
    (7,  r"activist investor|proxy (fight|battle|contest)|hostile bid|"
         r"board seats?|strategic (review|alternatives)"),
    (7,  r"\bCEO\b|\bCFO\b|steps down|resigns?|appoints?|\bousted\b"),
    (6,  r"\bsplit\b|\bdividend\b|buyback|share repurchase|offering\b|dilut"),
    (5,  r"\bupgrade[sd]?\b|\bdowngrade[sd]?\b|price target|initiat(?:es|ed) coverage"),
    (5,  r"\bcontract\b|\bpartnership\b|\bdeal\b|wins?\b|awarded"),
    (4,  r"\btariff|sanction|export control|\bban\b|regulat"),
)
_MATERIAL_RE = [(w, re.compile(p, re.IGNORECASE)) for w, p in _MATERIAL]

# Routine institutional-flow filings. Aggregators publish these by the thousand
# and they carry no information about the company — but "Acquires Shares of X"
# matches the M&A pattern above and scores a 10, which is enough to preempt a
# genuine stop-loss for the cycle's LLM budget. Matching headlines are capped
# below the trigger threshold rather than dropped, so they still show up in a
# news listing.
_NOISE_CAP = 2
_NOISE = re.compile(
    r"(acquires?|buys?|sells?|purchases?|takes?)\s+(new\s+)?(a\s+)?"
    r"[\d,]*\s*(shares?|stake|position)|"
    r"(shares?|stake|position|holdings?)\s+(sold|bought|acquired|purchased)\s+by|"
    r"(position|stake|holdings?|shares?)\s+(in|of)\s+.*\s+(raised|lowered|boosted|"
    r"trimmed|increased|decreased|cut|reduced|grows?|shortened)|"
    r"(raises?|lowers?|boosts?|trims?|increases?|reduces?|grows?)\s+(stock\s+)?"
    r"(holdings?|position|stake)\s+in|"
    # "Bridgewater Dramatically Trimmed Its Position in Micron" — the verb and
    # the noun swap places often enough to need both orderings.
    r"(trimm?ed|boosted|raised|lowered|increased|decreased|cut|reduced|sold|"
    r"bought|acquired|grew|added to|exited)\s+(its|their|his|her|the)?\s*"
    r"[\w\s]{0,20}?(stake|position|holdings?|shares?)\s+in|"
    r"short interest (update|down|up)|\b13[fdg]\b|"
    r"has \$[\d.]+ (million|billion) (stock )?(holdings|position|stake)|"
    r"(sells?|buys?) \$[\d,.]+ in stock|"
    # Aggregators phrase the same 13F filing a dozen ways. These four cover the
    # forms seen in one live sweep that the clauses above all missed:
    #   "Ninepoint Partners LP Makes New Investment in Guardant Health"
    #   "Silvant Capital Management LLC Takes $12.32 Million Position in Natera"
    #   "Ninepoint Partners LP Invests $2.26 Million in JFrog Ltd."
    #   "21,207 Shares in PayPal Holdings Bought by Edmond DE Rothschild"
    r"makes?\s+(a\s+)?new\s+(investment|position|stake|purchase)|"
    r"takes?\s+(a\s+)?\$?[\d,.]*\s*(million|billion)?\s*(new\s+)?"
    r"(position|stake|holding)|"
    r"invests?\s+\$[\d,.]+\s*(million|billion)?\s+in|"
    r"[\d,]+\s+shares?\s+(in|of)\s+.+\s+(bought|sold|acquired|purchased)\s+by",
    re.IGNORECASE)

# Directional lean. Not a sentiment model — a sign hint that keeps a "misses
# estimates" headline from reading the same as "beats estimates" during triage.
_BULLISH = re.compile(
    r"\bbeat|surge|soar|jump|rally|record high|upgrade|raises?|approval|wins?\b|"
    r"strong|outperform|top(?:s|ped)? estimates|breakthrough", re.IGNORECASE)
_BEARISH = re.compile(
    r"\bmiss|plunge|tumble|sink|slump|fall|drop|downgrade|cuts?\b|warning|halt|"
    r"probe|lawsuit|fraud|weak|underperform|recall|layoff", re.IGNORECASE)


# Legal-form suffixes stripped before a name is used as a search phrase. They
# add no precision and cost recall: few headlines write "Inc." at all.
#
# Deliberately limited to legal forms. Words like "Holdings", "Group",
# "Company" and "Trust" look like suffixes but are usually part of the name a
# journalist actually writes, and stripping them turns a precise phrase into a
# vague one — the exact failure this function exists to prevent.
_SUFFIX_RE = re.compile(
    r"[,\s]+(inc|incorporated|corp|corporation|ltd|limited|plc|"
    r"s\.?a\.?|n\.?v\.?|ag|se|llc|l\.?p\.?|"
    r"class\s+[a-c]|common stock|ordinary shares?)\.?$",
    re.IGNORECASE)


def _name_cache_path() -> Path:
    return (Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
            / "company_names.json")


def _clean_name(name: str) -> str:
    """'Advanced Micro Devices, Inc.' -> 'Advanced Micro Devices'."""
    prev = None
    out = (name or "").strip()
    while out and out != prev:          # 'X Holdings, Inc.' needs two passes
        prev = out
        out = _SUFFIX_RE.sub("", out).strip(" ,.")
    return out


def company_name(ticker: str, cache: dict | None = None) -> str:
    """Resolve a ticker to its company name, cached permanently on disk.

    One yfinance lookup per symbol, ever. A failure returns "" and the caller
    falls back to the ticker — a slightly noisier search beats no search.
    """
    path = _name_cache_path()
    own = cache is None
    if own:
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if ticker in cache:
        return cache[ticker]

    name = ""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        name = _clean_name(info.get("longName") or info.get("shortName") or "")
    except Exception:
        name = ""
    cache[ticker] = name
    if own:
        with suppress(Exception):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return name


@dataclass
class NewsItem:
    ticker: str            # "" for macro
    title: str
    link: str
    source: str
    published: str         # ISO8601 UTC, best effort
    materiality: int = 0
    lean: str = "neutral"  # bullish | bearish | neutral
    fingerprint: str = ""

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        try:
            pub = datetime.fromisoformat(self.published)
        except (ValueError, TypeError):
            return 0.0
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return max(0.0, (now - pub).total_seconds() / 3600)


def _fingerprint(title: str, ticker: str) -> str:
    """Identity of a *story*, not of a URL.

    Google News and Yahoo syndicate the same wire story under different links
    and slightly different trailing publisher suffixes, so hashing the link
    would let one event through several times. Normalising to lowercase
    alphanumerics of the headline collapses those into one item.
    """
    norm = re.sub(r"\s*-\s*[A-Za-z0-9 .&']+$", "", title)   # strip " - Publisher"
    norm = re.sub(r"[^a-z0-9]+", "", norm.lower())
    return hashlib.sha1(f"{ticker}:{norm[:120]}".encode()).hexdigest()[:16]


def score(title: str) -> tuple[int, str]:
    """(materiality 0-12, directional lean)."""
    mat = max((w for w, rx in _MATERIAL_RE if rx.search(title)), default=0)
    if _NOISE.search(title):
        mat = min(mat, _NOISE_CAP)
    bull, bear = bool(_BULLISH.search(title)), bool(_BEARISH.search(title))
    lean = "bullish" if bull and not bear else "bearish" if bear and not bull else "neutral"
    return mat, lean


def _parse_date(raw: str) -> str:
    """RFC-822 or ISO timestamp → ISO8601 UTC. Unparseable → now."""
    raw = (raw or "").strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def fetch_rss(url: str, timeout: int = 20) -> list[dict]:
    """Fetch and parse an RSS feed. Never raises — a dead feed yields []."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        root = ET.fromstring(raw)
    except Exception:
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "link": (item.findtext("link") or "").strip(),
            "published": _parse_date(item.findtext("pubDate") or ""),
            "source": (item.findtext("source") or "").strip(),
        })
    return out


class NewsMonitor:
    """Stateful poller. ``poll`` returns only items not seen before.

    The seen set is capped and pruned by age so a long-running loop does not
    grow without bound; anything older than ``retain_hours`` can safely be
    forgotten because it will also have aged out of the source feeds.
    """

    def __init__(self, state_path: Path | None = None, retain_hours: int = 72):
        self.state_path = Path(state_path) if state_path else (
            Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
            / "news_seen.json"
        )
        self.retain_hours = retain_hours
        self.seen: dict[str, str] = self._load()
        # Held in memory for the life of the monitor so a long run does not
        # re-read the cache file once per ticker per cycle.
        try:
            self._names: dict[str, str] = json.loads(
                _name_cache_path().read_text(encoding="utf-8"))
        except Exception:
            self._names = {}

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retain_hours)
        self.seen = {
            k: v for k, v in self.seen.items()
            if v >= cutoff.isoformat()
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.seen), encoding="utf-8")
        os.replace(tmp, self.state_path)
        with suppress(Exception):
            _name_cache_path().write_text(json.dumps(self._names, indent=2),
                                          encoding="utf-8")

    # --- polling ------------------------------------------------------------

    def _collect(self, ticker: str, urls: list[tuple[str, str]]) -> list[NewsItem]:
        items: list[NewsItem] = []
        for source, url in urls:
            for raw in fetch_rss(url):
                fp = _fingerprint(raw["title"], ticker)
                if fp in self.seen:
                    continue
                # Mark inside the loop so two feeds carrying the same wire
                # story in one cycle still yield only one item.
                self.seen[fp] = datetime.now(timezone.utc).isoformat()
                mat, lean = score(raw["title"])
                items.append(NewsItem(
                    ticker=ticker, title=raw["title"], link=raw["link"],
                    source=raw["source"] or source, published=raw["published"],
                    materiality=mat, lean=lean, fingerprint=fp,
                ))
        return items

    def poll_ticker(self, ticker: str) -> list[NewsItem]:
        # Yahoo's feed is ticker-scoped and first-party, so it needs no help.
        # Google is a text search, so it gets the company name where we know
        # one — quoted, to keep the phrase together.
        name = company_name(ticker, self._names)
        query = f'"{name}" stock' if name else f"{ticker} stock"
        return self._collect(ticker, [
            ("yahoo", YAHOO_TICKER.format(t=urllib.parse.quote(ticker))),
            ("google", GOOGLE_QUERY.format(q=urllib.parse.quote(query))),
        ])

    def poll_macro(self) -> list[NewsItem]:
        urls = [("google", GOOGLE_QUERY.format(q=urllib.parse.quote(q)))
                for q in MACRO_QUERIES]
        return self._collect("", urls)

    def poll(self, tickers: list[str], macro: bool = True,
             pause: float = 0.4) -> list[NewsItem]:
        """Poll every ticker plus the macro feeds; returns new items only.

        ``pause`` throttles between tickers. Google News will start returning
        empty bodies to a client that hammers it, and an empty body is
        indistinguishable from "no news" — the failure mode is silence, so
        the throttle is not optional.
        """
        out: list[NewsItem] = []
        if macro:
            out += self.poll_macro()
        for tkr in tickers:
            out += self.poll_ticker(tkr)
            time.sleep(pause)
        self._save()
        out.sort(key=lambda i: (-i.materiality, i.published), reverse=False)
        return sorted(out, key=lambda i: -i.materiality)

    def prime(self, tickers: list[str]) -> int:
        """Mark everything currently in the feeds as seen, without acting.

        Called on first start so the desk does not wake up, discover three days
        of backlog, and treat all of it as breaking.
        """
        n = len(self.poll(tickers))
        return n


def format_items(items: list[NewsItem], limit: int = 25) -> str:
    if not items:
        return "(no new headlines)"
    lines = []
    for i in items[:limit]:
        tag = i.ticker or "MACRO"
        arrow = {"bullish": "+", "bearish": "-", "neutral": " "}[i.lean]
        lines.append(f"[{i.materiality:>2}]{arrow} {tag:<6} {i.title[:110]}")
    if len(items) > limit:
        lines.append(f"... and {len(items) - limit} more")
    return "\n".join(lines)
