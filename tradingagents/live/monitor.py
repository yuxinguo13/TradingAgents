"""The always-on loop: watch, detect, decide, execute, repeat.

This is the piece that makes the desk an agent rather than a command. It runs
indefinitely, and what it does on each pass is a function of the market clock:

* **Regular session** — full cycle. Refresh the account from Investopedia,
  snapshot every symbol under coverage, poll news, evaluate triggers, and put
  only the triggered names in front of the panel.
* **Pre-market and after-hours** — news only. Nothing is tradeable, but the
  headlines that will move the open arrive now, and holding them until 09:30
  means starting the session already behind.
* **Closed** — sleep to the next open, waking periodically for news, and run
  the nightly universe rescan once per day.

Coverage is *not* a fixed list. Every night the screener ranks the whole
exchange and the top names become the watch set, so what the agent follows is
whatever currently clears a quantitative bar, plus whatever is actually held.
The only permanent members are the open positions, because those carry risk
whether or not they still screen well.

    python -m tradingagents.live.cli run
    python -m tradingagents.live.cli run --dry-run --once
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG

from . import clock
from .brain import Panel, Snapshot, Trigger, build_evidence, snapshot, triggers
from .broker import Account, configured_venue, open_broker
from .newsfeed import NewsItem, NewsMonitor, format_items
from .secretary import RiskLimits, Secretary, TradeLedger, kill_switch_engaged

logger = logging.getLogger("live")


def _home() -> Path:
    p = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class MonitorConfig:
    open_interval: int = 120           # seconds between full cycles while open
    closed_interval: int = 1800        # news-only cadence outside the session
    screen_top: int = 40               # names kept from the nightly screen
    max_coverage: int = 60             # hard cap on symbols per cycle
    max_panels_per_cycle: int = 4      # LLM budget guard: worst triggers first
    dry_run: bool = False              # fill tickets, never submit
    headless: bool = True
    auto_screen: bool = True
    screen_exchange: str = "nasdaq"
    # Paper-only escape hatch. A real venue simply will not fill outside RTH,
    # so this cannot make a live run behave differently — but the local paper
    # book has no such constraint, and requiring the user to wait for Monday
    # to see whether the agent works at all is a poor trade.
    trade_when_closed: bool = False


class LiveDesk:
    """Owns the loop, the coverage set, and the one browser session."""

    def __init__(self, cfg: MonitorConfig | None = None,
                 limits: RiskLimits | None = None, llm=None):
        self.cfg = cfg or MonitorConfig()
        self.secretary = Secretary(limits=limits, ledger=TradeLedger())
        self.news = NewsMonitor()
        self.llm = llm
        self.panel = Panel(llm, self.secretary) if llm is not None else None
        self.coverage: list[str] = []
        self.screen_rank: dict[str, int] = {}
        self.last_screen: date | None = None
        self.state_path = _home() / "live_state.json"
        self._load_state()

    # --- persistence --------------------------------------------------------

    def _load_state(self) -> None:
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.screen_rank = {k: int(v) for k, v in d.get("screen_rank", {}).items()}
            self.coverage = list(d.get("coverage", []))
            ls = d.get("last_screen")
            self.last_screen = date.fromisoformat(ls) if ls else None
        except Exception:
            pass

    def _save_state(self) -> None:
        payload = {
            "screen_rank": self.screen_rank,
            "coverage": self.coverage,
            "last_screen": self.last_screen.isoformat() if self.last_screen else None,
            "updated": datetime.now().isoformat(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    # --- coverage -----------------------------------------------------------

    def refresh_screen(self, data_date: str, force: bool = False) -> None:
        """Rescan the exchange and rebuild the candidate set.

        Runs at most once a day: the factors are built from daily bars, so a
        second run before the next close would return the same ranking at full
        cost.
        """
        today = date.today()
        if not force and self.last_screen == today:
            return
        if not self.cfg.auto_screen:
            return
        logger.info("running universe screen (%s)...", self.cfg.screen_exchange)
        try:
            from tradingagents.trading.screener import screen
            g, stats = screen(
                data_date, exchange=self.cfg.screen_exchange,
                top=self.cfg.screen_top, log=lambda m: logger.debug("%s", m),
            )
            self.screen_rank = {t: int(r["rank"]) for t, r in g.iterrows()}
            self.last_screen = today
            logger.info("screen: %s names → top %d candidates",
                        stats.get("universe", "?"), len(self.screen_rank))
        except Exception as exc:
            logger.error("screen failed (keeping previous ranking): %s", exc)
        self._save_state()

    def build_coverage(self, account: Account) -> list[str]:
        """Held names first, then the best-ranked screen candidates."""
        held = [h.symbol for h in account.holdings]
        cands = [t for t, _ in sorted(self.screen_rank.items(), key=lambda kv: kv[1])]
        seen, out = set(), []
        for t in held + cands:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= self.cfg.max_coverage:
                break
        self.coverage = out
        return out

    # --- one cycle ----------------------------------------------------------

    def cycle(self, broker, state: clock.MarketState) -> dict:
        """One full pass. Returns a summary dict for logging."""
        data_date = clock.last_trading_day(state.now).isoformat()
        summary = {"at": state.now.isoformat(), "session": state.session,
                   "phase": state.phase(), "triggered": [], "orders": [],
                   "news": 0, "errors": []}

        if kill_switch_engaged():
            summary["errors"].append("kill switch engaged — monitoring only")

        try:
            account = broker.account()
        except Exception as exc:
            logger.error("could not read the account: %s", exc)
            summary["errors"].append(f"account read failed: {exc}")
            return summary

        self.refresh_screen(data_date)
        cover = self.build_coverage(account)
        logger.info("account $%s · cash $%s · %d holdings · covering %d symbols",
                    f"{account.account_value:,.0f}", f"{account.cash:,.0f}",
                    len(account.holdings), len(cover))

        # --- news ---
        # A cold start (or a wiped state file) sees the whole RSS backlog as
        # unseen. Priming marks it read without acting, so the desk does not
        # wake up and trade last quarter's earnings as if they had just landed.
        if not self.news.seen:
            n = self.news.prime(cover)
            logger.info("first run: primed %d existing headlines (no action taken)", n)
            summary["news"] = 0
            summary["primed"] = n
            return summary

        items = self.news.poll(cover, macro=True)
        macro = [i for i in items if not i.ticker]
        summary["news"] = len(items)
        if items:
            logger.info("news: %d new headlines\n%s", len(items), format_items(items, 10))

        # --- snapshots + triggers ---
        fired: list[tuple[str, Snapshot, list[Trigger]]] = []
        for sym in cover:
            snap = snapshot(sym, data_date)
            trg = triggers(sym, snap, items, account, self.screen_rank.get(sym))
            if trg:
                fired.append((sym, snap, trg))
                summary["triggered"] += [str(t) for t in trg]

        if fired:
            logger.info("%d symbols triggered:\n  %s", len(fired),
                        "\n  ".join(str(t) for _, _, ts in fired for t in ts))
        else:
            logger.info("no triggers this cycle")
            return summary

        if not state.is_tradeable and not self.cfg.trade_when_closed:
            logger.info("market not open — triggers noted, no orders placed")
            return summary
        if not state.is_tradeable:
            logger.warning("market is CLOSED — trading anyway at the last close "
                           "(--trade-when-closed; paper books only)")
        if kill_switch_engaged():
            return summary
        if self.panel is None:
            logger.warning("no LLM configured — triggers detected but nothing decides")
            return summary

        # Worst first: a stop-loss on a held name must not lose its LLM budget
        # to a screen-entry idea that will still be there next cycle.
        fired.sort(key=lambda f: -max(t.urgency for t in f[2]))

        for sym, snap, trg in fired[:self.cfg.max_panels_per_cycle]:
            try:
                res = self._decide_and_trade(broker, account, sym, snap, trg,
                                             macro, items, state)
                if res:
                    summary["orders"].append(res)
            except Exception as exc:
                logger.exception("decision failed for %s", sym)
                summary["errors"].append(f"{sym}: {exc}")
        return summary

    def _decide_and_trade(self, broker, account: Account, sym: str, snap: Snapshot,
                          trg: list[Trigger], macro: list[NewsItem],
                          items: list[NewsItem], state) -> dict | None:
        evidence = build_evidence(sym, snap, items, account, trg, macro, state.phase())
        result = self.panel.deliberate(sym, evidence, account, snap.price)
        logger.info("%s", result.summary())

        if result.order is None:
            return None

        # Price the order at the venue's own quote where possible: sizing
        # against a yfinance close while filling at Investopedia's price is how
        # a position ends up a different size than the one that was approved.
        price = snap.price
        try:
            q = broker.quote(sym)
            if q > 0:
                price = q
        except Exception:
            pass

        verdict = self.secretary.check(result.order, account, price,
                                       market_open=state.is_tradeable)
        if not verdict.ok:
            logger.info("%s BLOCKED by risk gate: %s", sym, verdict.reason)
            return {"symbol": sym, "blocked": verdict.reason,
                    "wanted": result.order.to_dict()}

        order = verdict.order
        logger.info("%s → %s %d @ ~%.2f (%s) — %s", sym, order.action, order.quantity,
                    price, verdict.reason, order.rationale[:160])

        if self.cfg.dry_run:
            res = broker.place_order(order.symbol, order.action, order.quantity,
                                     dry_run=True)
            logger.info("DRY RUN: %s (%s)", res.message, res.artifact)
            return {"symbol": sym, "dry_run": True, "order": order.to_dict()}

        res = broker.place_order(order.symbol, order.action, order.quantity,
                                 order.order_type, order.limit_price)
        self.secretary.ledger.record(order, price, res.ok, res.message,
                                     venue=getattr(self.secretary, "venue", ""))
        if res.ok:
            logger.info("ORDER PLACED: %s %d %s — %s", order.action, order.quantity,
                        order.symbol, res.message)
        else:
            logger.error("ORDER FAILED: %s %d %s — %s (%s)", order.action,
                         order.quantity, order.symbol, res.message, res.artifact)
        return {"symbol": sym, "placed": res.ok, "message": res.message,
                "order": order.to_dict()}

    # --- the loop -----------------------------------------------------------

    def run(self, once: bool = False, max_cycles: int | None = None) -> None:
        logger.info("live desk starting (dry_run=%s, headless=%s)",
                    self.cfg.dry_run, self.cfg.headless)
        logger.info("kill switch: touch %s to stop trading",
                    _home() / "STOP")
        cycles = 0
        venue = configured_venue()
        # The daily budgets are a fraction of *this* account's value, so they
        # are counted from this venue's rows only. See TradeLedger._for.
        self.secretary.venue = venue
        with open_broker(venue, headless=self.cfg.headless) as broker:
            if not broker.is_logged_in():
                logger.error(
                    "%s is not usable. %s", venue,
                    "Set ALPACA_API_KEY / ALPACA_SECRET_KEY (free paper keys at "
                    "https://app.alpaca.markets/signup)." if venue == "alpaca"
                    else "Run: python -m tradingagents.live.cli login")
                return
            logger.info("%s session is live", venue)

            while True:
                state = clock.market_state()
                try:
                    summary = self.cycle(broker, state)
                    self._append_journal(summary)
                except Exception:
                    logger.error("cycle failed:\n%s", traceback.format_exc())

                cycles += 1
                if once or (max_cycles and cycles >= max_cycles):
                    logger.info("stopping after %d cycle(s)", cycles)
                    return

                nap = self._sleep_seconds(state)
                logger.info("next cycle in %s", _human(nap))
                time.sleep(nap)

    def _sleep_seconds(self, state) -> float:
        """Cadence follows the session, not a fixed timer.

        Sleeping through a closed market in one long block would also sleep
        through the pre-market news that sets up the open, so the closed-state
        wait is capped rather than run to the bell.
        """
        if state.session == clock.OPEN:
            # Tighten near the close: the last half hour is when a stop that
            # has not been acted on stops being fixable today.
            if (state.minutes_to_close or 999) <= 30:
                return max(30, self.cfg.open_interval / 2)
            return self.cfg.open_interval
        if state.session in (clock.PRE, clock.AFTER):
            return min(self.cfg.closed_interval, 600)
        return min(self.cfg.closed_interval, clock.seconds_until_open(state.now) or 60)

    def _append_journal(self, summary: dict) -> None:
        path = _home() / "live_journal.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def build_llm(config: dict | None = None):
    """Quick-think client from the project's provider config."""
    from tradingagents.llm_clients import create_llm_client
    cfg = config or DEFAULT_CONFIG.copy()
    client = create_llm_client(
        provider=cfg["llm_provider"],
        model=cfg.get("quick_think_llm") or cfg["deep_think_llm"],
        base_url=cfg.get("backend_url"),
    )
    return client.get_llm()
