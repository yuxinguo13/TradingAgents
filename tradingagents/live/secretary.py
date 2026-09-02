"""The Secretary: nothing reaches the venue until this module says yes.

StockAgent puts a "secretary" between each LLM agent and the market whose only
job is to reject malformed or infeasible actions and hand back a reason the
agent can retry against. That separation is the single most transferable idea
in the paper, and it matters far more here than it does in a simulation:
StockAgent's secretary guards a fictional order book, this one guards a live
account that the agent will keep trading against for weeks unattended.

So the same seam carries two jobs that must not be confused:

* **Validation** — is this JSON well-formed, and does it name a real action,
  a real symbol, a positive integer quantity? Failures here are conversational:
  they come back as a message the model can fix.
* **Risk** — even a perfectly-formed order can be a bad idea. Position caps,
  cash, per-day trade budget, churn cooldowns, price sanity, and a kill switch
  live here. Failures here are terminal for that order; the model does not get
  to argue its way past a limit.

Every limit is a plain number in :class:`RiskLimits`, not a prompt instruction,
because a limit an LLM can talk itself out of is not a limit.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .investopedia import ACTIONS, BUY, COVER, LIMIT, MARKET, SELL, SHORT, STOP, Account

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")


@dataclass
class RiskLimits:
    """Hard bounds on what the agent may do. Edit deliberately."""

    max_position_weight: float = 0.12     # any one name, as a share of account value
    max_new_position_weight: float = 0.08  # a fresh entry starts smaller than the cap
    max_gross_exposure: float = 0.95      # always hold some cash
    max_trades_per_day: int = 12
    max_turnover_per_day: float = 0.35    # traded notional / account value
    min_order_value: float = 250.0        # below this, fees-equivalent noise
    max_order_value_pct: float = 0.10     # no single order larger than this
    symbol_cooldown_minutes: int = 45     # do not re-trade a name immediately
    max_limit_deviation: float = 0.05     # limit price within 5% of last
    allow_short: bool = False             # shorting is off unless switched on
    min_price: float = 3.00               # no sub-$3 names
    require_market_open: bool = True

    @classmethod
    def from_env(cls) -> "RiskLimits":
        """Env overrides, so limits can be tightened without a code edit."""
        lim = cls()
        for f in lim.__dataclass_fields__:
            raw = os.getenv(f"TRADINGAGENTS_RISK_{f.upper()}")
            if raw is None:
                continue
            cur = getattr(lim, f)
            if isinstance(cur, bool):
                setattr(lim, f, raw.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(lim, f, int(raw))
            else:
                setattr(lim, f, float(raw))
        return lim


@dataclass
class Order:
    symbol: str
    action: str                       # Buy | Sell | Sell Short | Buy to Cover
    quantity: int
    order_type: str = MARKET
    limit_price: float | None = None
    rationale: str = ""
    confidence: float = 0.5
    source: str = ""                  # which persona / rule produced it

    def notional(self, price: float) -> float:
        return self.quantity * price

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    order: Order | None = None
    retryable: bool = False           # True = malformed, model may fix and resubmit


def kill_switch_path() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents")) / "STOP"


def kill_switch_engaged() -> bool:
    """A file the user can touch to stop all trading immediately.

    A file rather than a config flag or a signal: it works when the process is
    detached, from any shell, with no restart, and it survives the agent
    crashing and being restarted by a supervisor.
    """
    return kill_switch_path().exists()


class TradeLedger:
    """Per-day record of what the agent has already done.

    Both the daily budget and the churn cooldown need memory that outlives a
    single loop iteration and a process restart, so it is persisted.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else (
            Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
            / "live_ledger.json"
        )
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            return [e for e in data if e.get("at", "") >= cutoff]
        except Exception:
            return []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def record(self, order: Order, price: float, ok: bool, message: str = "",
               venue: str = "") -> None:
        self.entries.append({
            "at": datetime.now().isoformat(), "symbol": order.symbol,
            "action": order.action, "quantity": order.quantity, "price": price,
            "notional": order.notional(price), "ok": ok, "message": message,
            "source": order.source, "venue": str(venue or ""),
            "rationale": order.rationale[:400],
        })
        self.save()

    def _for(self, venue: str) -> list[dict]:
        """The rows that count against one venue's budgets.

        There is one ledger and more than one venue behind it, and the daily
        limits are not venue-neutral: ``max_turnover_per_day`` is a fraction of
        *account value*, and each venue is a separate account with its own.
        Summed across venues and divided by one of them, the number means
        nothing — and it locks a venue out over trades that never touched it.
        Five cleanup sells on the local paper book spent $38,258 and shut the
        Alpaca bridge for the day against a $35,000 limit it had not used.

        The churn cooldown is the same story one symbol at a time: selling NVDA
        on paper should not stop the desk buying NVDA at the broker.

        A row with no venue recorded predates this field and cannot be
        attributed, so it counts everywhere. Over-counting refuses trades that
        were allowed; under-counting allows trades that were not, and only one
        of those is recoverable. The ambiguity ages out with the day.
        """
        if not venue:
            return self.entries
        v = str(venue).strip().lower()
        return [e for e in self.entries
                if not e.get("venue") or str(e["venue"]).strip().lower() == v]

    def today(self, venue: str = "") -> list[dict]:
        today = date.today().isoformat()
        return [e for e in self._for(venue) if e.get("at", "").startswith(today)
                and e.get("ok")]

    def trades_today(self, venue: str = "") -> int:
        return len(self.today(venue))

    def turnover_today(self, venue: str = "") -> float:
        return sum(abs(e.get("notional", 0.0)) for e in self.today(venue))

    def last_trade_at(self, symbol: str, venue: str = "") -> datetime | None:
        stamps = [e["at"] for e in self._for(venue)
                  if e.get("symbol") == symbol.upper() and e.get("ok")]
        if not stamps:
            return None
        try:
            return datetime.fromisoformat(max(stamps))
        except ValueError:
            return None


class Secretary:
    """Validate, then risk-check. Nothing else may call the broker."""

    def __init__(self, limits: RiskLimits | None = None,
                 ledger: TradeLedger | None = None, venue: str = ""):
        self.limits = limits or RiskLimits.from_env()
        self.ledger = ledger or TradeLedger()
        # Which account the budgets below are being spent from. Empty means
        # "every row counts", which is the old behaviour and the right default
        # for a caller that does not know. See :meth:`TradeLedger._for`.
        self.venue = str(venue or "")

    # --- 1. format validation ----------------------------------------------

    def parse_order(self, raw: str | dict) -> Verdict:
        """Turn model output into an :class:`Order`, or explain what is wrong.

        Rejections here are ``retryable``: the message is written to be handed
        straight back to the model as the next prompt, which is how StockAgent
        converts a malformed response into a usable one instead of dropping it.
        """
        if isinstance(raw, dict):
            data = raw
        else:
            text = str(raw)
            # Models wrap JSON in prose and fences; take the outermost object.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return Verdict(False, "No JSON object found in the response. "
                                      "Reply with a single JSON object only.",
                               retryable=True)
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                return Verdict(False, f"Invalid JSON ({e}). Reply with one valid "
                                      f"JSON object and nothing else.", retryable=True)

        act = str(data.get("action", "")).strip().title()
        # Accept the shorthand a model naturally produces.
        act = {"Buy": BUY, "Sell": SELL, "Short": SHORT, "Sell Short": SHORT,
               "Cover": COVER, "Buy To Cover": COVER, "Hold": "Hold",
               "None": "Hold", "No": "Hold"}.get(act, act)

        if act == "Hold":
            return Verdict(True, "hold", order=None)
        if act not in ACTIONS:
            return Verdict(False, f"'action' must be one of {list(ACTIONS)} or 'Hold'; "
                                  f"got {data.get('action')!r}.", retryable=True)

        sym = str(data.get("symbol", "")).strip().upper()
        if not _SYMBOL_RE.match(sym):
            return Verdict(False, f"'symbol' must be a ticker like 'NVDA'; got {sym!r}.",
                           retryable=True)

        try:
            qty = int(float(data.get("quantity", 0)))
        except (TypeError, ValueError):
            return Verdict(False, "'quantity' must be a whole number of shares.",
                           retryable=True)
        if qty <= 0:
            return Verdict(False, "'quantity' must be a positive whole number of shares.",
                           retryable=True)

        otype = str(data.get("order_type", MARKET)).strip().title()
        otype = {"Market": MARKET, "Limit": LIMIT, "Stop": STOP}.get(otype, MARKET)
        lim = data.get("limit_price")
        lim = float(lim) if lim not in (None, "", "null") else None
        if otype == LIMIT and lim is None:
            return Verdict(False, "A Limit order requires a numeric 'limit_price'.",
                           retryable=True)

        conf = data.get("confidence", 0.5)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5

        return Verdict(True, "parsed", order=Order(
            symbol=sym, action=act, quantity=qty, order_type=otype,
            limit_price=lim, rationale=str(data.get("rationale", ""))[:1000],
            confidence=conf, source=str(data.get("source", "")),
        ))

    # --- 2. risk check ------------------------------------------------------

    def check(self, order: Order, account: Account, price: float,
              market_open: bool = True) -> Verdict:
        """Approve, reject, or *resize* the order against the live account.

        Resizing rather than rejecting is deliberate for size breaches: an
        order that is merely too large carries a valid view, and trimming it to
        the cap keeps the view while enforcing the limit. Everything else — no
        cash, no shares, wrong session, kill switch — is a flat no.
        """
        L = self.limits

        if kill_switch_engaged():
            return Verdict(False, f"kill switch engaged ({kill_switch_path()})")

        if L.require_market_open and not market_open:
            return Verdict(False, "market is closed")

        if order.action in (SHORT, COVER) and not L.allow_short:
            return Verdict(False, "shorting disabled (allow_short=False)")

        if price <= 0:
            return Verdict(False, "no usable price")
        if price < L.min_price:
            return Verdict(False, f"price {price:.2f} below floor {L.min_price:.2f}")

        equity = account.account_value or (account.cash + sum(
            h.market_value for h in account.holdings))
        if equity <= 0:
            return Verdict(False, "account value is zero — cannot size anything")

        # --- daily budget ---
        if self.ledger.trades_today(self.venue) >= L.max_trades_per_day:
            return Verdict(False, f"daily trade budget spent "
                                  f"({L.max_trades_per_day} trades)")
        turn = self.ledger.turnover_today(self.venue)
        if turn >= L.max_turnover_per_day * equity:
            return Verdict(False, f"daily turnover budget spent "
                                  f"({turn:,.0f} / {L.max_turnover_per_day:.0%} of equity)")

        # --- churn cooldown ---
        last = self.ledger.last_trade_at(order.symbol, self.venue)
        if last and (datetime.now() - last) < timedelta(minutes=L.symbol_cooldown_minutes):
            mins = (datetime.now() - last).total_seconds() / 60
            return Verdict(False, f"{order.symbol} traded {mins:.0f}m ago; "
                                  f"cooldown is {L.symbol_cooldown_minutes}m")

        # --- limit sanity ---
        if order.order_type == LIMIT and order.limit_price:
            dev = abs(order.limit_price - price) / price
            if dev > L.max_limit_deviation:
                return Verdict(False, f"limit {order.limit_price:.2f} is {dev:.1%} from "
                                      f"last {price:.2f}; max {L.max_limit_deviation:.0%}")

        held = account.position(order.symbol)
        held_qty = held.quantity if held else 0.0
        qty = order.quantity

        if order.action in (SELL, COVER):
            if held_qty <= 0:
                return Verdict(False, f"no {order.symbol} position to sell")
            if qty > held_qty:
                qty = int(held_qty)   # sell what exists, not what was imagined
            if qty <= 0:
                return Verdict(False, f"{order.symbol} position too small to sell")
            return Verdict(True, "approved",
                           order=Order(**{**order.to_dict(), "quantity": qty}))

        # --- BUY / SHORT sizing ---
        cap_w = L.max_position_weight if held_qty > 0 else L.max_new_position_weight
        current_val = held_qty * price
        room_value = max(0.0, cap_w * equity - current_val)
        if room_value < L.min_order_value:
            return Verdict(False, f"{order.symbol} already at {current_val / equity:.1%} "
                                  f"of equity; cap is {cap_w:.0%}")

        gross = sum(h.market_value for h in account.holdings)
        gross_room = max(0.0, L.max_gross_exposure * equity - gross)
        cash_room = max(0.0, account.buying_power or account.cash)
        order_cap = L.max_order_value_pct * equity

        budget = min(room_value, gross_room, cash_room, order_cap)
        if budget < L.min_order_value:
            binding = min(
                [(room_value, "position cap"), (gross_room, "gross exposure cap"),
                 (cash_room, "available cash"), (order_cap, "per-order cap")],
                key=lambda p: p[0])[1]
            return Verdict(False, f"no room to buy {order.symbol}: {binding} "
                                  f"leaves ${budget:,.0f} (< ${L.min_order_value:,.0f})")

        max_qty = int(budget // price)
        if max_qty <= 0:
            return Verdict(False, f"${budget:,.0f} buys no whole shares at {price:.2f}")
        if qty > max_qty:
            qty = max_qty

        if qty * price < L.min_order_value:
            return Verdict(False, f"order value ${qty * price:,.0f} below minimum "
                                  f"${L.min_order_value:,.0f}")

        resized = " (resized to fit limits)" if qty != order.quantity else ""
        return Verdict(True, f"approved{resized}",
                       order=Order(**{**order.to_dict(), "quantity": qty}))

    # --- convenience --------------------------------------------------------

    def vet(self, raw: str | dict, account: Account, price: float,
            market_open: bool = True) -> Verdict:
        """parse + risk-check in one call."""
        v = self.parse_order(raw)
        if not v.ok or v.order is None:
            return v
        return self.check(v.order, account, price, market_open)
