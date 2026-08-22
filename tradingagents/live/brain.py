"""Hybrid decision layer: rules decide *when* to think, the panel decides *what*.

An always-on agent that asks an LLM about every name on every cycle burns its
quota by 10am and spends most of those calls concluding "nothing has changed".
So the two halves of the decision are split by what each is actually good at.

**Rules decide when.** :func:`triggers` is pure arithmetic over price, volume
and the news feed. It answers one question — has anything happened to this
symbol that could change a position? — and on a quiet cycle it answers "no" for
every name, at zero cost. This is also the honest reading of "monitor all the
time": the monitoring is continuous, the *reasoning* is event-driven.

**The panel decides what.** Only triggered symbols get an evidence pack and a
vote. Four personas answer independently, a weighted consensus is taken, and
the Risk Officer reviews the result. Anything that survives goes to the
Secretary, which can still say no.

The consensus rule is deliberately blunt: a trade needs agreement, not a
majority of one. Disagreement resolves to Hold, because the cost of a missed
trade is bounded and the cost of a coin-flip trade compounds.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv

from .investopedia import Account
from .newsfeed import NewsItem
from .personas import PANEL, risk_officer_prompt
from .secretary import Order

logger = logging.getLogger(__name__)

# --- trigger thresholds -----------------------------------------------------
NEWS_MATERIALITY_TRIGGER = 7      # keyword score at which news alone is enough
NEWS_MAX_AGE_HOURS = 24           # older than this is history, not an event
NEWS_TRIGGERS_PER_SYMBOL = 3      # the most material few, not the whole feed
MOVE_ATR_TRIGGER = 1.2            # move since last close, in ATRs
STOP_LOSS_PCT = -0.08             # a position down this much demands a decision
TAKE_PROFIT_PCT = 0.20            # so does one up this much
VOLUME_SURGE = 2.0                # today's volume vs 20-day average


@dataclass
class Trigger:
    symbol: str
    kind: str
    detail: str
    urgency: int = 1              # 1 = routine, 3 = act now

    def __str__(self) -> str:
        return f"{self.symbol}: {self.kind} — {self.detail}"


@dataclass
class Snapshot:
    """The cheap numeric read of a symbol, computed once and reused."""
    symbol: str
    price: float = 0.0
    prev_close: float = 0.0
    change_pct: float = 0.0
    atr_pct: float = 0.0
    move_atrs: float = 0.0
    rsi14: float = float("nan")
    sma20: float = float("nan")
    sma50: float = float("nan")
    sma200: float = float("nan")
    vol_ratio: float = float("nan")
    ret_1m: float = float("nan")
    ret_3m: float = float("nan")
    off_high_52w: float = float("nan")
    ok: bool = False
    error: str = ""


def snapshot(symbol: str, date: str) -> Snapshot:
    """Compute the numeric state of one symbol from cached OHLCV.

    Uses the same look-ahead-filtered loader the rest of the framework uses, so
    a trigger can never fire on a bar the desk would not have had.
    """
    s = Snapshot(symbol=symbol)
    try:
        df = load_ohlcv(symbol, date)
        if df is None or len(df) < 30:
            s.error = "insufficient history"
            return s
        c = pd.to_numeric(df["Close"], errors="coerce").dropna()
        h = pd.to_numeric(df["High"], errors="coerce")
        l = pd.to_numeric(df["Low"], errors="coerce")
        v = pd.to_numeric(df["Volume"], errors="coerce")

        s.price = float(c.iloc[-1])
        s.prev_close = float(c.iloc[-2])
        s.change_pct = s.price / s.prev_close - 1.0

        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
        s.atr_pct = atr / s.price if s.price else 0.0
        s.move_atrs = abs(s.price - s.prev_close) / atr if atr else 0.0

        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        s.rsi14 = float(100 - 100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else float("nan")

        for w, attr in ((20, "sma20"), (50, "sma50"), (200, "sma200")):
            if len(c) >= w:
                setattr(s, attr, float(c.rolling(w).mean().iloc[-1]))

        if len(v) >= 21 and v.tail(20).mean():
            s.vol_ratio = float(v.iloc[-1] / v.tail(20).mean())
        if len(c) > 21:
            s.ret_1m = float(c.iloc[-1] / c.iloc[-22] - 1)
        if len(c) > 63:
            s.ret_3m = float(c.iloc[-1] / c.iloc[-64] - 1)
        hi52 = float(c.tail(252).max())
        s.off_high_52w = s.price / hi52 - 1 if hi52 else float("nan")
        s.ok = True
    except Exception as exc:
        s.error = f"{type(exc).__name__}: {exc}"
    return s


def triggers(
    symbol: str,
    snap: Snapshot,
    news: list[NewsItem],
    account: Account,
    screen_rank: int | None = None,
) -> list[Trigger]:
    """Everything about ``symbol`` that argues for a decision right now.

    An empty list is the normal, desirable outcome. Held names are checked more
    aggressively than watched ones because an open position has downside that a
    watchlist entry does not.
    """
    out: list[Trigger] = []
    held = account.position(symbol)

    # "New to this process" and "new to the world" are different things. Google
    # News happily returns a quarter of back-coverage, so a first run — or any
    # run after the seen-set is pruned — would otherwise fire dozens of
    # triggers on last quarter's earnings. Age is the guard that makes the
    # novelty check mean "this just happened".
    fresh = [n for n in news
             if n.ticker == symbol
             and n.materiality >= NEWS_MATERIALITY_TRIGGER
             and n.age_hours() <= NEWS_MAX_AGE_HOURS]
    fresh.sort(key=lambda n: (-n.materiality, n.age_hours()))
    for item in fresh[:NEWS_TRIGGERS_PER_SYMBOL]:
        out.append(Trigger(symbol, "news",
                           f"[{item.materiality}/{item.lean}] {item.title[:120]}",
                           urgency=3 if item.materiality >= 9 else 2))

    if not snap.ok:
        return out

    if snap.move_atrs >= MOVE_ATR_TRIGGER:
        out.append(Trigger(symbol, "price_move",
                           f"{snap.change_pct:+.2%} = {snap.move_atrs:.1f} ATR",
                           urgency=2 if snap.move_atrs >= 2 else 1))

    if not pd.isna(snap.vol_ratio) and snap.vol_ratio >= VOLUME_SURGE:
        out.append(Trigger(symbol, "volume",
                           f"{snap.vol_ratio:.1f}x 20d average volume"))

    if held and held.avg_cost > 0:
        pl = snap.price / held.avg_cost - 1
        if pl <= STOP_LOSS_PCT:
            out.append(Trigger(symbol, "stop_loss",
                               f"position {pl:+.1%} vs cost {held.avg_cost:.2f}",
                               urgency=3))
        elif pl >= TAKE_PROFIT_PCT:
            out.append(Trigger(symbol, "take_profit", f"position {pl:+.1%}", urgency=2))
        # A held name losing its trend is a decision even with no news: this is
        # the check that stops a "long-term conviction" quietly becoming a
        # position nobody chose to keep.
        if not pd.isna(snap.sma50) and snap.price < snap.sma50 and snap.ret_1m < -0.05:
            out.append(Trigger(symbol, "trend_break",
                               f"below SMA50 ({snap.sma50:.2f}), 1m {snap.ret_1m:+.1%}",
                               urgency=2))

    if not held and screen_rank is not None and screen_rank <= 15:
        # Urgency 0: strictly below every event-driven trigger. A screen entry
        # says only "this name ranks well and you do not own it" — it is not
        # news, it will still be true next cycle, and it must never take the
        # LLM budget from a stop-loss or a name that just moved 1.4 ATR.
        out.append(Trigger(symbol, "screen_entry",
                           f"rank #{screen_rank} on the universe screen, no position",
                           urgency=0))

    return out


# ----------------------------------------------------------------------------
# evidence pack
# ----------------------------------------------------------------------------

def _fmt(x, pct=False, plus=False):
    if x is None or (isinstance(x, float) and (pd.isna(x) or np.isinf(x))):
        return "n/a"
    if pct:
        return f"{x:+.2%}" if plus else f"{x:.2%}"
    return f"{x:,.2f}"


def build_evidence(
    symbol: str,
    snap: Snapshot,
    news: list[NewsItem],
    account: Account,
    trigs: list[Trigger],
    macro: list[NewsItem] | None = None,
    phase: str = "",
) -> str:
    """A compact, LLM-sized evidence pack.

    Compact on purpose. The framework's full brief runs tens of thousands of
    characters, which is right for one considered daily decision and wrong for
    a loop that may run this many times an hour across many names. Everything
    here is something a trader would actually look at before acting.
    """
    held = account.position(symbol)
    equity = account.account_value or account.cash
    lines = [f"# {symbol} — decision pack ({datetime.now():%Y-%m-%d %H:%M} ET"
             + (f", {phase}" if phase else "") + ")"]

    lines += ["", "## Why you are being asked"]
    lines += [f"- {t.kind}: {t.detail}" for t in trigs] or ["- routine review"]

    lines += ["", "## Account"]
    lines.append(f"- Account value ${equity:,.2f} · cash ${account.cash:,.2f} · "
                 f"buying power ${account.buying_power:,.2f}")
    if held:
        pl = (snap.price / held.avg_cost - 1) if held.avg_cost else 0.0
        lines.append(f"- **Position: {held.quantity:g} shares @ ${held.avg_cost:,.2f} "
                     f"avg cost, now ${snap.price:,.2f} ({pl:+.1%}), "
                     f"{(held.market_value / equity if equity else 0):.1%} of account**")
    else:
        lines.append(f"- **No position in {symbol}**")
    if account.holdings:
        book = ", ".join(f"{h.symbol} {h.quantity:g}" for h in account.holdings[:15])
        lines.append(f"- Rest of book: {book}")

    lines += ["", "## Price and trend"]
    if snap.ok:
        lines += [
            f"- Last ${snap.price:,.2f}, prior close ${snap.prev_close:,.2f} "
            f"({snap.change_pct:+.2%}, {snap.move_atrs:.1f} ATR)",
            f"- 14d ATR {_fmt(snap.atr_pct, pct=True)} of price · RSI(14) {_fmt(snap.rsi14)}",
            f"- SMA20 {_fmt(snap.sma20)} · SMA50 {_fmt(snap.sma50)} · SMA200 {_fmt(snap.sma200)}",
            f"- Return 1m {_fmt(snap.ret_1m, pct=True, plus=True)} · "
            f"3m {_fmt(snap.ret_3m, pct=True, plus=True)} · "
            f"{_fmt(snap.off_high_52w, pct=True, plus=True)} from the 52-week high",
            f"- Volume {_fmt(snap.vol_ratio)}x its 20-day average",
        ]
    else:
        lines.append(f"- price data unavailable: {snap.error}")

    tnews = [n for n in news if n.ticker == symbol]
    lines += ["", f"## Fresh news on {symbol}"]
    if tnews:
        lines += [f"- [{n.materiality}/{n.lean}] {n.title} ({n.source}, "
                  f"{n.age_hours():.0f}h ago)" for n in tnews[:10]]
    else:
        lines.append("- (nothing new since the last check)")

    if macro:
        lines += ["", "## Market-wide headlines"]
        lines += [f"- [{n.materiality}/{n.lean}] {n.title}" for n in macro[:5]]

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# the panel
# ----------------------------------------------------------------------------

@dataclass
class Vote:
    persona: str
    action: str = "Hold"
    quantity: int = 0
    confidence: float = 0.0
    rationale: str = ""
    error: str = ""


@dataclass
class PanelResult:
    symbol: str
    votes: list[Vote] = field(default_factory=list)
    consensus: str = "Hold"
    order: Order | None = None
    veto: bool = False
    concern: str = ""
    scale: float = 1.0

    def summary(self) -> str:
        parts = [f"{v.persona.split()[0]}:{v.action}"
                 f"{f'({v.confidence:.1f})' if v.action != 'Hold' else ''}"
                 for v in self.votes]
        s = f"{self.symbol} panel → {' '.join(parts)} ⇒ {self.consensus}"
        if self.veto:
            s += f" [VETOED: {self.concern}]"
        elif self.scale < 1.0:
            s += f" [scaled {self.scale:.0%}: {self.concern}]"
        return s


class Panel:
    """Runs the personas against one evidence pack and reconciles their votes."""

    def __init__(self, llm, secretary, min_agreement: float = 0.5,
                 min_confidence: float = 0.55):
        self.llm = llm
        self.secretary = secretary
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence

    def _ask(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return getattr(resp, "content", str(resp)) or ""

    def _vote(self, persona, evidence: str) -> Vote:
        v = Vote(persona=persona.name)
        try:
            raw = self._ask(persona.system_prompt(), evidence)
            parsed = self.secretary.parse_order(raw)
            if not parsed.ok:
                # One retry with the validator's own complaint, exactly as
                # StockAgent's secretary does — most format failures are fixed
                # by being told precisely what was wrong.
                raw = self._ask(persona.system_prompt(),
                                f"{evidence}\n\nYour previous reply was rejected: "
                                f"{parsed.reason}\nReply again, correctly.")
                parsed = self.secretary.parse_order(raw)
            if not parsed.ok:
                v.error = parsed.reason
                return v
            if parsed.order is None:          # a well-formed Hold
                v.action = "Hold"
                return v
            o = parsed.order
            v.action, v.quantity = o.action, o.quantity
            v.confidence, v.rationale = o.confidence, o.rationale
        except Exception as exc:
            v.error = f"{type(exc).__name__}: {exc}"
            logger.warning("persona %s failed: %s", persona.name, exc)
        return v

    def deliberate(self, symbol: str, evidence: str, account: Account,
                   price: float) -> PanelResult:
        res = PanelResult(symbol=symbol)
        res.votes = [self._vote(p, evidence) for p in PANEL]

        # Weighted tally by *seat*, not by conviction. Each persona contributes
        # its full weight to whichever action it named, and confidence is used
        # only for sizing afterwards. Folding confidence into the tally lets one
        # emphatic member outvote two calm ones — and since a Hold is naturally
        # reported at low confidence ("no view"), that scheme makes the panel
        # structurally unable to decline a trade. A persona that errored
        # abstains rather than counting as a Hold: an API failure is not a
        # market opinion.
        weights = {p.name: p.weight for p in PANEL}
        tally: dict[str, float] = {}
        live = 0.0
        for v in res.votes:
            if v.error:
                continue
            w = weights.get(v.persona, 1.0)
            live += w
            tally[v.action] = tally.get(v.action, 0.0) + w

        if not live:
            res.consensus = "Hold"
            res.concern = "every panel member failed to respond"
            return res

        winner = max(tally, key=tally.get) if tally else "Hold"
        share = tally.get(winner, 0.0) / live
        if winner == "Hold" or share < self.min_agreement:
            res.consensus = "Hold"
            return res

        backers = [v for v in res.votes if v.action == winner and not v.error]
        conf = float(np.mean([v.confidence for v in backers])) if backers else 0.0
        # Agreement is necessary but not sufficient: a panel that all shrugs its
        # way to the same weak Buy has not found a trade, it has found a mood.
        if conf < self.min_confidence:
            res.consensus = "Hold"
            res.concern = (f"agreement on {winner} but mean confidence "
                           f"{conf:.2f} < {self.min_confidence:.2f}")
            return res
        res.consensus = winner

        # Size on the median of those who actually wanted this action, so one
        # enthusiastic persona cannot set the position by itself.
        qtys = [v.quantity for v in res.votes if v.action == winner and v.quantity > 0]
        if not qtys:
            res.consensus = "Hold"
            return res
        qty = int(np.median(qtys))

        rationale = " | ".join(f"{v.persona.split()[0]}: {v.rationale}"
                               for v in backers if v.rationale)[:900]

        proposed = Order(symbol=symbol, action=winner, quantity=qty,
                         confidence=conf, rationale=rationale,
                         source=f"panel({len(backers)}/{len(res.votes)})")

        res.veto, res.concern, res.scale = self._risk_review(proposed, evidence, price)
        if res.veto:
            res.consensus = "Hold"
            return res
        if res.scale < 1.0:
            proposed.quantity = max(1, int(qty * res.scale))
        res.order = proposed
        return res

    def _risk_review(self, order: Order, evidence: str, price: float) -> tuple[bool, str, float]:
        """Ask the Risk Officer. A failure here is not a free pass.

        If the reviewer cannot be reached, the trade is scaled down rather than
        waved through: an unreviewed order should be smaller than a reviewed
        one, not the same size.
        """
        import json
        try:
            body = (f"{evidence}\n\n## Proposed trade\n"
                    f"{order.action} {order.quantity} {order.symbol} at ~${price:,.2f} "
                    f"(${order.quantity * price:,.0f} notional)\n"
                    f"Panel rationale: {order.rationale}")
            raw = self._ask(risk_officer_prompt(), body)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return False, "risk officer gave no verdict", 0.6
            d = json.loads(m.group(0))
            veto = bool(d.get("veto", False))
            concern = str(d.get("concern", ""))[:300]
            scale = float(d.get("scale", 1.0))
            return veto, concern, max(0.1, min(1.0, scale))
        except Exception as exc:
            logger.warning("risk review failed: %s", exc)
            return False, f"risk review unavailable ({type(exc).__name__})", 0.6
