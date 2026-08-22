"""Size a position from its risk, not from its dollar weight.

The desk's existing rule is a weight cap: no new name may exceed 8% of the
account. That equalises *exposure*, which is the wrong quantity to equalise.
Two $8,000 positions — one in a utility that moves 1% a day, one in a small-cap
semi that moves 6% — are not the same bet. The second loses six times as much
when the thesis breaks, so a book sized by weight is quietly concentrated in
whatever it happens to hold that is most volatile, and nobody chose that.

The rule Van Tharp popularised fixes it by inverting which quantity is the
input:

    quantity = account_value * (risk_pct / 100) / |entry - stop|

Pick the fraction of the account you are willing to lose when the stop is hit
and let the distance to the stop decide the share count. Every position now
loses the same amount when it is wrong, whatever its volatility, and "how much
of the account is in this name" becomes an output. The weight cap stays as an
outer bound (``cap_fraction`` here, ``max_position_weight`` in the Secretary),
because a fixed risk budget divided by a very tight stop still asks for an
absurd number of shares.

That last sentence is the whole reason this module is written defensively. The
formula divides by the stop distance, so a stop equal to the entry asks for an
infinite position and a stop a cent away asks for one that a single gap would
end the account with. Both states are reachable in normal operation: the stop
usually arrives inside an LLM's JSON, an unfilled numeric field arrives as 0.0
or None, and "stop at the entry price" is what a model writes when it means "I
did not set one". So every degenerate case is checked before the division, and
each one returns a quantity of zero plus a sentence saying why — never an
exception, because the caller is a loop that has to still be running next
month.

Ranking uses the same three levels. :func:`r_multiple` is reward over risk;
:func:`expectancy` weights the two legs by a guessed win chance. Only the first
is trusted for ordering — see the comment above :func:`expectancy`.

Nothing here talks to the Secretary. This module proposes a size; the Secretary
still gets the last word, and its limits are the ones that bind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Trade directions, spelled as in ``Holding.side``. Note that investopedia.py
# also exports a constant called SHORT, and it means something else there (the
# order action "Sell Short"). A module importing from both has two different
# SHORTs in scope, so import these by module or alias them.
LONG = "long"
SHORT = "short"

DEFAULT_RISK_PCT = 1.0          # percent of the account lost if the stop fills
MAX_RISK_PCT = 5.0              # above this it is not sizing, it is a bet
DEFAULT_CAP_FRACTION = 0.08     # mirrors RiskLimits.max_new_position_weight
DEFAULT_ATR_STOP_MULT = 2.0
DEFAULT_WIN_CHANCE = 0.5        # a coin flip: a placeholder, not an estimate

# US equities quote in whole cents, so a stop nearer than this to the entry is
# not a different price — it is the entry with rounding noise on it, and the
# division would return a size in the millions.
STOP_TICK = 0.01

# "Sell" is deliberately absent: it closes a long rather than opening a short,
# and an exit is sized by the shares actually held, not by a risk budget.
_DIRECTIONS = {
    "long": LONG, "buy": LONG,
    "short": SHORT, "sell short": SHORT, "sell_short": SHORT,
}


def _as_float(value: object) -> float:
    """Coerce anything to a float, or NaN.

    Values reach this module straight from LLM JSON and from Snapshot fields
    that default to NaN, so ``None``, ``"n/a"`` and ``inf`` are all ordinary
    inputs rather than programming errors. Collapsing every one of them onto
    NaN means the guards below have a single case to test.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _direction(value: object) -> str | None:
    """Normalise a direction, or None if it is not one.

    Accepts the order action a caller may have on hand ("Buy") as well as the
    direction itself, because the two travel together through the panel.
    """
    if not isinstance(value, str):
        return None
    return _DIRECTIONS.get(value.strip().lower())


def _levels(entry: object, stop: object, target: object,
            direction: object) -> tuple[float, float, float] | None:
    """Coerce and order-check the three levels; None when they are incoherent.

    The published expectancy formula takes the absolute value of both legs,
    which means it reports a healthy expectancy for a "long" whose target sits
    *below* its entry: the arithmetic cannot see that the trade is backwards.
    Checking the ordering here is what stops such a trade ranking first.
    """
    d = _direction(direction)
    e, s, t = _as_float(entry), _as_float(stop), _as_float(target)
    if d is None or math.isnan(e) or math.isnan(s) or math.isnan(t):
        return None
    if e <= 0 or s <= 0 or t <= 0:
        return None
    if d == LONG and not s < e < t:
        return None
    if d == SHORT and not t < e < s:
        return None
    if abs(e - s) < STOP_TICK:
        return None
    return e, s, t


# ---------------------------------------------------------------------------
# position sizing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SizingResult:
    """A share count and the sentence that justifies it.

    The reason is populated on success as well as failure. A rejected size is
    only useful if the log says which guard fired, and an accepted
    one is only auditable if the log says what it actually risks.
    """

    quantity: int
    reason: str
    risk_per_share: float = float("nan")
    risk_amount: float = 0.0        # dollars lost if the stop fills exactly
    notional: float = 0.0
    uncapped_quantity: int = 0      # what the rule asked for before the cap test

    def __bool__(self) -> bool:
        return self.quantity > 0


def size_position(
    account_value: float,
    entry: float,
    stop: float,
    risk_pct: float = DEFAULT_RISK_PCT,
    cap_fraction: float = DEFAULT_CAP_FRACTION,
    direction: str = LONG,
) -> SizingResult:
    """Shares to trade so that being stopped out costs ``risk_pct`` of the account.

    Every input the formula cannot handle returns quantity 0 with a reason. The
    arguments are annotated as numbers because numbers are what a caller should
    pass; None, a string or a NaN is a refusal here, not an exception.
    """
    d = _direction(direction)
    if d is None:
        return SizingResult(0, f"unknown direction {direction!r}; expected "
                               f"{LONG!r} or {SHORT!r}")

    av, e, s = _as_float(account_value), _as_float(entry), _as_float(stop)
    rp, cf = _as_float(risk_pct), _as_float(cap_fraction)

    # NaN is tested by name, not left to the bounds checks. Every comparison
    # against NaN is False, so `av <= 0` waves it through and it surfaces much
    # later at int(NaN), which raises ValueError in the middle of a cycle.
    if math.isnan(av) or av <= 0:
        return SizingResult(0, f"account value {account_value!r} is not a positive number")
    if math.isnan(e) or e <= 0:
        return SizingResult(0, f"entry price {entry!r} is not a positive number")
    # A stop of 0.0 is what an unfilled JSON field looks like, and it would be
    # read as "risk the entire share price", which is not what anyone meant.
    if math.isnan(s) or s <= 0:
        return SizingResult(0, f"stop {stop!r} is not a positive number")
    if math.isnan(rp) or rp <= 0:
        return SizingResult(0, f"risk_pct {risk_pct!r} is not a positive number")
    if rp > MAX_RISK_PCT:
        return SizingResult(0, f"risk_pct {rp:g} is above the {MAX_RISK_PCT:g}% ceiling "
                               f"(risk_pct is a percent, so 1.0 means 1%)")
    if math.isnan(cf) or cf <= 0 or cf > 1:
        return SizingResult(0, f"cap_fraction {cap_fraction!r} must be a fraction in (0, 1]")

    # The classic way this formula ends an account: risk per share of zero.
    if e == s:
        return SizingResult(0, f"stop equals entry ({e:,.2f}), so risk per share is zero "
                               f"and the rule asks for an unbounded position")
    if d == LONG and s > e:
        return SizingResult(0, f"long stop {s:,.2f} is above entry {e:,.2f}; a long stops "
                               f"out below its entry")
    if d == SHORT and s < e:
        return SizingResult(0, f"short stop {s:,.2f} is below entry {e:,.2f}; a short stops "
                               f"out above its entry")

    dist = abs(e - s)
    if dist < STOP_TICK:
        return SizingResult(0, f"stop {s:,.4f} is {dist:,.4f} from entry {e:,.4f}, inside "
                               f"the one cent equities quote in", risk_per_share=dist)

    budget = av * rp / 100.0
    # Floor, never round: rounding up spends more than the risk budget, and the
    # point of the budget is that it is never exceeded.
    raw = int(budget / dist)
    if raw <= 0:
        return SizingResult(0, f"risk budget ${budget:,.2f} does not cover one share's "
                               f"${dist:,.2f} of risk", risk_per_share=dist)

    cap_shares = int(cf * av / e)
    if cap_shares <= 0:
        return SizingResult(0, f"one share at ${e:,.2f} exceeds the {cf:.0%} exposure cap "
                               f"(${cf * av:,.2f})", risk_per_share=dist,
                            uncapped_quantity=raw)
    if raw > cap_shares:
        # Refused rather than trimmed to the cap, which is the opposite of what
        # the Secretary does with an oversized order, and deliberately so. The
        # Secretary is resizing a view it believes; here the breach is evidence
        # that the *stop* is wrong — a stop this tight relative to the risk
        # budget is usually noise-width, not a level the trade is invalidated
        # at. Trimming would keep the bad trade and hide the bad stop. The cost
        # of this choice is that a genuinely tight stop on a quiet name gets
        # refused too; uncapped_quantity is reported so a caller that wants to
        # trim anyway can see what was asked for.
        return SizingResult(0, f"risk sizing wants {raw:,} shares (${raw * e:,.0f}) but the "
                               f"{cf:.0%} cap allows {cap_shares:,}; the stop is too tight "
                               f"for this risk budget", risk_per_share=dist,
                            uncapped_quantity=raw)

    return SizingResult(
        quantity=raw,
        reason=(f"{raw:,} {d} shares at ${e:,.2f} risks ${raw * dist:,.2f} "
                f"({raw * dist / av:.2%} of the account) if the stop at ${s:,.2f} fills"),
        risk_per_share=dist, risk_amount=raw * dist, notional=raw * e,
        uncapped_quantity=raw,
    )


def risk_based_quantity(
    account_value: float,
    entry: float,
    stop: float,
    risk_pct: float = DEFAULT_RISK_PCT,
    cap_fraction: float = DEFAULT_CAP_FRACTION,
    direction: str = LONG,
) -> int:
    """Share count only. 0 means "do not trade this"; :func:`size_position` says why."""
    return size_position(account_value, entry, stop, risk_pct, cap_fraction, direction).quantity


def stop_from_atr(
    entry: float,
    atr: float | None = None,
    *,
    atr_pct: float | None = None,
    k: float = DEFAULT_ATR_STOP_MULT,
    direction: str = LONG,
) -> float | None:
    """A stop k ATRs from the entry, or None if one cannot be derived.

    The panel is not required to name a stop, and most of the time it does not.
    Without one the sizing rule above has nothing to divide by, so the existing
    ATR the desk already computes per symbol supplies a stand-in: a level far
    enough out that ordinary daily noise does not reach it.

    ``atr_pct`` is accepted because that is the form Snapshot carries (ATR as a
    fraction of price); pass either, not both.

    k=2 is a convention, not a fitted number. It puts the stop near 6% out on a
    3%-ATR name, just inside the desk's own -8% stop_loss trigger, so the two
    mechanisms do not fight. Nothing in this repo has back-tested it.
    """
    e, mult = _as_float(entry), _as_float(k)
    if math.isnan(e) or e <= 0 or math.isnan(mult) or mult <= 0:
        return None

    a = _as_float(atr)
    if math.isnan(a):
        pct = _as_float(atr_pct)
        if math.isnan(pct) or pct <= 0:
            return None
        a = pct * e
    if a <= 0:
        return None

    d = _direction(direction)
    if d is None:
        return None

    stop = e - mult * a if d == LONG else e + mult * a
    # Rounded to a real price: an unrounded stop implies a precision the venue
    # cannot fill at and makes risk-per-share look more exact than it is.
    stop = round(stop, 2)
    # A near-zero ATR (an untraded name, or a bad bar) produces a stop that is
    # either non-positive or indistinguishable from the entry. Report it as
    # unavailable here rather than handing a degenerate stop to the sizer.
    if stop <= 0 or abs(e - stop) < STOP_TICK:
        return None
    return stop


# ---------------------------------------------------------------------------
# ranking statistics
# ---------------------------------------------------------------------------

def r_multiple(entry: float, stop: float, target: float,
               direction: str = LONG) -> float:
    """Reward over risk. NaN when the levels do not describe a coherent trade.

    This is the honest ranking statistic. It is made entirely of numbers the
    trade itself asserts — where it is entered, where it is wrong, where it is
    taken — and it needs no opinion about how often it works.
    """
    lv = _levels(entry, stop, target, direction)
    if lv is None:
        return float("nan")
    e, s, t = lv
    return abs(t - e) / abs(e - s)


# --- why win_chance is the weak link ---------------------------------------
#
# expectancy() below is the reference implementation's ranking statistic, and
# its first three arguments are facts about the trade. The fourth is not: nobody
# on this desk has ever validated a win_chance. It is a human guess, or worse a
# number an LLM emitted because the schema had a field for it, and no persona
# here has been scored against its own realised outcomes.
#
# That would be tolerable if expectancy were insensitive to it. It is the
# opposite: expectancy is *linear* in win_chance, with slope (reward + risk).
# A panel that is systematically 15 points overconfident does not produce
# slightly optimistic rankings, it adds a constant to every trade's score, and
# it will push trades whose true expectancy is negative across zero. If that
# number were also allowed to set size, the same bias would inflate every
# position at once — which is precisely why risk_based_quantity() takes no
# win_chance argument, and why nothing in this module lets one reach a share
# count. Size comes from the stop distance, which is a level, not a belief.
#
# So: rank with r_multiple(), which requires no guess. Treat expectancy() as
# advisory — useful for comparing two setups where someone has an actual
# reason to believe the win rates differ, and misleading everywhere else.
# breakeven_win_chance() inverts the problem into the more answerable form:
# instead of guessing how often this wins, it states how often it must win, and
# leaves a human to judge whether that is plausible.


def expectancy(entry: float, stop: float, target: float,
               win_chance: float = DEFAULT_WIN_CHANCE,
               direction: str = LONG) -> float:
    """Probability-weighted return per dollar invested. Advisory only.

    Both legs are expressed as a fraction of the entry price, matching the
    reference implementation this was taken from. NaN when the levels are
    incoherent or ``win_chance`` is not a probability.
    """
    lv = _levels(entry, stop, target, direction)
    if lv is None:
        return float("nan")
    p = _as_float(win_chance)
    if math.isnan(p) or not 0.0 <= p <= 1.0:
        return float("nan")
    e, s, t = lv
    reward = abs(t - e) / e
    risk = abs(e - s) / e
    return reward * p - risk * (1.0 - p)


def breakeven_win_chance(entry: float, stop: float, target: float,
                         direction: str = LONG) -> float:
    """How often this trade must work to break even: 1 / (1 + R).

    The question a trader can actually answer, unlike "what is the win rate".
    NaN when the levels are incoherent.
    """
    r = r_multiple(entry, stop, target, direction)
    if math.isnan(r):
        return float("nan")
    return 1.0 / (1.0 + r)


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """One proposed trade, described by the three levels that define it.

    ``win_chance`` is optional and defaults to None rather than to a number,
    so that "nobody estimated this" stays distinguishable from "somebody
    estimated a coin flip".
    """

    symbol: str
    entry: float
    stop: float
    target: float
    direction: str = LONG
    win_chance: float | None = None
    source: str = ""

    def problem(self) -> str:
        """Empty when the levels are coherent, otherwise why they are not."""
        if _direction(self.direction) is None:
            return f"unknown direction {self.direction!r}"
        if _levels(self.entry, self.stop, self.target, self.direction) is None:
            return (f"levels do not describe a {self.direction} trade "
                    f"(entry {self.entry!r}, stop {self.stop!r}, target {self.target!r})")
        return ""

    def is_valid(self) -> bool:
        return not self.problem()

    def risk_per_share(self) -> float:
        lv = _levels(self.entry, self.stop, self.target, self.direction)
        return float("nan") if lv is None else abs(lv[0] - lv[1])

    def reward_per_share(self) -> float:
        lv = _levels(self.entry, self.stop, self.target, self.direction)
        return float("nan") if lv is None else abs(lv[2] - lv[0])

    def r_multiple(self) -> float:
        return r_multiple(self.entry, self.stop, self.target, self.direction)

    def expectancy(self, win_chance: float | None = None) -> float:
        """Advisory. Falls back to the trade's own guess, then to a coin flip."""
        p = win_chance if win_chance is not None else self.win_chance
        if p is None:
            p = DEFAULT_WIN_CHANCE
        return expectancy(self.entry, self.stop, self.target, p, self.direction)

    def breakeven_win_chance(self) -> float:
        return breakeven_win_chance(self.entry, self.stop, self.target, self.direction)

    def size(self, account_value: float, risk_pct: float = DEFAULT_RISK_PCT,
             cap_fraction: float = DEFAULT_CAP_FRACTION) -> SizingResult:
        return size_position(account_value, self.entry, self.stop, risk_pct,
                             cap_fraction, self.direction)

    def __str__(self) -> str:
        r = self.r_multiple()
        rs = "n/a" if math.isnan(r) else f"{r:.2f}R"
        return (f"{self.symbol} {self.direction} @ {self.entry:,.2f} "
                f"stop {self.stop:,.2f} target {self.target:,.2f} ({rs})")


def _rank_key(value: float) -> float:
    """NaN sorts last.

    sorted() with a NaN key does not raise, it silently produces an arbitrary
    order — NaN compares False against everything, so an unrankable trade can
    land anywhere in the list, first included. Mapping it to -inf makes "we
    could not score this" mean "bottom", deterministically.
    """
    return float("-inf") if math.isnan(value) else value


def rank_by_r_multiple(trades: list[Trade]) -> list[Trade]:
    """Best reward-to-risk first. The preferred ordering: it guesses nothing."""
    return sorted(trades, key=lambda t: _rank_key(t.r_multiple()), reverse=True)


def rank_by_expectancy(trades: list[Trade],
                       default_win_chance: float = DEFAULT_WIN_CHANCE) -> list[Trade]:
    """Best expectancy first, breaking ties on R. Advisory — read the note above.

    Trades carrying no win_chance are scored at ``default_win_chance``, which is
    0.5 and is a placeholder rather than an estimate. Sorting is stable, so
    equally-scored trades keep the order they came in, which keeps a cycle's
    output reproducible.
    """
    return sorted(
        trades,
        key=lambda t: (_rank_key(t.expectancy(t.win_chance if t.win_chance is not None
                                              else default_win_chance)),
                       _rank_key(t.r_multiple())),
        reverse=True,
    )
