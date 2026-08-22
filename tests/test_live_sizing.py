"""Risk-based sizing: the arithmetic, and every way it can be fed nonsense.

The sizing rule divides by the distance from entry to stop, so the interesting
tests are not the ones that check the multiplication. They are the ones that
feed it the inputs an LLM actually produces — a stop equal to the entry, a stop
on the wrong side, a missing field arriving as None or NaN — and assert that
the result is zero shares and a sentence, never an exception and never a
position the account cannot survive.
"""

import math
from itertools import pairwise

import pytest

from tradingagents.live.sizing import (
    DEFAULT_CAP_FRACTION,
    LONG,
    MAX_RISK_PCT,
    SHORT,
    Trade,
    breakeven_win_chance,
    expectancy,
    r_multiple,
    rank_by_expectancy,
    rank_by_r_multiple,
    risk_based_quantity,
    size_position,
    stop_from_atr,
)

ACCOUNT = 100_000.0

# Junk that reaches this module in normal operation: unfilled JSON fields,
# Snapshot attributes that default to NaN, and a model writing prose in a
# numeric slot.
JUNK = [None, "", "n/a", float("nan"), float("inf"), float("-inf"), [], {}, object()]


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRiskBasedQuantity:
    def test_size_is_risk_budget_over_stop_distance(self):
        # 1% of $100k = $1,000 of risk; $10 of risk per share → 100 shares.
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0) == 100

    def test_result_reports_what_it_actually_risks(self):
        r = size_position(ACCOUNT, 50.0, 40.0, 1.0)
        assert r.risk_per_share == 10.0
        assert r.risk_amount == 1_000.0
        assert r.notional == 5_000.0

    def test_equal_risk_across_unequal_volatility(self):
        """The whole point: a wider stop buys fewer shares, not more risk.

        Sizing by dollar weight would put the same $8,000 into both names and
        leave the volatile one carrying several times the loss.
        """
        tight = size_position(ACCOUNT, 100.0, 98.0, 1.0, cap_fraction=0.5)
        wide = size_position(ACCOUNT, 100.0, 90.0, 1.0, cap_fraction=0.5)
        assert tight.quantity > wide.quantity
        assert tight.notional > wide.notional
        # Both lose ~$1,000 at their stop, within the one share lost to flooring.
        assert abs(tight.risk_amount - wide.risk_amount) <= 10.0

    def test_short_sizes_from_a_stop_above_the_entry(self):
        assert risk_based_quantity(ACCOUNT, 50.0, 60.0, 1.0, direction=SHORT) == 100

    def test_quantity_floors_and_never_exceeds_the_budget(self):
        # $1,000 / $3 = 333.33 shares. Rounding up would risk $1,002.
        r = size_position(ACCOUNT, 30.0, 27.0, 1.0, cap_fraction=0.5)
        assert r.quantity == 333
        assert r.risk_amount <= 1_000.0

    def test_order_action_is_accepted_as_a_direction(self):
        # The action travels with the trade through the panel; "Buy" is a long.
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0, direction="Buy") == 100
        assert risk_based_quantity(ACCOUNT, 50.0, 60.0, 1.0, direction="Sell Short") == 100

    def test_win_chance_cannot_reach_a_share_count(self):
        """A guessed win rate must never be able to inflate a position.

        Sizing keys off the stop distance, which is a level. If win_chance were
        an argument here, a systematically overconfident panel would size every
        position up at once.
        """
        import inspect

        params = inspect.signature(risk_based_quantity).parameters
        assert "win_chance" not in params
        assert "confidence" not in params


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDegenerateStops:
    def test_stop_equal_to_entry_returns_zero(self):
        """Division by zero risk-per-share. The classic account-ender.

        A model that has not set a stop writes the entry price into the field,
        and unguarded this asks for an infinite position.
        """
        r = size_position(ACCOUNT, 50.0, 50.0, 1.0)
        assert r.quantity == 0
        assert "stop equals entry" in r.reason

    def test_stop_inside_a_cent_of_entry_returns_zero(self):
        """The continuous form of the same failure: $1,000 / $0.001 = a million shares."""
        r = size_position(ACCOUNT, 50.0, 49.999, 1.0)
        assert r.quantity == 0 and r.reason

    def test_long_stop_above_entry_returns_zero(self):
        r = size_position(ACCOUNT, 50.0, 55.0, 1.0, direction=LONG)
        assert r.quantity == 0
        assert "above entry" in r.reason

    def test_short_stop_below_entry_returns_zero(self):
        r = size_position(ACCOUNT, 50.0, 45.0, 1.0, direction=SHORT)
        assert r.quantity == 0
        assert "below entry" in r.reason

    def test_wrong_side_is_not_rescued_by_absolute_value(self):
        # |entry - stop| alone would happily size a backwards trade.
        assert risk_based_quantity(ACCOUNT, 50.0, 55.0, 1.0, direction=LONG) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, 45.0, 1.0, direction=SHORT) == 0

    def test_zero_stop_returns_zero(self):
        # An unfilled numeric field arrives as 0.0 and would mean "risk the
        # entire share price", which is not what the omission meant.
        r = size_position(ACCOUNT, 50.0, 0.0, 1.0)
        assert r.quantity == 0 and "stop" in r.reason

    def test_negative_prices_return_zero(self):
        assert risk_based_quantity(ACCOUNT, -50.0, -60.0, 1.0) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, -10.0, 1.0) == 0


@pytest.mark.unit
class TestInputGuards:
    def test_zero_or_negative_account_returns_zero(self):
        assert risk_based_quantity(0.0, 50.0, 40.0, 1.0) == 0
        assert risk_based_quantity(-1_000.0, 50.0, 40.0, 1.0) == 0

    def test_zero_entry_returns_zero(self):
        assert risk_based_quantity(ACCOUNT, 0.0, 40.0, 1.0) == 0

    def test_non_positive_risk_pct_returns_zero(self):
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 0.0) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, -2.0) == 0

    def test_risk_pct_above_the_ceiling_returns_zero(self):
        """A risk_pct of 50 is a units mistake, not a mandate to bet half the account."""
        r = size_position(ACCOUNT, 50.0, 40.0, MAX_RISK_PCT + 0.1)
        assert r.quantity == 0
        assert "ceiling" in r.reason

    def test_fraction_instead_of_percent_undersizes_rather_than_oversizes(self):
        # Passing 0.01 meaning "1%" is the likely confusion. It must fail small.
        small = risk_based_quantity(ACCOUNT, 50.0, 40.0, 0.01)
        correct = risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0)
        assert 0 <= small < correct

    def test_bad_cap_fraction_returns_zero(self):
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0, cap_fraction=0.0) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0, cap_fraction=1.5) == 0

    def test_unknown_direction_returns_zero(self):
        r = size_position(ACCOUNT, 50.0, 40.0, 1.0, direction="sideways")
        assert r.quantity == 0 and "direction" in r.reason

    def test_plain_sell_is_not_a_direction(self):
        # "Sell" closes a long; an exit is sized by shares held, not by risk.
        assert risk_based_quantity(ACCOUNT, 50.0, 60.0, 1.0, direction="Sell") == 0

    @pytest.mark.parametrize("junk", JUNK)
    def test_junk_never_raises_and_never_sizes(self, junk):
        """NaN slips through every `<= 0` test and only blows up at int(NaN).

        This module runs inside a loop meant to survive weeks unattended, so a
        malformed field has to degrade to "no trade", not to a traceback.
        """
        assert risk_based_quantity(junk, 50.0, 40.0, 1.0) == 0
        assert risk_based_quantity(ACCOUNT, junk, 40.0, 1.0) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, junk, 1.0) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, junk) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0, cap_fraction=junk) == 0
        assert risk_based_quantity(ACCOUNT, 50.0, 40.0, 1.0, direction=junk) == 0

    @pytest.mark.parametrize("case", [
        (ACCOUNT, 50.0, 50.0, 1.0),          # stop == entry
        (ACCOUNT, 50.0, 55.0, 1.0),          # wrong side
        (ACCOUNT, 50.0, 0.0, 1.0),           # unfilled stop
        (0.0, 50.0, 40.0, 1.0),              # no account
        (ACCOUNT, 50.0, 40.0, 0.0),          # no risk budget
        (ACCOUNT, 50.0, 40.0, 99.0),         # risk_pct above the ceiling
        (ACCOUNT, 50.0, 49.995, 1.0),        # sub-cent stop
    ])
    def test_every_rejection_states_a_reason(self, case):
        r = size_position(*case)
        assert r.quantity == 0
        assert r.reason.strip(), "a rejection with no reason is unloggable"


@pytest.mark.unit
class TestExposureCap:
    def test_position_above_the_cap_is_trimmed_to_the_cap(self):
        # A $0.50 stop on a $50 stock: $1,000 of risk would buy 2,000 shares =
        # the entire account. The cap trims it to 160 shares ($8,000), which
        # risks $80 — under budget, which is the safe direction to miss in.
        r = size_position(ACCOUNT, 50.0, 49.5, 1.0, cap_fraction=DEFAULT_CAP_FRACTION)
        assert r.quantity == 160
        assert r.notional == pytest.approx(8_000.0)
        assert r.risk_amount == pytest.approx(80.0)
        assert "cap" in r.reason

    def test_a_trimmed_position_never_exceeds_the_risk_budget(self):
        # Trimming can only reduce risk. Assert it across a range of stops so a
        # future change cannot quietly invert the inequality.
        budget = ACCOUNT * 0.01
        for stop in (49.5, 48.0, 45.0, 43.75, 40.0, 30.0):
            r = size_position(ACCOUNT, 50.0, stop, 1.0, cap_fraction=0.08)
            assert r.risk_amount <= budget + 1e-9, f"stop {stop} over budget"

    def test_refusal_is_still_available_explicitly(self):
        r = size_position(ACCOUNT, 50.0, 49.5, 1.0,
                          cap_fraction=DEFAULT_CAP_FRACTION, clamp_to_cap=False)
        assert r.quantity == 0 and "cap" in r.reason

    def test_trimmed_size_still_reports_what_was_asked_for(self):
        r = size_position(ACCOUNT, 50.0, 49.5, 1.0, cap_fraction=DEFAULT_CAP_FRACTION)
        assert r.uncapped_quantity == 2_000

    def test_an_ordinary_atr_stop_is_not_rejected(self):
        # Regression: with 1% risk and an 8% cap, refusing on cap breach
        # rejected any stop nearer than 12.5% — i.e. a 2xATR stop on a 3%-ATR
        # name, which is entirely ordinary. That returned zero recommendations
        # for nearly every real candidate.
        r = size_position(ACCOUNT, 100.0, 94.0, 1.0, cap_fraction=0.08)
        assert r.quantity > 0

    def test_a_position_exactly_at_the_cap_is_allowed(self):
        # 8% of $100k = $8,000 = 160 shares at $50; a $6.25 stop asks for 160.
        r = size_position(ACCOUNT, 50.0, 43.75, 1.0, cap_fraction=0.08)
        assert r.quantity == 160

    def test_share_price_above_the_cap_returns_zero(self):
        # 8% of a $10k account is $800; one $1,200 share does not fit in it.
        r = size_position(10_000.0, 1_200.0, 1_150.0, 1.0, cap_fraction=0.08)
        assert r.quantity == 0 and "cap" in r.reason

    def test_risk_budget_below_one_share_of_risk_returns_zero(self):
        # $1 of budget against $10 of risk per share buys nothing.
        r = size_position(100.0, 50.0, 40.0, 1.0)
        assert r.quantity == 0 and "risk budget" in r.reason


# ---------------------------------------------------------------------------
# ATR-derived stops
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStopFromAtr:
    def test_long_stop_is_k_atrs_below_entry(self):
        assert stop_from_atr(100.0, 2.0, k=2.0) == 96.0

    def test_short_stop_is_k_atrs_above_entry(self):
        assert stop_from_atr(100.0, 2.0, k=2.0, direction=SHORT) == 104.0

    def test_accepts_atr_as_a_fraction_of_price(self):
        # Snapshot carries atr_pct, not ATR in dollars.
        assert stop_from_atr(100.0, atr_pct=0.03, k=2.0) == 94.0

    def test_derived_stop_feeds_the_sizer(self):
        stop = stop_from_atr(100.0, atr_pct=0.03, k=2.0)
        assert risk_based_quantity(ACCOUNT, 100.0, stop, 1.0, cap_fraction=0.5) == 166

    def test_stop_is_rounded_to_a_tradeable_price(self):
        s = stop_from_atr(100.0, 1.337, k=2.0)
        assert s == round(s, 2)

    @pytest.mark.parametrize("atr", [0.0, -1.0, None, float("nan"), "n/a"])
    def test_unusable_atr_reports_unavailable(self, atr):
        """No stop is better than a fabricated one: the sizer refuses, nothing raises."""
        assert stop_from_atr(100.0, atr) is None

    def test_near_zero_atr_reports_unavailable(self):
        # Rounds back onto the entry, which the sizer would reject anyway.
        assert stop_from_atr(100.0, 0.001, k=2.0) is None

    def test_atr_wider_than_the_price_reports_unavailable(self):
        # entry - k*ATR <= 0 is not a stop.
        assert stop_from_atr(10.0, 8.0, k=2.0) is None

    def test_bad_entry_or_multiple_reports_unavailable(self):
        assert stop_from_atr(0.0, 2.0) is None
        assert stop_from_atr(100.0, 2.0, k=0.0) is None
        assert stop_from_atr(100.0, 2.0, direction="sideways") is None


# ---------------------------------------------------------------------------
# ranking statistics
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRMultiple:
    def test_reward_over_risk(self):
        assert r_multiple(100.0, 90.0, 130.0) == 3.0

    def test_short_reward_over_risk(self):
        assert r_multiple(100.0, 110.0, 70.0, direction=SHORT) == 3.0

    def test_stop_at_entry_is_not_a_number(self):
        assert math.isnan(r_multiple(100.0, 100.0, 130.0))

    def test_target_on_the_wrong_side_is_not_a_number(self):
        """The reference formula's abs() reports a reward for a backwards trade."""
        assert math.isnan(r_multiple(100.0, 90.0, 70.0, direction=LONG))
        assert math.isnan(r_multiple(100.0, 110.0, 130.0, direction=SHORT))

    @pytest.mark.parametrize("junk", JUNK)
    def test_junk_is_not_a_number_rather_than_an_exception(self, junk):
        assert math.isnan(r_multiple(junk, 90.0, 130.0))
        assert math.isnan(r_multiple(100.0, junk, 130.0))
        assert math.isnan(r_multiple(100.0, 90.0, junk))


@pytest.mark.unit
class TestExpectancy:
    def test_matches_the_published_formula(self):
        # |(130-100)/100| * 0.5 - |(100-90)/100| * 0.5 = 0.15 - 0.05
        assert expectancy(100.0, 90.0, 130.0, 0.5) == pytest.approx(0.10)

    def test_certainty_collapses_to_one_leg(self):
        assert expectancy(100.0, 90.0, 130.0, 1.0) == pytest.approx(0.30)
        assert expectancy(100.0, 90.0, 130.0, 0.0) == pytest.approx(-0.10)

    def test_expectancy_is_linear_in_win_chance(self):
        """The reason win_chance is the weak link, stated as an assertion.

        Constant slope means a systematically overconfident panel does not
        shade the ranking, it adds a constant to every score.
        """
        e = [expectancy(100.0, 90.0, 130.0, p) for p in (0.4, 0.5, 0.6, 0.7)]
        steps = [b - a for a, b in pairwise(e)]
        assert steps[0] == pytest.approx(steps[1]) == pytest.approx(steps[2])
        # Slope is reward + risk, both as fractions of the entry.
        assert steps[0] == pytest.approx(0.1 * (0.30 + 0.10))

    def test_overconfidence_flips_the_verdict(self):
        """Same trade, two guesses: one says take it, the other says do not."""
        assert expectancy(100.0, 90.0, 105.0, 0.5) < 0
        assert expectancy(100.0, 90.0, 105.0, 0.8) > 0

    def test_win_chance_outside_zero_to_one_is_not_a_number(self):
        assert math.isnan(expectancy(100.0, 90.0, 130.0, 1.4))
        assert math.isnan(expectancy(100.0, 90.0, 130.0, -0.2))

    def test_backwards_trade_is_not_a_number(self):
        assert math.isnan(expectancy(100.0, 90.0, 70.0, 0.6, direction=LONG))

    @pytest.mark.parametrize("junk", JUNK)
    def test_junk_is_not_a_number_rather_than_an_exception(self, junk):
        assert math.isnan(expectancy(100.0, 90.0, 130.0, junk))
        assert math.isnan(expectancy(junk, 90.0, 130.0, 0.5))


@pytest.mark.unit
class TestBreakevenWinChance:
    def test_inverse_of_one_plus_r(self):
        assert breakeven_win_chance(100.0, 90.0, 130.0) == pytest.approx(0.25)
        assert breakeven_win_chance(100.0, 90.0, 110.0) == pytest.approx(0.5)

    def test_expectancy_at_breakeven_is_zero(self):
        p = breakeven_win_chance(100.0, 90.0, 130.0)
        assert expectancy(100.0, 90.0, 130.0, p) == pytest.approx(0.0, abs=1e-12)

    def test_incoherent_levels_are_not_a_number(self):
        assert math.isnan(breakeven_win_chance(100.0, 100.0, 130.0))


# ---------------------------------------------------------------------------
# trades and ranking
# ---------------------------------------------------------------------------

def trade(symbol="MU", entry=100.0, stop=90.0, target=130.0, direction=LONG,
          win_chance=None) -> Trade:
    return Trade(symbol=symbol, entry=entry, stop=stop, target=target,
                 direction=direction, win_chance=win_chance)


@pytest.mark.unit
class TestTrade:
    def test_levels_drive_the_statistics(self):
        t = trade()
        assert t.risk_per_share() == 10.0
        assert t.reward_per_share() == 30.0
        assert t.r_multiple() == 3.0

    def test_missing_win_chance_falls_back_to_a_coin_flip(self):
        # None means "nobody estimated one" and must stay distinguishable.
        assert trade().win_chance is None
        assert trade().expectancy() == pytest.approx(0.10)

    def test_stated_win_chance_is_used(self):
        assert trade(win_chance=1.0).expectancy() == pytest.approx(0.30)

    def test_backwards_trade_reports_its_problem(self):
        t = trade(target=70.0)
        assert not t.is_valid()
        assert t.problem()

    def test_valid_trade_has_no_problem(self):
        assert trade().is_valid()
        assert trade().problem() == ""

    def test_sizing_runs_off_the_trade(self):
        assert trade(entry=50.0, stop=40.0).size(ACCOUNT, 1.0).quantity == 100

    def test_str_survives_an_incoherent_trade(self):
        # This lands in a log line during an unattended cycle; it must not raise.
        assert "n/a" in str(trade(stop=100.0))


@pytest.mark.unit
class TestRanking:
    def test_empty_list(self):
        assert rank_by_expectancy([]) == []
        assert rank_by_r_multiple([]) == []

    def test_r_multiple_ranks_best_reward_to_risk_first(self):
        a = trade("AAA", 100.0, 95.0, 130.0)     # 6R
        b = trade("BBB", 100.0, 90.0, 130.0)     # 3R
        c = trade("CCC", 100.0, 80.0, 110.0)     # 0.5R
        assert [t.symbol for t in rank_by_r_multiple([c, b, a])] == ["AAA", "BBB", "CCC"]

    def test_expectancy_ranks_best_first(self):
        a = trade("AAA", 100.0, 90.0, 150.0)
        b = trade("BBB", 100.0, 90.0, 110.0)
        assert [t.symbol for t in rank_by_expectancy([b, a])] == ["AAA", "BBB"]

    def test_per_trade_win_chance_is_honoured(self):
        modest = trade("AAA", 100.0, 90.0, 115.0, win_chance=0.9)
        rich = trade("BBB", 100.0, 90.0, 130.0, win_chance=0.1)
        assert [t.symbol for t in rank_by_expectancy([rich, modest])] == ["AAA", "BBB"]

    def test_confident_guess_outranks_the_better_trade(self):
        """Documents the weakness rather than pretending it is not there.

        BBB has twice the reward-to-risk and still ranks second because someone
        typed a higher number into AAA's win_chance. r_multiple is immune.
        """
        aaa = trade("AAA", 100.0, 90.0, 115.0, win_chance=0.9)
        bbb = trade("BBB", 100.0, 90.0, 130.0, win_chance=0.1)
        assert [t.symbol for t in rank_by_expectancy([aaa, bbb])] == ["AAA", "BBB"]
        assert [t.symbol for t in rank_by_r_multiple([aaa, bbb])] == ["BBB", "AAA"]

    def test_unrankable_trades_sink_to_the_bottom(self):
        """A NaN key does not raise, it silently scatters the order.

        Without the -inf mapping a trade nobody can score could rank first and
        be the one the desk acts on.
        """
        broken = trade("BAD", 100.0, 100.0, 130.0)
        good = trade("OK", 100.0, 90.0, 130.0)
        assert [t.symbol for t in rank_by_r_multiple([broken, good])] == ["OK", "BAD"]
        assert [t.symbol for t in rank_by_expectancy([broken, good])] == ["OK", "BAD"]

    def test_all_unrankable_still_returns_every_trade(self):
        trades = [trade("A", 1.0, 1.0, 1.0), trade("B", 2.0, 2.0, 2.0)]
        assert len(rank_by_expectancy(trades)) == 2

    def test_ties_keep_their_input_order(self):
        # A cycle's output has to be reproducible from one run to the next.
        a, b, c = trade("AAA"), trade("BBB"), trade("CCC")
        assert [t.symbol for t in rank_by_expectancy([c, a, b])] == ["CCC", "AAA", "BBB"]
