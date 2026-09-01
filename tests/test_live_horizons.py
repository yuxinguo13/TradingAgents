"""Three books, three clocks — and the one that must not change.

The defect this module exists for is a report that looks like a new portfolio
every morning. Two halves of the fix are tested here: a core list with
hysteresis, so it does not churn on ranking noise, and a swing slot count, so
new ideas fill free slots instead of accumulating forever.

The third failure is subtler and has its own test: a "core" proposal ranked on
raw twelve-month return is a momentum screen with a long-term label on it.
"""

import json
import math
from datetime import date

import pytest

from tradingagents.live import horizons
from tradingagents.live.horizons import (
    CoreHolding,
    DayTradeIdea,
    daytrade_candidates,
    free_slots,
    is_review_day,
    load_core,
    open_swing,
    propose_core,
    review_core,
    save_core,
)


def facts(price=100.0, sma200=80.0, ret_12m=0.4, ret_6m=0.2, off_high=-0.05,
          dollar_vol=5e8, atr_pct=0.02, **kw):
    out = {"price": price, "sma200": sma200, "ret_12m": ret_12m, "ret_6m": ret_6m,
           "off_high": off_high, "dollar_vol": dollar_vol, "atr_pct": atr_pct,
           "spark": "▁▃▅█"}
    out.update(kw)
    return out


class Rec:
    """The subset of Recommendation open_swing reads."""

    def __init__(self, symbol, entry=100.0, stop=95.0, target=115.0, shares=10,
                 issued="2026-08-20", horizon=30):
        self.symbol = symbol
        self.reference_price = entry
        self.initial_stop_price = stop
        self.stop_price = stop
        self.target_price = target
        self.shares = shares
        self.initial_shares = shares
        self.issued_date = issued
        self.horizon_days = horizon

    def days_held(self, as_of=None):
        return (as_of - date.fromisoformat(self.issued_date)).days

    def r_at(self, price):
        return (price - self.reference_price) / (self.reference_price - self.initial_stop_price)

    def pnl_at(self, price):
        return (price - self.reference_price) * self.shares


# ---------------------------------------------------------------------------
# 核心长仓
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCoreFile:
    def test_the_terse_form_a_human_types_is_accepted(self, tmp_path):
        """A file nobody dares edit is not a hand-maintained list."""
        path = tmp_path / "core.json"
        path.write_text(json.dumps({"MSFT": 0.08, "aapl": 0.05}), encoding="utf-8")
        got = {h.symbol: h.weight for h in load_core(path)}
        assert got == {"MSFT": 0.08, "AAPL": 0.05}

    def test_the_full_form_round_trips(self, tmp_path):
        path = tmp_path / "core.json"
        save_core([CoreHolding("MSFT", 0.08, "云与AI", "2026-01-05")], path)
        back = load_core(path)[0]
        assert (back.symbol, back.weight, back.thesis) == ("MSFT", 0.08, "云与AI")

    def test_a_malformed_file_reads_as_empty_rather_than_raising(self, tmp_path):
        path = tmp_path / "core.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_core(path) == []

    def test_a_missing_file_is_an_ordinary_state(self, tmp_path):
        assert load_core(tmp_path / "nope.json") == []


@pytest.mark.unit
class TestReviewCore:
    def test_a_healthy_holding_is_explicitly_left_alone(self):
        line = review_core([CoreHolding("AAA", 0.08)], {"AAA": facts()})[0]
        assert line.status == "持有" and line.action == "不动"

    def test_a_small_dip_below_the_200_day_does_not_fire_the_rule(self):
        """Without the buffer a name oscillating around the line churns monthly."""
        line = review_core([CoreHolding("AAA", 0.08)],
                           {"AAA": facts(price=98.0, sma200=100.0)})[0]
        assert not line.breached and line.status == "持有"

    def test_a_real_break_of_the_200_day_fires_and_says_by_how_much(self):
        line = review_core([CoreHolding("AAA", 0.08)],
                           {"AAA": facts(price=90.0, sma200=100.0)})[0]
        assert line.breached and line.status == "破位"
        assert "10.0%" in line.note and "容忍度 4%" in line.note

    def test_a_year_below_the_floor_fires_even_above_the_200_day(self):
        line = review_core([CoreHolding("AAA", 0.08)],
                           {"AAA": facts(ret_12m=-0.30)})[0]
        assert line.breached and "长期底线" in line.note

    def test_a_break_is_a_prompt_to_think_not_an_instruction_to_sell(self):
        line = review_core([CoreHolding("AAA", 0.08)],
                           {"AAA": facts(ret_12m=-0.30)})[0]
        assert line.action == "复核长期逻辑" and "规则不替你决定" in line.note

    def test_drift_is_only_actioned_on_a_review_day(self):
        holdings = [CoreHolding("AAA", 0.08)]
        quiet = review_core(holdings, {"AAA": facts()}, held_weights={"AAA": 0.12})[0]
        assert quiet.status == "持有"
        loud = review_core(holdings, {"AAA": facts()}, held_weights={"AAA": 0.12},
                           review_day=True)[0]
        assert loud.status == "偏离目标" and loud.action == "减回目标"

    def test_drift_inside_the_band_is_left_alone_even_on_a_review_day(self):
        line = review_core([CoreHolding("AAA", 0.08)], {"AAA": facts()},
                           held_weights={"AAA": 0.09}, review_day=True)[0]
        assert line.status == "持有"

    def test_a_holding_with_no_data_is_shown_as_unknown_not_as_fine(self):
        """Dropping the row would report "nothing wrong" about a name nobody priced."""
        line = review_core([CoreHolding("AAA", 0.08)], {})[0]
        assert line.status == "数据缺失" and "不是「没事」" in line.note

    def test_the_review_window_spans_the_first_days_of_a_month(self):
        assert is_review_day(date(2026, 9, 1)) and is_review_day(date(2026, 9, 4))
        assert not is_review_day(date(2026, 9, 5))


@pytest.mark.unit
class TestProposeCore:
    def test_a_twentyfold_microcap_does_not_outrank_a_real_business(self):
        """Ranking on the raw 12-month return made "长期" a momentum screen."""
        got = propose_core(
            ["MOON", "BIGCO"],
            {"MOON": facts(ret_12m=19.0, dollar_vol=1.2e8),
             "BIGCO": facts(ret_12m=0.35, dollar_vol=9e9)},
            count=1)
        assert [h.symbol for h in got] == ["BIGCO"]

    def test_an_illiquid_name_is_never_proposed_as_a_core_holding(self):
        assert propose_core(["THIN"], {"THIN": facts(dollar_vol=1e6)}) == []

    def test_a_name_below_its_200_day_is_never_proposed(self):
        assert propose_core(["AAA"], {"AAA": facts(price=70.0, sma200=100.0)}) == []

    def test_a_name_already_deep_in_a_drawdown_is_never_proposed(self):
        assert propose_core(["AAA"], {"AAA": facts(off_high=-0.55)}) == []

    def test_a_five_percent_a_day_mover_is_a_trading_vehicle_not_a_holding(self):
        assert propose_core(["AAA"], {"AAA": facts(atr_pct=0.09)}) == []

    def test_the_weights_are_equal_and_leave_room_for_the_rest_of_the_book(self):
        got = propose_core(["A", "B", "C"],
                           {s: facts() for s in "ABC"}, count=3)
        assert len({h.weight for h in got}) == 1
        assert sum(h.weight for h in got) == pytest.approx(0.6, abs=1e-3)

    def test_a_universe_with_nothing_qualifying_proposes_nothing(self):
        assert propose_core(["AAA"], {}) == []


# ---------------------------------------------------------------------------
# 波段
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenSwing:
    def test_the_position_nearest_its_stop_is_listed_first(self):
        """A list sorted by profit puts the one needing a decision last."""
        recs = [Rec("WIN"), Rec("LOSE")]
        out = open_swing(recs, {"WIN": 112.0, "LOSE": 96.0}, date(2026, 8, 25))
        assert [i.symbol for i in out] == ["LOSE", "WIN"]

    def test_a_position_within_two_percent_of_its_stop_is_flagged(self):
        out = open_swing([Rec("AAA")], {"AAA": 96.0}, date(2026, 8, 25))[0]
        assert out.status == "贴近止损"

    def test_a_position_past_one_r_is_told_to_move_its_stop(self):
        out = open_swing([Rec("AAA")], {"AAA": 106.0}, date(2026, 8, 25))[0]
        assert out.r_now == pytest.approx(1.2) and "止损应移到成本" in out.status

    def test_an_unpriceable_position_says_so_rather_than_reading_as_flat(self):
        out = open_swing([Rec("AAA")], {}, date(2026, 8, 25))[0]
        assert out.status == "无法定价" and math.isnan(out.price)

    def test_the_time_stop_countdown_uses_the_horizon_it_was_issued_with(self):
        out = open_swing([Rec("AAA", horizon=30)], {"AAA": 101.0}, date(2026, 9, 15))[0]
        assert out.days_held == 26 and out.days_left == 4

    def test_the_last_three_days_of_the_horizon_are_flagged(self):
        out = open_swing([Rec("AAA", horizon=30)], {"AAA": 101.0}, date(2026, 9, 16))[0]
        assert out.days_left == 3 and out.status == "时间止损将到期"


@pytest.mark.unit
class TestSlots:
    def test_a_full_book_has_no_room_for_a_new_idea(self):
        assert free_slots(6, 6) == 0

    def test_slots_never_go_negative(self):
        assert free_slots(9, 6) == 0

    def test_an_empty_book_has_every_slot(self):
        assert free_slots(0, 6) == 6


# ---------------------------------------------------------------------------
# 日内
# ---------------------------------------------------------------------------

def dt_facts(**kw):
    out = {"price": 100.0, "prev_high": 102.0, "prev_low": 97.0, "atr": 3.0,
           "atr_pct": 0.03, "vol_ratio": 1.8, "dollar_vol": 5e8, "change_pct": 0.02}
    out.update(kw)
    return out


@pytest.mark.unit
class TestDayTrade:
    def test_an_illiquid_name_is_rejected_before_anything_else(self):
        """An intraday trade that cannot be exited is not a trade."""
        assert daytrade_candidates({"AAA": dt_facts(dollar_vol=1e6)}) == []

    def test_a_quiet_name_pays_nobody_intraday(self):
        assert daytrade_candidates({"AAA": dt_facts(atr_pct=0.005)}) == []

    def test_a_name_with_no_volume_expansion_is_not_worth_watching(self):
        assert daytrade_candidates({"AAA": dt_facts(vol_ratio=0.9)}) == []

    def test_the_trigger_sits_just_above_the_prior_days_high(self):
        idea = daytrade_candidates({"AAA": dt_facts()})[0]
        assert idea.long_trigger > idea.prev_high
        assert idea.short_trigger < idea.prev_low

    def test_the_intraday_stop_is_a_fraction_of_the_swing_stop(self):
        """A day trade given two ATRs of room is a swing trade with a new label."""
        idea = daytrade_candidates({"AAA": dt_facts()})[0]
        assert idea.long_trigger - idea.long_stop == pytest.approx(1.5, abs=0.05)
        assert idea.rr == pytest.approx(2.0, abs=0.05)

    def test_the_most_volatile_qualifier_is_listed_first(self):
        out = daytrade_candidates({"CALM": dt_facts(atr_pct=0.026),
                                   "WILD": dt_facts(atr_pct=0.045)})
        assert [i.symbol for i in out] == ["WILD", "CALM"]

    def test_the_section_says_out_loud_that_it_is_not_an_order(self):
        assert "不是下单指令" in horizons.DAYTRADE_CAVEAT
        assert "日线" in horizons.DAYTRADE_CAVEAT
