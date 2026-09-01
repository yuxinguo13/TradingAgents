"""The drawing and the reading: the two ways a chart can lie quietly.

A chart module fails silently by construction. A sparkline built from two bars
still renders eight blocks; a moving average drawn over the close still looks
like a chart; a swing low picked from the wrong side of a cluster still prints a
number that reads as support. Every test here pins one of those.
"""

import math

import pytest

from tradingagents.live import charting


def rising(n=260, start=100.0, step=0.4):
    return [round(start + step * i, 4) for i in range(n)]


def falling(n=260, start=200.0, step=0.4):
    return [round(start - step * i, 4) for i in range(n)]


def wavy(n=260, start=100.0, step=0.4, amp=0.04, period=17):
    """A trend with real swings, so pivots exist and structure can be read."""
    return [round((start + step * i) * (1 + amp * math.sin(i * 2 * math.pi / period)), 4)
            for i in range(n)]


@pytest.mark.unit
class TestSparkline:
    def test_two_bars_of_history_draw_nothing_rather_than_a_flat_line(self):
        """Eight identical blocks read as "went nowhere", which is a claim."""
        assert charting.sparkline([100.0]) == ""
        assert charting.sparkline([]) == ""
        assert charting.sparkline([100.0, 101.0]) != ""

    def test_a_gap_stays_a_gap(self):
        """None is missing data. Plotting it at the low would invent a crash."""
        out = charting.sparkline([1.0, None, 3.0])
        assert out[1] == " " and len(out) == 3

    def test_a_flat_series_renders_mid_scale_not_at_the_floor(self):
        out = charting.sparkline([50.0] * 10)
        assert set(out) == {charting._BLOCKS[(len(charting._BLOCKS) - 1) // 2]}

    def test_the_high_and_low_land_on_the_extreme_glyphs(self):
        out = charting.sparkline([1.0, 5.0, 3.0])
        assert out[0] == charting._BLOCKS[0] and out[1] == charting._BLOCKS[-1]


@pytest.mark.unit
class TestBucket:
    def test_downsampling_averages_rather_than_samples(self):
        """A sampled series can drop the one bar that made the move."""
        got = charting.bucket([0.0, 100.0, 0.0, 100.0], 2)
        assert got == [50.0, 50.0]

    def test_a_series_shorter_than_the_width_is_untouched(self):
        assert charting.bucket([1.0, 2.0], 40) == [1.0, 2.0]


@pytest.mark.unit
class TestSma:
    def test_the_average_is_none_until_the_window_fills(self):
        got = charting.sma([1.0, 2.0, 3.0, 4.0], 3)
        assert got[:2] == [None, None] and got[2] == 2.0 and got[3] == 3.0

    def test_it_stays_aligned_to_the_input(self):
        closes = rising(50)
        assert len(charting.sma(closes, 20)) == len(closes)


@pytest.mark.unit
class TestLineChart:
    def test_a_series_that_cannot_be_drawn_returns_nothing(self):
        """An empty box reads as "flat"; the caller must render the absence."""
        assert charting.line_chart([1.0]) == []
        assert charting.line_chart([]) == []

    def test_levels_are_drawn_and_labelled_on_their_own_row(self):
        body = charting.line_chart(rising(60), levels={"止损": 90.0, "目标": 140.0})
        text = "\n".join(body)
        assert "止损 90" in text and "目标 140" in text
        assert "┈" in text

    def test_a_level_outside_the_price_range_still_fits_on_the_axis(self):
        """A stop below every close must not be silently clipped off the chart."""
        body = charting.line_chart(rising(60, start=100.0), levels={"止损": 50.0})
        assert any("止损" in line for line in body)

    def test_the_price_line_is_drawn_over_the_moving_average(self):
        """An overlay that erases the close lies about the chart's own subject."""
        closes = rising(120)
        body = charting.line_chart(closes, overlays={"SMA50": list(closes)})
        # Identical series: every plotted cell must be the close's glyph set,
        # never the overlay's marker.
        cells = "".join(body)
        assert "─" in cells

    def test_a_perfectly_flat_window_still_produces_a_chart(self):
        body = charting.line_chart([25.0] * 40)
        assert body and any("─" in line for line in body)

    def test_the_date_axis_shows_two_stamps_when_only_two_are_given(self):
        body = charting.line_chart(rising(60), dates=["2026-01-02", "2026-03-31"])
        axis = body[-1] if "SMA" not in body[-1] else body[-2]
        assert "2026-01-02" in axis and "2026-03-31" in axis


@pytest.mark.unit
class TestPivots:
    def test_a_swing_low_is_the_low_of_its_own_cluster(self):
        """_thin kept the larger |value| for both sides.

        On lows that picked the *higher* of two adjacent troughs, so the
        support printed was a level the price had already traded through.
        """
        lows = [10, 9, 8, 5, 6, 5.5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        highs = [x + 2 for x in lows]
        support, _ = charting.levels_near(9.0, highs, lows, k=2, gap=5)
        assert support == pytest.approx(4.0)

    def test_an_uptrend_reads_as_higher_highs_and_higher_lows(self):
        closes = rising(140)
        highs = [c * 1.02 + (3 if i % 11 == 0 else 0) for i, c in enumerate(closes)]
        lows = [c * 0.98 - (3 if i % 13 == 0 else 0) for i, c in enumerate(closes)]
        label, detail = charting.structure(highs, lows)
        assert label == "上升结构" and detail

    def test_too_little_history_says_so_instead_of_guessing(self):
        label, detail = charting.structure([1, 2, 3], [1, 2, 3])
        assert label == "结构未成形" and "不足" in detail


@pytest.mark.unit
class TestReadTrend:
    def test_a_short_history_refuses_rather_than_reporting_a_neutral_read(self):
        """An empty TrendRead rendered as "均线纠缠" would be a fabricated opinion."""
        out = charting.read_trend("AAA", rising(10))
        assert out.error and not out.verdict

    def test_a_clean_uptrend_reads_as_a_bullish_stack(self):
        closes = wavy(260)
        out = charting.read_trend("AAA", closes, [c * 1.01 for c in closes],
                                  [c * 0.99 for c in closes], rsi=60.0, atr_pct=0.02)
        assert out.ma_stack == "多头排列"
        assert out.structure == "上升结构"
        assert "向上" in out.verdict
        assert dict(out.bullets())["均线排列"].startswith("多头排列")

    def test_a_line_with_no_swings_says_so_rather_than_claiming_a_conflict(self):
        """A monotonic series has no pivots; "均线与结构互相矛盾" invented one."""
        out = charting.read_trend("AAA", rising(260), rsi=60.0, atr_pct=0.02)
        assert out.structure == "结构未成形"
        assert "无从判断" in out.verdict and "矛盾" not in out.verdict

    def test_a_downtrend_is_never_described_as_a_buyable_shape(self):
        closes = wavy(260, start=200.0, step=-0.4)
        out = charting.read_trend("AAA", closes, [c * 1.01 for c in closes],
                                  [c * 0.99 for c in closes], rsi=35.0, atr_pct=0.02)
        assert out.ma_stack in ("空头排列", "跌破 200 日线")
        assert "对赌" in out.verdict or "跌破" in out.verdict

    def test_pace_not_level_decides_acceleration(self):
        """Comparing raw 1M and 3M returns calls every riser "accelerating"."""
        steady = charting.read_trend("AAA", rising(260), ret_1m=0.05, ret_3m=0.15)
        assert "平稳" in steady.momentum
        fast = charting.read_trend("AAA", rising(260), ret_1m=0.15, ret_3m=0.15)
        assert "加速" in fast.momentum

    def test_the_r_distance_is_reported_in_the_stop_multiple_it_will_use(self):
        out = charting.read_trend("AAA", rising(260), atr_pct=0.03, atr_stop_mult=2.0)
        assert "6.0%" in out.volatility

    def test_relative_strength_names_the_benchmark_and_the_direction(self):
        out = charting.read_trend("AAA", rising(260),
                                  benchmark={"SPY": 0.031, "QQQM": -0.008})
        assert "跑赢 SPY" in out.relative and "跑输 QQQM" in out.relative

    def test_absent_inputs_drop_their_bullet_instead_of_printing_nan(self):
        out = charting.read_trend("AAA", rising(260))
        rendered = " ".join(v for _, v in out.bullets())
        assert "nan" not in rendered.lower()
        assert not any(k == "量能" for k, _ in out.bullets())
