"""Financial statements: what an absent number must never render as.

A momentum screen over the Nasdaq returns mostly pre-revenue biotech. For those
names P/E, PEG, margin and revenue growth are not low — they do not exist, and
a renderer that prints 0.00 for "unknowable" turns a missing fact into a bad
one. Most of this file is that distinction.
"""

import json
import math
import time

import pytest

from tradingagents.live import fundamentals as fund
from tradingagents.live.fundamentals import (
    Fundamentals,
    FundamentalsBook,
    Quarter,
    Surprise,
)


def profitable(**kw):
    f = Fundamentals(symbol="AAA", name="A Co", sector="Technology",
                     market_cap=7.5e11, pe_trailing=30.0, pe_forward=24.0,
                     ps=8.0, eps_trailing=4.0, revenue_ttm=4.1e10,
                     gross_margin=0.55, operating_margin=0.17, profit_margin=0.15,
                     roe=0.22, debt_to_equity=64.0, current_ratio=2.6,
                     total_cash=1.3e10, total_debt=4.2e9, free_cashflow=8.8e9,
                     target_mean=600.0, target_low=365.0, target_high=800.0,
                     rating_mean=1.5, analysts=49.0)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def quarters(n=5, revenue=1e10, growth=1.1):
    return [Quarter(label=f"2026-{12 - 3 * i:02d}-30",
                    revenue=revenue * growth ** (n - 1 - i),
                    gross=revenue * growth ** (n - 1 - i) * 0.5,
                    operating=revenue * growth ** (n - 1 - i) * 0.17,
                    net=revenue * growth ** (n - 1 - i) * 0.15,
                    eps=1.0 * growth ** (n - 1 - i))
            for i in range(n)]


@pytest.mark.unit
class TestFormatting:
    def test_a_missing_number_is_an_em_dash_not_a_zero(self):
        assert fund.money(float("nan")) == "—"
        assert fund.pct(None) == "—"
        assert fund.ratio("") == "—"

    def test_money_scales_to_the_unit_a_reader_reads_in(self):
        assert fund.money(7.5e11) == "$7,500.00亿"
        assert fund.money(4.1e10) == "$410.00亿"
        assert fund.money(-1.2e9) == "-$12.00亿"
        assert fund.money(5_000) == "$5,000"


@pytest.mark.unit
class TestDerived:
    def test_a_loss_making_company_has_no_meaningful_pe(self):
        f = profitable(eps_trailing=-1.4)
        assert not f.profitable
        assert "市盈率没有意义" in fund.valuation_read(f)

    def test_a_company_with_neither_earnings_nor_revenue_says_so(self):
        f = Fundamentals(symbol="BBB", eps_trailing=float("nan"))
        assert "无法用倍数衡量" in fund.valuation_read(f)

    def test_yoy_needs_five_quarters_and_refuses_a_sequential_substitute(self):
        """Sequential growth is a different and much weaker claim on a seasonal book."""
        f = profitable(quarters=quarters(4))
        assert math.isnan(f.yoy("revenue"))
        f = profitable(quarters=quarters(5))
        assert f.yoy("revenue") == pytest.approx(1.1 ** 4 - 1, rel=1e-6)

    def test_the_beat_rate_counts_only_reported_quarters(self):
        f = profitable(surprises=[
            Surprise("2026-11-03", 1.93, float("nan"), float("nan")),  # unreported
            Surprise("2026-08-04", 1.61, 1.66, 0.032),
            Surprise("2026-05-05", 1.29, 1.20, -0.070),
        ])
        assert f.beat_rate() == (1, 2)

    def test_upside_is_measured_against_the_price_it_is_given(self):
        f = profitable()
        assert f.upside(500.0) == pytest.approx(0.2)
        assert math.isnan(f.upside(0.0))

    def test_the_rating_scale_is_inverted_back_to_how_a_human_reads_it(self):
        assert profitable(rating_mean=1.5).rating_text == "买入"
        assert profitable(rating_mean=4.4).rating_text == "减持"
        assert profitable(rating_mean=4.6).rating_text == "卖出"


@pytest.mark.unit
class TestReads:
    def test_three_falling_quarters_are_called_out_rather_than_averaged_away(self):
        f = profitable(quarters=quarters(5, growth=0.9))
        assert "连续环比下滑" in fund.growth_read(f)

    def test_a_negative_operating_margin_is_named_not_just_printed(self):
        f = profitable(operating_margin=-0.12)
        assert "本身还在消耗现金" in fund.quality_read(f)

    def test_negative_free_cash_flow_says_the_company_is_burning(self):
        assert "净烧钱" in fund.balance_read(profitable(free_cashflow=-3.0e8))

    def test_the_street_block_warns_that_targets_move_with_the_price(self):
        text = fund.street_read(profitable(), 500.0)
        assert "共识而不是预测" in text and "+20.0%" in text

    def test_no_coverage_is_reported_as_no_coverage(self):
        assert "没有覆盖" in fund.street_read(profitable(analysts=0.0), 500.0)

    def test_a_symbol_with_no_statements_reads_as_an_error_not_as_zeros(self):
        f = Fundamentals(symbol="ZZZZ", error="delisted")
        assert fund.read(f) == [("财报", "读不到财务数据：delisted")]
        assert "读不到财务数据" in fund.markdown_block(f)


@pytest.mark.unit
class TestMarkdown:
    def test_every_table_carries_the_vendor_caveat(self):
        text = fund.markdown_block(profitable(quarters=quarters(5)), 500.0)
        assert "SEC EDGAR" in text and "会被重述、会滞后" in text

    def test_the_quarterly_table_prints_the_margins_it_derived(self):
        text = fund.markdown_block(profitable(quarters=quarters(5)), 500.0)
        assert "| 毛利率 |" in text and "50.0%" in text

    def test_a_surprise_that_has_not_happened_yet_is_labelled_not_blanked(self):
        f = profitable(surprises=[Surprise("2026-11-03", 1.93, float("nan"), float("nan"))])
        assert "尚未公布" in fund.markdown_block(f, 500.0)


@pytest.mark.unit
class TestBook:
    def test_a_fresh_entry_is_not_refetched(self, tmp_path):
        calls = []

        def fetcher(symbol, log=None, **kw):
            calls.append(symbol)
            return Fundamentals(symbol=symbol, fetched_at=time.time(), market_cap=1e9)

        book = FundamentalsBook(tmp_path / "f.json", fetcher=fetcher)
        book.get(["AAA"])
        book.get(["AAA"])
        assert calls == ["AAA"]

    def test_a_stale_entry_is_refetched(self, tmp_path):
        calls = []

        def fetcher(symbol, log=None, **kw):
            calls.append(symbol)
            return Fundamentals(symbol=symbol, fetched_at=time.time() - 10_000,
                                market_cap=1e9)

        book = FundamentalsBook(tmp_path / "f.json", ttl_hours=1.0, fetcher=fetcher)
        book.get(["AAA"])
        book.get(["AAA"])
        assert calls == ["AAA", "AAA"]

    def test_the_cache_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "f.json"

        def fetcher(symbol, log=None, **kw):
            return Fundamentals(symbol=symbol, fetched_at=time.time(),
                                market_cap=1e9, quarters=quarters(2),
                                surprises=[Surprise("2026-08-04", 1.6, 1.7, 0.06)])

        FundamentalsBook(path, fetcher=fetcher).get(["AAA"])
        again = FundamentalsBook(path, fetcher=lambda *a, **k: pytest.fail("refetched"))
        got = again.get(["AAA"])["AAA"]
        assert got.market_cap == 1e9
        assert isinstance(got.quarters[0], Quarter)
        assert isinstance(got.surprises[0], Surprise)

    def test_an_unreadable_cache_starts_empty_instead_of_raising(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("{broken", encoding="utf-8")
        book = FundamentalsBook(path, fetcher=lambda s, log=None, **kw:
                                Fundamentals(symbol=s, fetched_at=time.time()))
        assert book.get(["AAA"])["AAA"].symbol == "AAA"

    def test_the_dividend_yield_comes_from_the_unambiguous_field(self):
        """dividendYield alone is 0.34 for Apple's 0.34% and once meant 34%.

        No magnitude test separates those, so the fraction-valued field wins
        and an ambiguous value is refused rather than guessed at.
        """
        assert fund._yield({"dividendYield": 0.34,
                            "trailingAnnualDividendYield": 0.0033}) == pytest.approx(0.0033)
        assert fund._yield({"dividendYield": 5.25}) == pytest.approx(0.0525)
        assert math.isnan(fund._yield({"dividendYield": 0.34}))
        assert math.isnan(fund._yield({}))
