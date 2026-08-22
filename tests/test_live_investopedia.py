"""Pure-parsing tests for the Investopedia adapter. No browser is launched."""

import pytest

from tradingagents.live.investopedia import (
    ACTIONS, BUY, MARKET, SELL, Account, Holding, OrderResult, _money,
)


@pytest.mark.unit
class TestMoneyParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("$12,345.67", 12345.67),
        ("12,345.67", 12345.67),
        ("$0.00", 0.0),
        ("(1,234.00)", -1234.0),        # parenthesised negatives
        ("-45.10", -45.1),
        ("−45.10", -45.1),              # unicode minus, as rendered by the site
        ("+1,000", 1000.0),
        ("—", 0.0),
        ("", 0.0),
        (None, 0.0),
        ("N/A", 0.0),
        ("$1,234.56 USD", 1234.56),
    ])
    def test_parses(self, raw, expected):
        assert _money(raw) == pytest.approx(expected)


@pytest.mark.unit
class TestAccount:
    def test_position_lookup_is_case_insensitive(self):
        a = Account(holdings=[Holding("NVDA", 10, 200.0)])
        assert a.position("nvda").quantity == 10

    def test_missing_position_is_none(self):
        assert Account().position("TSLA") is None

    def test_serialises_for_the_journal(self):
        a = Account(account_value=1.0, holdings=[Holding("NVDA", 10, 200.0)])
        d = a.to_dict()
        assert d["account_value"] == 1.0 and d["holdings"][0]["symbol"] == "NVDA"


@pytest.mark.unit
class TestOrderGuards:
    """Guards that reject before any browser work begins."""

    def _broker(self):
        from tradingagents.live.investopedia import InvestopediaBroker
        return InvestopediaBroker.__new__(InvestopediaBroker)

    def test_unknown_action_rejected_without_a_browser(self):
        res = self._broker().place_order("MU", "Yeet", 10)
        assert not res.ok and "unknown action" in res.message

    def test_fractional_quantity_below_one_share_rejected(self):
        # Investopedia takes whole shares only; 0.4 would silently become 0.
        res = self._broker().place_order("MU", BUY, 0.4)
        assert not res.ok and "rounds to zero" in res.message

    def test_action_vocabulary_is_the_site_wording(self):
        assert ACTIONS == ("Buy", "Sell", "Sell Short", "Buy to Cover")
