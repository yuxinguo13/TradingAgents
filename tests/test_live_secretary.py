"""The risk gate. Every limit here is one an LLM must not be able to argue past."""

import json
from datetime import datetime, timedelta

import pytest

from tradingagents.live.investopedia import BUY, SELL, Account, Holding
from tradingagents.live.secretary import (
    Order, RiskLimits, Secretary, TradeLedger,
)


@pytest.fixture
def ledger(tmp_path):
    return TradeLedger(path=tmp_path / "ledger.json")


@pytest.fixture
def sec(ledger, monkeypatch, tmp_path):
    # Point the kill switch at the temp dir so a real one on the host machine
    # cannot make these tests pass or fail spuriously.
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    return Secretary(limits=RiskLimits(), ledger=ledger)


@pytest.fixture
def account():
    return Account(
        account_value=100_000.0, cash=40_000.0, buying_power=40_000.0,
        holdings=[Holding("NVDA", 100, 200.0, 215.0, 21_500.0, 1_500.0)],
    )


# ---------------------------------------------------------------------------
# format validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseOrder:
    def test_rejects_non_json_with_retryable_message(self, sec):
        v = sec.parse_order("I think you should buy some Nvidia.")
        assert not v.ok and v.retryable

    def test_extracts_json_from_prose_and_fences(self, sec):
        v = sec.parse_order('Sure!\n```json\n{"action":"Buy","symbol":"MU",'
                            '"quantity":10}\n```\nHope that helps.')
        assert v.ok and v.order.symbol == "MU" and v.order.quantity == 10

    def test_hold_yields_no_order(self, sec):
        v = sec.parse_order('{"action":"Hold","symbol":"MU","quantity":0}')
        assert v.ok and v.order is None

    def test_string_quantity_is_coerced(self, sec):
        assert sec.parse_order('{"action":"Buy","symbol":"MU","quantity":"25"}'
                               ).order.quantity == 25

    def test_fractional_quantity_truncates_to_whole_shares(self, sec):
        assert sec.parse_order('{"action":"Buy","symbol":"MU","quantity":7.9}'
                               ).order.quantity == 7

    def test_zero_quantity_rejected(self, sec):
        v = sec.parse_order('{"action":"Buy","symbol":"MU","quantity":0}')
        assert not v.ok and v.retryable

    def test_bad_symbol_rejected(self, sec):
        v = sec.parse_order('{"action":"Buy","symbol":"not a ticker","quantity":5}')
        assert not v.ok and v.retryable

    def test_unknown_action_rejected(self, sec):
        v = sec.parse_order('{"action":"YOLO","symbol":"MU","quantity":5}')
        assert not v.ok and v.retryable

    def test_limit_order_requires_price(self, sec):
        v = sec.parse_order('{"action":"Buy","symbol":"MU","quantity":5,'
                            '"order_type":"Limit"}')
        assert not v.ok and "limit_price" in v.reason

    def test_confidence_clamped_to_unit_interval(self, sec):
        o = sec.parse_order('{"action":"Buy","symbol":"MU","quantity":5,'
                            '"confidence":9}').order
        assert o.confidence == 1.0


# ---------------------------------------------------------------------------
# risk limits
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRiskGate:
    def test_market_closed_blocks_everything(self, sec, account):
        v = sec.check(Order("MU", "Buy", 10), account, 120.0, market_open=False)
        assert not v.ok and "closed" in v.reason

    def test_kill_switch_blocks_everything(self, sec, account, tmp_path):
        (tmp_path / "STOP").write_text("stop")
        v = sec.check(Order("MU", "Buy", 10), account, 120.0)
        assert not v.ok and "kill switch" in v.reason

    def test_oversized_buy_is_resized_not_rejected(self, sec, account):
        v = sec.check(Order("MU", "Buy", 9999), account, 120.0)
        # 8% new-position cap on $100k = $8,000 → 66 shares at $120.
        assert v.ok and v.order.quantity == 66

    def test_position_at_cap_is_blocked(self, sec, account):
        # NVDA already sits at 21.5% of the account; the cap is 12%.
        v = sec.check(Order("NVDA", "Buy", 10), account, 215.0)
        assert not v.ok and "cap" in v.reason

    def test_sell_clamped_to_shares_actually_held(self, sec, account):
        v = sec.check(Order("NVDA", "Sell", 5_000), account, 215.0)
        assert v.ok and v.order.quantity == 100

    def test_sell_without_position_rejected(self, sec, account):
        v = sec.check(Order("TSLA", "Sell", 5), account, 400.0)
        assert not v.ok

    def test_shorting_disabled_by_default(self, sec, account):
        v = sec.check(Order("MU", "Sell Short", 10), account, 120.0)
        assert not v.ok and "short" in v.reason.lower()

    def test_penny_stock_blocked_by_price_floor(self, sec, account):
        v = sec.check(Order("XYZ", "Buy", 100), account, 1.50)
        assert not v.ok and "floor" in v.reason

    def test_limit_far_from_last_rejected(self, sec, account):
        v = sec.check(Order("MU", "Buy", 10, order_type="Limit", limit_price=200.0),
                      account, 120.0)
        assert not v.ok and "limit" in v.reason

    def test_cash_constrains_size_below_position_cap(self, sec, account):
        poor = Account(account_value=100_000.0, cash=1_000.0, buying_power=1_000.0)
        v = sec.check(Order("MU", "Buy", 500), poor, 120.0)
        assert v.ok and v.order.quantity == 8      # $1,000 / $120

    def test_zero_equity_account_blocked(self, sec):
        v = sec.check(Order("MU", "Buy", 10), Account(), 120.0)
        assert not v.ok


@pytest.mark.unit
class TestDailyBudgets:
    def test_trade_count_budget_exhausts(self, sec, account):
        for i in range(sec.limits.max_trades_per_day):
            sec.ledger.record(Order("AAA", "Buy", 1), 10.0, ok=True)
        v = sec.check(Order("MU", "Buy", 10), account, 120.0)
        assert not v.ok and "trade budget" in v.reason

    def test_turnover_budget_exhausts(self, sec, account):
        # One huge fill blows the 35%-of-equity turnover budget on its own.
        sec.ledger.record(Order("AAA", "Buy", 1_000), 40.0, ok=True)
        v = sec.check(Order("MU", "Buy", 10), account, 120.0)
        assert not v.ok and "turnover" in v.reason

    def test_failed_orders_do_not_consume_budget(self, sec, account):
        for _ in range(20):
            sec.ledger.record(Order("AAA", "Buy", 1), 10.0, ok=False, message="rejected")
        assert sec.ledger.trades_today() == 0
        assert sec.check(Order("MU", "Buy", 10), account, 120.0).ok

    def test_symbol_cooldown_blocks_immediate_rechurn(self, sec, account):
        sec.ledger.record(Order("MU", "Buy", 10), 120.0, ok=True)
        v = sec.check(Order("MU", "Buy", 10), account, 120.0)
        assert not v.ok and "cooldown" in v.reason

    def test_cooldown_expires(self, sec, account):
        stale = (datetime.now() - timedelta(hours=3)).isoformat()
        sec.ledger.entries.append({"at": stale, "symbol": "MU", "action": "Buy",
                                   "quantity": 10, "price": 120.0,
                                   "notional": 1200.0, "ok": True})
        assert sec.check(Order("MU", "Buy", 10), account, 120.0).ok


@pytest.mark.unit
class TestLedgerPersistence:
    def test_ledger_survives_a_restart(self, tmp_path):
        p = tmp_path / "ledger.json"
        a = TradeLedger(path=p)
        a.record(Order("MU", "Buy", 10), 120.0, ok=True)
        assert TradeLedger(path=p).trades_today() == 1

    def test_env_overrides_limits(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_RISK_MAX_TRADES_PER_DAY", "3")
        monkeypatch.setenv("TRADINGAGENTS_RISK_ALLOW_SHORT", "true")
        lim = RiskLimits.from_env()
        assert lim.max_trades_per_day == 3 and lim.allow_short is True


@pytest.mark.unit
class TestBudgetsAreScopedToTheVenue:
    """One ledger, more than one account behind it.

    ``max_turnover_per_day`` is a fraction of *account value*, and each venue
    is a separate account with its own. Summed across venues and divided by one
    of them the number means nothing — and it locks a venue out over trades
    that never touched it. Found live: five cleanup sells on the local paper
    book spent $38,258 and shut the Alpaca bridge for the day against a $35,000
    limit it had not used a cent of.

    The churn cooldown is the same story one symbol at a time: selling NVDA on
    paper must not stop the desk buying NVDA at the broker.
    """

    def secretary(self, ledger, tmp_path, monkeypatch, **limits):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        return Secretary(limits=RiskLimits(**limits), ledger=ledger, venue="alpaca")

    def test_another_venues_turnover_does_not_spend_this_ones(self, ledger):
        ledger.record(Order("NVDA", SELL, 400), 100.0, True, venue="paper")
        assert ledger.turnover_today("alpaca") == 0.0
        assert ledger.turnover_today("paper") == 40_000.0

    def test_another_venues_trade_count_does_not_spend_this_ones(
            self, ledger, account, tmp_path, monkeypatch):
        for i in range(20):
            ledger.record(Order(f"A{i}", BUY, 1), 10.0, True, venue="paper")
        sec = self.secretary(ledger, tmp_path, monkeypatch, max_trades_per_day=2)
        v = sec.check(Order("AAA", BUY, 10), account, 100.0)
        assert v.ok, v.reason

    def test_another_venues_turnover_does_not_refuse_this_ones_order(
            self, ledger, account, tmp_path, monkeypatch):
        ledger.record(Order("NVDA", SELL, 400), 100.0, True, venue="paper")
        sec = self.secretary(ledger, tmp_path, monkeypatch, max_turnover_per_day=0.05)
        v = sec.check(Order("AAA", BUY, 10), account, 100.0)
        assert v.ok, v.reason

    def test_a_cooldown_on_one_venue_does_not_bind_the_other(
            self, ledger, account, tmp_path, monkeypatch):
        ledger.record(Order("AAA", BUY, 10), 100.0, True, venue="paper")
        sec = self.secretary(ledger, tmp_path, monkeypatch,
                             symbol_cooldown_minutes=45)
        v = sec.check(Order("AAA", BUY, 10), account, 100.0)
        assert v.ok, v.reason

    def test_the_same_venue_still_binds(self, ledger, account, tmp_path, monkeypatch):
        ledger.record(Order("AAA", BUY, 10), 100.0, True, venue="alpaca")
        sec = self.secretary(ledger, tmp_path, monkeypatch,
                             symbol_cooldown_minutes=45)
        v = sec.check(Order("AAA", BUY, 10), account, 100.0)
        assert not v.ok and "cooldown" in v.reason

    def test_a_row_with_no_venue_still_counts_everywhere(self, ledger):
        """It predates the field and cannot be attributed. Over-counting
        refuses trades that were allowed; under-counting allows trades that
        were not, and only one of those is recoverable."""
        ledger.entries.append({"at": datetime.now().isoformat(), "symbol": "NVDA",
                               "action": SELL, "quantity": 400, "price": 100.0,
                               "notional": 40_000.0, "ok": True, "message": "",
                               "source": "cleanup"})
        assert ledger.turnover_today("alpaca") == 40_000.0
        assert ledger.turnover_today("paper") == 40_000.0

    def test_a_caller_that_names_no_venue_counts_everything(self, ledger):
        ledger.record(Order("NVDA", SELL, 400), 100.0, True, venue="paper")
        ledger.record(Order("MSFT", SELL, 100), 100.0, True, venue="alpaca")
        assert ledger.turnover_today() == 50_000.0
