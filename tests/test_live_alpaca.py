"""Alpaca adapter. Fully mocked — no network, no credentials required.

These cover the two things that are genuinely easy to get wrong at this
boundary: Alpaca returning every number as a string, and its two-value
OrderSide being unable to express the desk's four-verb vocabulary on its own.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.live import alpaca as A
from tradingagents.live.broker import (
    BUY, COVER, LIMIT, MARKET, SELL, SHORT, STOP, Account,
)


@pytest.fixture
def broker(monkeypatch):
    """An AlpacaBroker with both SDK clients replaced by mocks."""
    b = A.AlpacaBroker.__new__(A.AlpacaBroker)
    b.paper = True
    b.trading = MagicMock()
    b.data = MagicMock()
    return b


def acct_obj(**kw):
    """Alpaca returns every numeric field as a STRING. Mirror that exactly."""
    base = dict(equity="100000.00", portfolio_value="100000.00", cash="40000.00",
                buying_power="80000.00", account_blocked=False, trading_blocked=False)
    base.update(kw)
    return SimpleNamespace(**base)


def pos_obj(symbol="NVDA", qty="100", avg="200.00", last="215.00",
            mv="21500.00", pl="1500.00", short=False):
    from alpaca.trading.enums import PositionSide
    return SimpleNamespace(
        symbol=symbol, qty=qty, avg_entry_price=avg, current_price=last,
        market_value=mv, unrealized_pl=pl,
        side=PositionSide.SHORT if short else PositionSide.LONG,
    )


# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStringCoercion:
    @pytest.mark.parametrize("raw,expected", [
        ("12345.67", 12345.67), ("0", 0.0), ("-45.1", -45.1),
        (None, 0.0), ("", 0.0), ("n/a", 0.0), (12.5, 12.5),
    ])
    def test_f_never_raises(self, raw, expected):
        assert A._f(raw) == pytest.approx(expected)

    def test_account_numbers_become_floats(self, broker):
        broker.trading.get_account.return_value = acct_obj()
        broker.trading.get_all_positions.return_value = []
        a = broker.account()
        assert isinstance(a.account_value, float) and a.account_value == 100000.0
        assert isinstance(a.cash, float) and a.cash == 40000.0

    def test_falls_back_to_portfolio_value_when_equity_absent(self, broker):
        broker.trading.get_account.return_value = acct_obj(equity=None)
        broker.trading.get_all_positions.return_value = []
        assert broker.account().account_value == 100000.0


@pytest.mark.unit
class TestPositionMapping:
    def test_long_position(self, broker):
        broker.trading.get_account.return_value = acct_obj()
        broker.trading.get_all_positions.return_value = [pos_obj()]
        h = broker.account().position("NVDA")
        assert h.quantity == 100 and h.side == "long" and h.avg_cost == 200.0

    def test_short_position_is_reported_positive_with_side(self, broker):
        # Alpaca signs qty and market_value negative for shorts. The desk keeps
        # size positive so a short does not subtract from gross exposure.
        broker.trading.get_account.return_value = acct_obj()
        broker.trading.get_all_positions.return_value = [
            pos_obj(qty="-50", mv="-10750.00", short=True)]
        h = broker.account().position("NVDA")
        assert h.quantity == 50 and h.market_value == 10750.0 and h.side == "short"

    def test_position_lookup_is_case_insensitive(self, broker):
        broker.trading.get_account.return_value = acct_obj()
        broker.trading.get_all_positions.return_value = [pos_obj()]
        assert broker.account().position("nvda") is not None


@pytest.mark.unit
class TestFourVerbVocabulary:
    """Alpaca's OrderSide is only BUY/SELL.

    Mapping "Sell Short" to SELL and stopping there means shorting a name you
    already hold long would SELL THE LONG instead — the order reports success
    while doing the opposite of what was asked.
    """

    def test_every_action_is_mapped(self):
        from tradingagents.live.broker import ACTIONS
        assert set(A._INTENT) == set(ACTIONS)

    @pytest.mark.parametrize("action,side,intent", [
        (BUY,   "BUY",  "BUY_TO_OPEN"),
        (SELL,  "SELL", "SELL_TO_CLOSE"),
        (SHORT, "SELL", "SELL_TO_OPEN"),
        (COVER, "BUY",  "BUY_TO_CLOSE"),
    ])
    def test_intent_disambiguates_the_side(self, action, side, intent):
        assert A._INTENT[action] == (side, intent)

    def test_sell_and_short_share_a_side_but_differ_in_intent(self):
        assert A._INTENT[SELL][0] == A._INTENT[SHORT][0]
        assert A._INTENT[SELL][1] != A._INTENT[SHORT][1]

    def test_buy_and_cover_share_a_side_but_differ_in_intent(self):
        assert A._INTENT[BUY][0] == A._INTENT[COVER][0]
        assert A._INTENT[BUY][1] != A._INTENT[COVER][1]

    def test_request_carries_both_side_and_intent(self, broker):
        # The SDK validator rejects an intent without a side, so both must ship.
        req = broker._build_request("NVDA", SHORT, 10, MARKET, None)
        assert req.side.value == "sell" and req.position_intent.value == "sell_to_open"


@pytest.mark.unit
class TestOrderConstruction:
    def test_market_order(self, broker):
        req = broker._build_request("NVDA", BUY, 10, MARKET, None)
        assert req.qty == 10 and req.time_in_force.value == "day"

    def test_limit_order_carries_limit_price(self, broker):
        req = broker._build_request("NVDA", BUY, 10, LIMIT, 200.0)
        assert float(req.limit_price) == 200.0

    def test_stop_order_uses_stop_price_not_limit_price(self, broker):
        # place_order funnels both through one `limit_price` parameter; the SDK
        # field differs, and sending the wrong one silently drops the trigger.
        req = broker._build_request("NVDA", SELL, 10, STOP, 190.0)
        assert float(req.stop_price) == 190.0

    def test_day_not_gtc(self, broker):
        # A thesis formed this morning should not fill three days later.
        assert broker._build_request("NVDA", BUY, 1, MARKET, None).time_in_force.value == "day"


@pytest.mark.unit
class TestPlaceOrderGuards:
    """Guards that reject before anything reaches the network."""

    def test_unknown_action(self, broker):
        r = broker.place_order("NVDA", "Yeet", 10)
        assert not r.ok and "unknown action" in r.message
        broker.trading.submit_order.assert_not_called()

    def test_non_positive_quantity(self, broker):
        assert not broker.place_order("NVDA", BUY, 0).ok
        broker.trading.submit_order.assert_not_called()

    def test_limit_without_price(self, broker):
        r = broker.place_order("NVDA", BUY, 10, LIMIT, None)
        assert not r.ok and "needs a price" in r.message

    def test_dry_run_never_submits(self, broker):
        r = broker.place_order("NVDA", BUY, 10, dry_run=True)
        assert r.ok and "DRY RUN" in r.message
        broker.trading.submit_order.assert_not_called()


@pytest.mark.unit
class TestOrderResults:
    def _submit(self, broker, status="accepted", filled="0", avg=None, qty="10"):
        broker.trading.submit_order.return_value = SimpleNamespace(
            id="abc-123", status=SimpleNamespace(value=status),
            filled_qty=filled, filled_avg_price=avg, qty=qty)
        return broker.place_order("NVDA", BUY, 10)

    def test_accepted_but_unfilled_is_success(self, broker):
        # A working order is not a failure; the fill is a separate event.
        r = self._submit(broker, "accepted")
        assert r.ok and r.broker_order_id == "abc-123" and not r.is_filled

    def test_filled_order_reports_price_and_quantity(self, broker):
        r = self._submit(broker, "filled", filled="10", avg="214.72")
        assert r.ok and r.is_filled and r.filled_avg_price == 214.72

    @pytest.mark.parametrize("status", ["rejected", "canceled", "expired"])
    def test_terminal_negative_statuses_are_failures(self, broker, status):
        assert not self._submit(broker, status).ok

    def test_api_error_is_reported_not_raised(self, broker):
        # A rejection must not abort the sweep over the other symbols.
        broker.trading.submit_order.side_effect = RuntimeError("insufficient buying power")
        r = broker.place_order("NVDA", BUY, 10)
        assert not r.ok and "insufficient buying power" in r.message


@pytest.mark.unit
class TestQuoteFallback:
    """The free data plan is IEX-only, a few percent of US volume.

    A thin name can have no recent IEX print at all, and a 0.0 makes the risk
    gate reject with "no usable price" — so the chain matters.
    """

    def test_prefers_the_last_trade(self, broker):
        broker.data.get_stock_latest_trade.return_value = {"NVDA": SimpleNamespace(price="214.72")}
        assert broker.quote("NVDA") == 214.72

    def test_falls_back_to_the_quote_midpoint(self, broker):
        broker.data.get_stock_latest_trade.side_effect = RuntimeError("no IEX print")
        broker.data.get_stock_latest_quote.return_value = {
            "NVDA": SimpleNamespace(bid_price="214.00", ask_price="215.00")}
        assert broker.quote("NVDA") == 214.5

    def test_falls_back_to_the_daily_bar(self, broker, monkeypatch):
        import pandas as pd
        broker.data.get_stock_latest_trade.side_effect = RuntimeError("down")
        broker.data.get_stock_latest_quote.side_effect = RuntimeError("down")
        monkeypatch.setattr("tradingagents.dataflows.stockstats_utils.load_ohlcv",
                            lambda *a, **k: pd.DataFrame({"Close": [212.5]}))
        assert broker.quote("NVDA") == 212.5

    def test_returns_zero_rather_than_raising_when_all_sources_fail(self, broker, monkeypatch):
        broker.data.get_stock_latest_trade.side_effect = RuntimeError("down")
        broker.data.get_stock_latest_quote.side_effect = RuntimeError("down")
        monkeypatch.setattr("tradingagents.dataflows.stockstats_utils.load_ohlcv",
                            lambda *a, **k: None)
        assert broker.quote("NVDA") == 0.0


@pytest.mark.unit
class TestReachability:
    def test_blocked_account_is_not_usable(self, broker):
        broker.trading.get_account.return_value = acct_obj(trading_blocked=True)
        assert broker.is_logged_in() is False

    def test_network_failure_is_not_usable(self, broker):
        broker.trading.get_account.side_effect = RuntimeError("connection refused")
        assert broker.is_logged_in() is False

    def test_healthy_account_is_usable(self, broker):
        broker.trading.get_account.return_value = acct_obj()
        assert broker.is_logged_in() is True

    def test_open_orders_degrade_to_empty(self, broker):
        broker.trading.get_orders.side_effect = RuntimeError("down")
        assert broker.open_orders() == []


@pytest.mark.unit
class TestCredentialsAndSelection:
    def test_missing_credentials_names_the_fix(self, monkeypatch):
        for v in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                  "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
            monkeypatch.delenv(v, raising=False)
        with pytest.raises(A.MissingCredentials) as e:
            A.credentials()
        assert "signup" in str(e.value)

    def test_accepts_the_sdk_native_variable_names(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.setenv("APCA_API_KEY_ID", "PKTEST")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
        assert A.credentials() == ("PKTEST", "secret")

    def test_alpaca_is_the_default_venue(self, monkeypatch):
        from tradingagents.live.broker import configured_venue
        monkeypatch.delenv("TRADINGAGENTS_BROKER", raising=False)
        assert configured_venue() == "alpaca"

    def test_venue_is_overridable(self, monkeypatch):
        from tradingagents.live.broker import configured_venue
        monkeypatch.setenv("TRADINGAGENTS_BROKER", "investopedia")
        assert configured_venue() == "investopedia"

    def test_unknown_venue_is_rejected(self, monkeypatch):
        from tradingagents.live.broker import open_broker
        with pytest.raises(ValueError):
            open_broker("robinhood")

    def test_adapter_satisfies_the_broker_protocol(self):
        from tradingagents.live.broker import Broker
        from tradingagents.live.investopedia import InvestopediaBroker
        for cls in (A.AlpacaBroker, InvestopediaBroker):
            for m in ("is_logged_in", "account", "quote", "place_order"):
                assert callable(getattr(cls, m, None)), f"{cls.__name__} missing {m}"
