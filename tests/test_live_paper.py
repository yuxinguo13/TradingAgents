"""Local paper broker. Real book semantics, mocked prices — no network."""

import pytest

from tradingagents.live import paper as P
from tradingagents.live.broker import BUY, COVER, LIMIT, MARKET, SELL, SHORT


@pytest.fixture
def broker(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    b = P.LocalPaperBroker(starting_cash=100_000.0, path=tmp_path / "book.json")
    b._quotes = {"NVDA": 200.0, "MSFT": 400.0, "PENNY": 0.50}
    return b


@pytest.mark.unit
class TestBookkeeping:
    def test_starts_flat_with_the_full_balance(self, broker):
        a = broker.account()
        assert a.account_value == 100_000.0 and a.cash == 100_000.0 and not a.holdings

    def test_buy_moves_cash_into_a_position(self, broker):
        r = broker.place_order("NVDA", BUY, 10)
        assert r.ok and r.filled_quantity == 10 and r.filled_avg_price == 200.0
        a = broker.account()
        assert a.cash == 98_000.0
        assert a.position("NVDA").quantity == 10

    def test_equity_is_conserved_by_a_fill(self, broker):
        # Buying at the mark cannot create or destroy value.
        broker.place_order("NVDA", BUY, 10)
        assert broker.account().account_value == pytest.approx(100_000.0)

    def test_equity_tracks_price(self, broker):
        broker.place_order("NVDA", BUY, 10)
        broker._quotes["NVDA"] = 220.0
        a = broker.account()
        assert a.account_value == pytest.approx(100_200.0)
        assert a.position("NVDA").unrealized == pytest.approx(200.0)

    def test_sell_realises_into_cash(self, broker):
        broker.place_order("NVDA", BUY, 10)
        broker._quotes["NVDA"] = 250.0
        broker.place_order("NVDA", SELL, 10)
        a = broker.account()
        assert a.cash == pytest.approx(100_500.0) and not a.holdings

    def test_average_cost_across_two_buys(self, broker):
        broker.place_order("NVDA", BUY, 10)
        broker._quotes["NVDA"] = 300.0
        broker.place_order("NVDA", BUY, 10)
        assert broker.account().position("NVDA").avg_cost == pytest.approx(250.0)

    def test_book_survives_a_restart(self, broker, tmp_path):
        broker.place_order("NVDA", BUY, 10)
        again = P.LocalPaperBroker(path=tmp_path / "book.json")
        again._quotes = {"NVDA": 200.0}
        assert again.account().position("NVDA").quantity == 10


@pytest.mark.unit
class TestGuards:
    def test_cannot_spend_cash_it_does_not_have(self, broker):
        # Trimmed to what the cash buys, not rejected: the gate approved the
        # direction and a partial move toward it is still correct.
        r = broker.place_order("NVDA", BUY, 10_000)
        assert r.ok and r.filled_quantity == 500       # 100k / 200
        assert broker.account().cash == pytest.approx(0.0)

    def test_cannot_sell_what_is_not_held(self, broker):
        r = broker.place_order("NVDA", SELL, 10)
        assert not r.ok and "no NVDA position" in r.message

    def test_sell_is_clamped_to_the_position(self, broker):
        broker.place_order("NVDA", BUY, 10)
        r = broker.place_order("NVDA", SELL, 999)
        assert r.ok and r.filled_quantity == 10

    def test_shorting_is_refused_not_faked(self, broker):
        # The underlying Portfolio cannot hold a negative position; pretending
        # otherwise would invent a fill that the book cannot represent.
        for action in (SHORT, COVER):
            r = broker.place_order("NVDA", action, 10)
            assert not r.ok and "long-only" in r.message

    def test_unpriceable_symbol_is_rejected(self, broker):
        broker._quotes["GHOST"] = 0.0
        assert not broker.place_order("GHOST", BUY, 10).ok

    def test_fractional_quantity_below_one_share(self, broker):
        assert not broker.place_order("NVDA", BUY, 0.4).ok

    def test_unknown_action(self, broker):
        assert not broker.place_order("NVDA", "Yeet", 1).ok

    def test_dry_run_does_not_move_the_book(self, broker):
        r = broker.place_order("NVDA", BUY, 10, dry_run=True)
        assert r.ok and "DRY RUN" in r.message
        assert broker.account().cash == 100_000.0


@pytest.mark.unit
class TestLimitOrders:
    def test_marketable_buy_limit_fills_at_the_limit(self, broker):
        r = broker.place_order("NVDA", BUY, 10, LIMIT, 205.0)
        assert r.ok and r.filled_avg_price == 205.0

    def test_unmarketable_buy_limit_is_refused(self, broker):
        # No resting-order book here; claiming a fill would be a lie.
        r = broker.place_order("NVDA", BUY, 10, LIMIT, 190.0)
        assert not r.ok and "no resting orders" in r.message

    def test_unmarketable_sell_limit_is_refused(self, broker):
        broker.place_order("NVDA", BUY, 10)
        r = broker.place_order("NVDA", SELL, 10, LIMIT, 250.0)
        assert not r.ok and "no resting orders" in r.message


@pytest.mark.unit
class TestReporting:
    def test_pnl_is_zero_at_the_mark(self, broker):
        broker.place_order("NVDA", BUY, 10)
        assert broker.pnl()["pnl"] == pytest.approx(0.0)

    def test_pnl_follows_price(self, broker):
        broker.place_order("NVDA", BUY, 100)
        broker._quotes["NVDA"] = 210.0
        p = broker.pnl()
        assert p["pnl"] == pytest.approx(1_000.0) and p["pnl_pct"] == pytest.approx(0.01)

    def test_no_resting_orders_to_report(self, broker):
        assert broker.open_orders() == []

    def test_always_reachable(self, broker):
        assert broker.is_logged_in() is True


@pytest.mark.unit
class TestVenueSelection:
    def test_paper_is_selectable(self, monkeypatch, tmp_path):
        from tradingagents.live.broker import configured_venue, open_broker
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        monkeypatch.setenv("TRADINGAGENTS_BROKER", "paper")
        assert configured_venue() == "paper"
        with open_broker() as b:
            assert isinstance(b, P.LocalPaperBroker)

    def test_satisfies_the_broker_contract(self):
        for m in ("is_logged_in", "account", "quote", "place_order"):
            assert callable(getattr(P.LocalPaperBroker, m, None))
