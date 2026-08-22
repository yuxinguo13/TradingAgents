"""Loop cadence, coverage construction, and state persistence.

The loop itself is never run here; its decision points are exercised directly
with a stubbed broker so no browser, network, or LLM is involved.
"""

from datetime import date, datetime

import pytest

from tradingagents.live import clock
from tradingagents.live.investopedia import Account, Holding
from tradingagents.live.monitor import LiveDesk, MonitorConfig, _human


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    return LiveDesk(MonitorConfig(auto_screen=False))


def et(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=clock.ET)


@pytest.mark.unit
class TestCadence:
    def test_open_uses_the_configured_interval(self, desk):
        assert desk._sleep_seconds(clock.market_state(et(2026, 8, 20, 12))) == 120

    def test_tightens_into_the_close(self, desk):
        # The last half hour is the last chance to act on a stop today.
        assert desk._sleep_seconds(clock.market_state(et(2026, 8, 20, 15, 45))) == 60

    def test_extended_hours_poll_at_most_every_ten_minutes(self, desk):
        for t in (et(2026, 8, 20, 7), et(2026, 8, 20, 18)):
            assert desk._sleep_seconds(clock.market_state(t)) <= 600

    def test_closed_market_never_sleeps_past_the_open(self, desk):
        st = clock.market_state(et(2026, 8, 22, 12))       # Saturday
        assert desk._sleep_seconds(st) <= clock.seconds_until_open(st.now)

    def test_closed_wait_is_capped_so_premarket_news_is_not_missed(self, desk):
        st = clock.market_state(et(2026, 8, 22, 12))
        assert desk._sleep_seconds(st) <= desk.cfg.closed_interval


@pytest.mark.unit
class TestCoverage:
    def test_held_names_always_come_first(self, desk):
        desk.screen_rank = {"AAA": 1, "BBB": 2}
        acct = Account(holdings=[Holding("ZZZ", 10, 5.0)])
        cover = desk.build_coverage(acct)
        assert cover[0] == "ZZZ" and set(cover) == {"ZZZ", "AAA", "BBB"}

    def test_held_name_is_not_duplicated_by_the_screen(self, desk):
        desk.screen_rank = {"AAA": 1}
        cover = desk.build_coverage(Account(holdings=[Holding("AAA", 10, 5.0)]))
        assert cover == ["AAA"]

    def test_screen_candidates_follow_rank_order(self, desk):
        desk.screen_rank = {"CCC": 3, "AAA": 1, "BBB": 2}
        assert desk.build_coverage(Account()) == ["AAA", "BBB", "CCC"]

    def test_coverage_is_capped(self, desk):
        desk.cfg.max_coverage = 5
        desk.screen_rank = {f"S{i}": i for i in range(50)}
        assert len(desk.build_coverage(Account())) == 5

    def test_a_held_name_is_covered_even_when_it_no_longer_screens(self, desk):
        # An open position carries risk whether or not it still ranks.
        desk.cfg.max_coverage = 2
        desk.screen_rank = {"AAA": 1, "BBB": 2, "CCC": 3}
        cover = desk.build_coverage(Account(holdings=[Holding("ZZZ", 10, 5.0)]))
        assert "ZZZ" in cover


@pytest.mark.unit
class TestState:
    def test_screen_ranking_survives_a_restart(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        a = LiveDesk(MonitorConfig(auto_screen=False))
        a.screen_rank = {"AAA": 1}
        a.last_screen = date(2026, 8, 21)
        a._save_state()
        b = LiveDesk(MonitorConfig(auto_screen=False))
        assert b.screen_rank == {"AAA": 1} and b.last_screen == date(2026, 8, 21)

    def test_screen_is_skipped_when_already_run_today(self, desk, monkeypatch):
        desk.cfg.auto_screen = True
        desk.last_screen = date.today()
        called = []
        monkeypatch.setattr("tradingagents.trading.screener.screen",
                            lambda *a, **k: called.append(1))
        desk.refresh_screen("2026-08-21")
        assert not called

    def test_a_failed_screen_keeps_the_previous_ranking(self, desk, monkeypatch):
        desk.cfg.auto_screen = True
        desk.screen_rank = {"OLD": 1}
        def boom(*a, **k):
            raise RuntimeError("nasdaqtrader down")
        monkeypatch.setattr("tradingagents.trading.screener.screen", boom)
        desk.refresh_screen("2026-08-21")
        assert desk.screen_rank == {"OLD": 1}


@pytest.mark.unit
class TestCycleGuards:
    class _Broker:
        def __init__(self, acct): self._a = acct
        def account(self): return self._a
        def quote(self, s): return 100.0
        def place_order(self, *a, **k): raise AssertionError("must not trade")

    def test_no_orders_when_the_market_is_shut(self, desk):
        desk.screen_rank = {"AAA": 1}
        b = self._Broker(Account(account_value=1e5, cash=1e5, buying_power=1e5))
        st = clock.market_state(et(2026, 8, 22, 12))       # Saturday
        summary = desk.cycle(b, st)                        # _Broker explodes on trade
        assert summary["orders"] == []

    def test_account_read_failure_ends_the_cycle_cleanly(self, desk):
        class Dead:
            def account(self): raise RuntimeError("session expired")
        summary = desk.cycle(Dead(), clock.market_state(et(2026, 8, 20, 12)))
        assert summary["errors"] and "account read failed" in summary["errors"][0]


@pytest.mark.unit
class TestColdStart:
    class _Broker:
        def account(self):
            return Account(account_value=1e5, cash=1e5, buying_power=1e5)
        def quote(self, s): return 100.0
        def place_order(self, *a, **k): raise AssertionError("must not trade")

    def test_first_run_primes_instead_of_acting(self, desk, monkeypatch):
        desk.screen_rank = {"AAA": 1}
        assert desk.news.seen == {}
        seen_calls = []
        monkeypatch.setattr(desk.news, "prime",
                            lambda tickers: seen_calls.append(tickers) or 7)
        summary = desk.cycle(self._Broker(), clock.market_state(et(2026, 8, 20, 12)))
        assert summary.get("primed") == 7 and summary["orders"] == []
        assert seen_calls == [["AAA"]]

    def test_subsequent_runs_poll_normally(self, desk, monkeypatch):
        desk.screen_rank = {"AAA": 1}
        desk.news.seen = {"already": "2026-08-21T00:00:00+00:00"}
        monkeypatch.setattr(desk.news, "prime",
                            lambda t: (_ for _ in ()).throw(AssertionError("re-primed")))
        monkeypatch.setattr(desk.news, "poll", lambda *a, **k: [])
        summary = desk.cycle(self._Broker(), clock.market_state(et(2026, 8, 20, 12)))
        assert "primed" not in summary


@pytest.mark.unit
def test_human_durations():
    assert _human(45) == "45s" and _human(300) == "5m" and _human(7200) == "2.0h"
