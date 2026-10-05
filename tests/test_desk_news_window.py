"""A pack describes a window of news, not what is new since the last poll.

On 2026-10-04 the report pack for Monday was rebuilt after a container
restart, and its news columns came up blank: Friday's run had already "seen"
Friday's headlines, and the 48-hour cut dropped Friday afternoon. Prices stop
at the data session's close; the news must run from the close before it up to
the moment the pack is built."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.test_desk import news
from tradingagents.desk.market import Market, news_since, window_hours
from tradingagents.live import clock, newsfeed, policy


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    yield tmp_path


class RecordingNews:
    def __init__(self, items):
        self.items, self.calls = list(items), []

    def poll(self, tickers, macro=True, pause=0.0, **kw):
        self.calls.append(kw)
        return list(self.items)


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def poll(self, categories=None, pause=0.0, **kw):
        self.calls.append(kw)
        return []


@pytest.mark.unit
class TestNewsWindow:
    def test_the_window_opens_at_the_close_before_the_data_session(self):
        assert news_since(date(2026, 10, 2)) == datetime(2026, 10, 1, 16, 0, tzinfo=clock.ET)
        # a Monday's data session reaches back over the weekend to Friday's close
        assert news_since(date(2026, 10, 5)) == datetime(2026, 10, 2, 16, 0, tzinfo=clock.ET)

    def test_the_window_is_never_shorter_than_a_day(self):
        now = datetime(2026, 10, 5, 20, 30, tzinfo=timezone.utc)
        assert window_hours(now - timedelta(hours=2), now) == 24.0
        assert window_hours(now - timedelta(hours=80), now) == pytest.approx(80.0)

    def test_a_pack_gets_seen_headlines_and_the_whole_weekend(self):
        weekend = news("AAA", "AAA wins a contract on Friday", hours_ago=70)
        ancient = news("AAA", "AAA old story", hours_ago=200)
        feed = RecordingNews([weekend, ancient])
        m = Market(task="test", news=feed)
        since = datetime.now(timezone.utc) - timedelta(hours=80)
        by, _ = m.headlines(["AAA"], macro=False, since=since)
        assert [n.title for n in by["AAA"]] == ["AAA wins a contract on Friday"]
        assert feed.calls == [{"include_seen": True}]

    def test_the_loops_still_get_only_what_is_new(self):
        feed = RecordingNews([news("AAA", "AAA old-ish", hours_ago=70)])
        by, _ = Market(task="test", news=feed).headlines(["AAA"], macro=False)
        assert by["AAA"] == [] and feed.calls == [{}]

    def test_policy_takes_the_same_window(self):
        pol = RecordingPolicy()
        since = datetime.now(timezone.utc) - timedelta(hours=72)
        Market(task="test", policy=pol).policy(since=since)
        Market(task="test", policy=pol).policy()
        assert pol.calls[0]["include_seen"] is True
        assert pol.calls[0]["max_age_hours"] == pytest.approx(72.0, abs=0.1)
        assert pol.calls[1] == {}


@pytest.mark.unit
class TestMonitorsReplaySeenItems:
    def test_news_monitor_returns_seen_items_on_request(self, tmp_path, monkeypatch):
        pub = datetime.now(timezone.utc).isoformat()
        raw = [{"title": "ACME beats estimates", "link": "l", "published": pub, "source": "s"}]
        monkeypatch.setattr(newsfeed, "fetch_rss", lambda url, timeout=20: list(raw))
        mon = newsfeed.NewsMonitor(state_path=tmp_path / "seen.json")
        assert len(mon.poll_ticker("ACME")) == 1
        assert mon.poll_ticker("ACME") == []                       # a loop: nothing new
        again = mon.poll_ticker("ACME", include_seen=True)          # a pack: the window
        assert [i.title for i in again] == ["ACME beats estimates"]  # once, not once per feed

    def test_policy_monitor_returns_seen_events_on_request(self, tmp_path, monkeypatch):
        raw = [{"title": "Tariff headline", "link": "l", "published": "", "source": "s"}]
        monkeypatch.setattr(policy, "fetch_rss", lambda url, timeout=20: list(raw))
        ev = SimpleNamespace(age_hours=lambda now=None: 60.0, severity=5)
        monkeypatch.setattr(policy, "classify", lambda *a, **k: ev)
        mon = policy.PolicyMonitor(state_path=tmp_path / "p.json")
        assert mon.poll_category("trade") == []                     # 60h > the 48h default
        assert mon.poll_category("trade", include_seen=True, max_age_hours=72) == [ev]
        assert mon.poll_category("trade") == []
