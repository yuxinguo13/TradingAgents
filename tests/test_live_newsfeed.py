"""News novelty detection and materiality triage. No network in these tests."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.live import newsfeed
from tradingagents.live.newsfeed import NewsItem, NewsMonitor, _fingerprint, score


@pytest.mark.unit
class TestMateriality:
    @pytest.mark.parametrize("headline,floor", [
        ("Trading halted in ACME pending news", 12),
        ("BigCo to acquire ACME for $4B in cash", 10),
        ("FDA approves ACME's phase 3 therapy", 9),
        ("ACME cuts outlook for full year", 9),
        ("ACME beats Q3 earnings estimates", 8),
        ("SEC opens investigation into ACME", 7),
        ("ACME CEO steps down", 7),
        ("Analyst raises ACME price target", 5),
    ])
    def test_material_headlines_score(self, headline, floor):
        assert score(headline)[0] >= floor

    def test_filler_scores_zero(self):
        assert score("3 stocks to watch this week")[0] == 0

    def test_directional_lean(self):
        assert score("ACME beats estimates, shares surge")[1] == "bullish"
        assert score("ACME misses estimates, shares plunge")[1] == "bearish"
        assert score("ACME schedules its annual meeting")[1] == "neutral"

    def test_mixed_signals_stay_neutral(self):
        # "beats ... but downgrade" should not be asserted as bullish.
        assert score("ACME beats estimates but analyst issues downgrade")[1] == "neutral"


@pytest.mark.unit
class TestFingerprint:
    def test_publisher_suffix_is_ignored(self):
        # The same wire story syndicated by two outlets is one story.
        a = _fingerprint("Micron jumps on $10B lab plan - Yahoo Finance", "MU")
        b = _fingerprint("Micron jumps on $10B lab plan - Reuters", "MU")
        assert a == b

    def test_punctuation_and_case_are_ignored(self):
        assert (_fingerprint("ACME Beats Estimates!", "ACME")
                == _fingerprint("acme beats estimates", "ACME"))

    def test_different_stories_differ(self):
        assert (_fingerprint("ACME beats estimates", "ACME")
                != _fingerprint("ACME misses estimates", "ACME"))

    def test_same_headline_under_different_tickers_differs(self):
        assert _fingerprint("Chip stocks rally", "MU") != _fingerprint("Chip stocks rally", "NVDA")


@pytest.mark.unit
class TestNovelty:
    def test_second_poll_returns_nothing_new(self, tmp_path, monkeypatch):
        feed = [{"title": "ACME beats Q3 earnings estimates", "link": "u1",
                 "published": "2026-08-21T12:00:00+00:00", "source": "wire"}]
        monkeypatch.setattr(newsfeed, "fetch_rss", lambda url, timeout=20: feed)
        m = NewsMonitor(state_path=tmp_path / "seen.json")
        assert len(m.poll(["ACME"], macro=False)) == 1
        assert m.poll(["ACME"], macro=False) == []

    def test_novelty_survives_a_restart(self, tmp_path, monkeypatch):
        feed = [{"title": "ACME beats Q3 earnings estimates", "link": "u1",
                 "published": "2026-08-21T12:00:00+00:00", "source": "wire"}]
        monkeypatch.setattr(newsfeed, "fetch_rss", lambda url, timeout=20: feed)
        p = tmp_path / "seen.json"
        NewsMonitor(state_path=p).poll(["ACME"], macro=False)
        assert NewsMonitor(state_path=p).poll(["ACME"], macro=False) == []

    def test_duplicate_across_sources_yields_one_item(self, tmp_path, monkeypatch):
        # Yahoo and Google both carry it; the desk should see it once.
        def feed(url, timeout=20):
            title = ("ACME beats Q3 estimates - Yahoo Finance" if "yahoo" in url
                     else "ACME beats Q3 estimates - Reuters")
            return [{"title": title, "link": url,
                     "published": "2026-08-21T12:00:00+00:00", "source": ""}]
        monkeypatch.setattr(newsfeed, "fetch_rss", feed)
        m = NewsMonitor(state_path=tmp_path / "seen.json")
        assert len(m.poll(["ACME"], macro=False)) == 1

    def test_results_are_ranked_by_materiality(self, tmp_path, monkeypatch):
        feed = [
            {"title": "5 stocks to watch", "link": "a",
             "published": "2026-08-21T12:00:00+00:00", "source": ""},
            {"title": "Trading halted in ACME pending news", "link": "b",
             "published": "2026-08-21T12:00:00+00:00", "source": ""},
        ]
        monkeypatch.setattr(newsfeed, "fetch_rss", lambda url, timeout=20: feed)
        items = NewsMonitor(state_path=tmp_path / "seen.json").poll(["ACME"], macro=False)
        assert items[0].materiality > items[-1].materiality

    def test_stale_fingerprints_are_pruned(self, tmp_path, monkeypatch):
        p = tmp_path / "seen.json"
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        p.write_text(json.dumps({"deadbeef": old}))
        monkeypatch.setattr(newsfeed, "fetch_rss", lambda url, timeout=20: [])
        m = NewsMonitor(state_path=p)
        m.poll(["ACME"], macro=False)
        assert "deadbeef" not in json.loads(p.read_text())

    def test_a_dead_feed_is_silent_not_fatal(self, tmp_path, monkeypatch):
        # A network failure must degrade to "no news", never take down a loop
        # that is meant to run unattended for weeks.
        def boom(*a, **k):
            raise OSError("network down")
        monkeypatch.setattr(newsfeed.urllib.request, "urlopen", boom)
        m = NewsMonitor(state_path=tmp_path / "seen.json")
        assert m.poll(["ACME"], macro=False) == []

    def test_malformed_feed_body_yields_no_items(self, monkeypatch):
        class R:
            def read(self): return b"<not xml"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr(newsfeed.urllib.request, "urlopen", lambda *a, **k: R())
        assert newsfeed.fetch_rss("http://x") == []


@pytest.mark.unit
class TestNewsItem:
    def test_age_in_hours(self):
        two_h = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        assert 1.9 < NewsItem("MU", "t", "", "s", two_h).age_hours() < 2.1

    def test_unparseable_timestamp_is_treated_as_fresh(self):
        assert NewsItem("MU", "t", "", "s", "garbage").age_hours() == 0.0


@pytest.mark.unit
class TestInstitutionalNoise:
    """Aggregators publish thousands of 13F-flow headlines a day.

    They carry no information about the company, but several match the M&A and
    earnings patterns — "Acquires Shares of X" scores a 10 — which is enough to
    preempt a real stop-loss for the cycle's LLM budget.
    """

    @pytest.mark.parametrize("headline", [
        "GSA Capital Partners LLP Acquires Shares of 1,489 Advanced Micro Devices",
        "State of Wyoming Acquires 2,540 Shares of Advanced Micro Devices, Inc.",
        "Hsbc Holdings PLC Buys New Shares in Star Bulk Carriers Corp.",
        "Advanced Micro Devices, Inc. $AMD Shares Sold by Abbot Financial Management",
        "Star Bulk Carriers Corp. (NASDAQ:SBLK) Short Interest Update",
        "Cetera Investment Advisers Boosts Stock Holdings in Star Bulk Carriers Corp.",
        "Bridgewater Dramatically Trimmed Its Position in Micron During Q2",
        "NVIDIA Corporation $NVDA Stock Position Lowered by Jacobs & Co. CA",
        "Star Bulk Carriers (NASDAQ:SBLK) COO Sells $144,950.00 in Stock",
        "Soros Fund Exited Its Position in NVDA last quarter",
    ])
    def test_flow_headlines_stay_below_the_trigger_threshold(self, headline):
        from tradingagents.live.brain import NEWS_MATERIALITY_TRIGGER
        assert score(headline)[0] < NEWS_MATERIALITY_TRIGGER

    @pytest.mark.parametrize("headline", [
        "BigCo to acquire ACME for $4B in cash",
        "ACME acquired a rival chipmaker in an all-stock deal",
        "Trading halted in ACME pending news",
        "FDA approves ACME phase 3 therapy",
        "Activist investor builds stake in ACME and demands board seats",
    ])
    def test_real_events_survive_the_noise_filter(self, headline):
        from tradingagents.live.brain import NEWS_MATERIALITY_TRIGGER
        assert score(headline)[0] >= NEWS_MATERIALITY_TRIGGER


@pytest.mark.unit
class TestCompanyNameQueries:
    """Tickers collide with ordinary acronyms; company names do not.

    Searching Google for "AMD stock" returns age-related macular degeneration
    trial coverage, which scores 9 and would put a decision about the wrong
    company in front of the panel.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("Advanced Micro Devices, Inc.", "Advanced Micro Devices"),
        ("Star Bulk Carriers Corp.", "Star Bulk Carriers"),
        ("Alphabet Inc. Class A", "Alphabet"),
        ("Barclays PLC", "Barclays"),
        ("Airbus SE", "Airbus"),
        ("", ""),
    ])
    def test_legal_suffixes_are_stripped(self, raw, expected):
        assert newsfeed._clean_name(raw) == expected

    @pytest.mark.parametrize("raw", [
        "Some Company Holdings, Inc.",
        "The Goldman Sachs Group, Inc.",
        "Ford Motor Company",
    ])
    def test_name_words_are_not_mistaken_for_legal_forms(self, raw):
        # "Holdings"/"Group"/"Company" read like suffixes but are part of the
        # name a journalist writes; stripping them makes the query vaguer.
        cleaned = newsfeed._clean_name(raw)
        assert any(w in cleaned for w in ("Holdings", "Group", "Company"))

    def test_google_is_queried_by_name_not_ticker(self, tmp_path, monkeypatch):
        urls = []
        monkeypatch.setattr(newsfeed, "fetch_rss",
                            lambda url, timeout=20: urls.append(url) or [])
        m = NewsMonitor(state_path=tmp_path / "seen.json")
        m._names = {"AMD": "Advanced Micro Devices"}
        m.poll_ticker("AMD")
        google = [u for u in urls if "news.google.com" in u][0]
        assert "Advanced+Micro+Devices" in google or "Advanced%20Micro%20Devices" in google

    def test_falls_back_to_the_ticker_when_no_name_is_known(self, tmp_path, monkeypatch):
        urls = []
        monkeypatch.setattr(newsfeed, "fetch_rss",
                            lambda url, timeout=20: urls.append(url) or [])
        m = NewsMonitor(state_path=tmp_path / "seen.json")
        m._names = {"ZZZZ": ""}
        m.poll_ticker("ZZZZ")
        assert any("ZZZZ" in u for u in urls if "news.google.com" in u)

    def test_yahoo_stays_ticker_scoped(self, tmp_path, monkeypatch):
        urls = []
        monkeypatch.setattr(newsfeed, "fetch_rss",
                            lambda url, timeout=20: urls.append(url) or [])
        m = NewsMonitor(state_path=tmp_path / "seen.json")
        m._names = {"AMD": "Advanced Micro Devices"}
        m.poll_ticker("AMD")
        assert any("feeds.finance.yahoo.com" in u and "s=AMD" in u for u in urls)

    def test_name_lookup_is_cached_not_refetched(self, tmp_path, monkeypatch):
        calls = []
        class T:
            def __init__(self, t): calls.append(t)
            @property
            def info(self): return {"longName": "Advanced Micro Devices, Inc."}
        import yfinance
        monkeypatch.setattr(yfinance, "Ticker", T)
        cache = {}
        assert newsfeed.company_name("AMD", cache) == "Advanced Micro Devices"
        assert newsfeed.company_name("AMD", cache) == "Advanced Micro Devices"
        assert len(calls) == 1

    def test_a_failed_lookup_is_remembered_as_empty(self, monkeypatch):
        import yfinance
        def boom(t): raise RuntimeError("delisted")
        monkeypatch.setattr(yfinance, "Ticker", boom)
        cache = {}
        assert newsfeed.company_name("ZZZZ", cache) == ""
        assert cache == {"ZZZZ": ""}
