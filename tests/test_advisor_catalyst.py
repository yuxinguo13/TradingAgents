"""The WHY column must not present a filing as a reason to buy.

`catalyst` used to be `cand.news[0].title` — whatever the feed returned first.
Institutional-flow filings are capped at NOISE_CAP materiality rather than
dropped, because they belong in a news listing but not in a trigger; the
advisor never consulted that score, so on 2026-08-31 four of seven buys carried
a 13F or insider-sale headline as their stated reason.
"""

import pytest

from tradingagents.live import advisor as adv
from tradingagents.live.newsfeed import NOISE_CAP, NewsItem


def _item(title, materiality, age_h=1.0, source="Src"):
    from datetime import datetime, timedelta, timezone
    pub = (datetime.now(timezone.utc) - timedelta(hours=age_h)).isoformat()
    return NewsItem(ticker="X", title=title, link="http://e/x", source=source,
                    published=pub, materiality=materiality)


def _cand(news, rank=7, tilt=0.0):
    c = adv.Candidate(symbol="X", rank=rank)
    c.news = news
    c.tilt = tilt
    return c


@pytest.fixture
def desk():
    return adv.DailyAdvisor(adv.AdvisorConfig(), llm=None)


@pytest.mark.unit
def test_a_filing_headline_is_never_the_catalyst(desk):
    # The exact shape seen live: capped at NOISE_CAP, and the only news there is.
    noise = _item("76,221 Shares in Roku, Inc. $ROKU Bought by Czech National Bank",
                  NOISE_CAP)
    text, source, url, at = desk._catalyst(_cand([noise]))
    assert "76,221" not in text
    assert "Roku" not in text
    assert "no company news above filing noise" in text
    assert "rank #7" in text
    # Nothing is being claimed, so nothing is sourced. An empty provenance is
    # the signal that this is a rank-driven idea.
    assert (source, url, at) == ("", "", "")


@pytest.mark.unit
def test_real_news_is_preferred_over_a_filing_that_arrived_first(desk):
    # Order in the feed must not decide the reason.
    noise = _item("Meros Investment Management LP Invests $1.2 Million in BJRI",
                  NOISE_CAP, age_h=0.5)
    real = _item("FDA clears the company's lead program", 9, age_h=6.0)
    text, source, url, at = desk._catalyst(_cand([noise, real]))
    assert "FDA clears" in text
    assert "Meros" not in text
    assert url == "http://e/x"
    assert at


@pytest.mark.unit
def test_materiality_outranks_freshness_but_freshness_breaks_ties(desk):
    older_big = _item("Acquisition agreed at a premium", 9, age_h=20.0)
    newer_small = _item("Analyst nudges target", 5, age_h=0.2)
    assert "Acquisition agreed" in desk._catalyst(_cand([newer_small, older_big]))[0]

    stale9 = _item("Nine, yesterday", 9, age_h=20.0)
    fresh9 = _item("Nine, this morning", 9, age_h=1.0)
    assert "this morning" in desk._catalyst(_cand([stale9, fresh9]))[0]


@pytest.mark.unit
def test_the_catalyst_carries_source_link_and_time(desk):
    # Traceability: a claim the reader cannot date or open is one they must
    # go and re-find. Source, URL and an absolute timestamp travel separately
    # from the headline so a renderer can hyperlink and re-age them.
    text, source, url, at = desk._catalyst(
        _cand([_item("Guidance raised", 8, age_h=3.0, source="Reuters")]))
    assert text == "Guidance raised"
    assert source == "Reuters"
    assert url == "http://e/x"
    assert at  # ISO8601 publication time, not a rendered age


@pytest.mark.unit
def test_no_news_at_all_states_the_rank_without_apology(desk):
    text, source, url, at = desk._catalyst(_cand([], rank=3, tilt=-0.2))
    assert text.startswith("rank #3")
    assert "sector tilt -0.20" in text
    assert "filing noise" not in text   # there was no news, not noisy news
    assert (source, url, at) == ("", "", "")


@pytest.mark.unit
def test_an_item_missing_materiality_is_treated_as_noise(desk):
    # A stub or a future feed may not set the field; defaulting it to "usable"
    # would reopen exactly this bug.
    class Bare:
        title = "Something happened"
        source = "S"
        def age_hours(self):
            return 1.0
    text, *_ = desk._catalyst(_cand([Bare()]))
    assert "Something happened" not in text


# ---------------------------------------------------------------------------
# provenance survives the book, and an old book still loads
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_provenance_round_trips_through_the_book():
    from tradingagents.live.recommendations import Recommendation
    rec = Recommendation(
        id="X-1", issued_date="2026-08-31", symbol="X", action="BUY", shares=10,
        reference_price=10.0, stop_price=9.0, target_price=12.0,
        catalyst="Guidance raised", catalyst_source="Reuters",
        catalyst_url="https://example.test/a", catalyst_at="2026-08-28T13:00:00+00:00")
    back = Recommendation.from_dict(rec.to_dict())
    assert back.catalyst_source == "Reuters"
    assert back.catalyst_url == "https://example.test/a"
    assert back.catalyst_at == "2026-08-28T13:00:00+00:00"


@pytest.mark.unit
def test_a_book_written_before_these_fields_still_loads():
    # The book IS the track record. A schema addition that made an older file
    # unloadable would be the worst failure this module has.
    from tradingagents.live.recommendations import Recommendation
    old_row = {
        "id": "OLD-1", "issued_date": "2026-08-26", "symbol": "NRIX",
        "action": "BUY", "shares": 304, "reference_price": 26.27,
        "stop_price": 24.42, "target_price": 30.75, "catalyst": "rank #1",
    }
    rec = Recommendation.from_dict(old_row)
    assert rec is not None
    assert rec.symbol == "NRIX"
    assert (rec.catalyst_source, rec.catalyst_url, rec.catalyst_at) == ("", "", "")


@pytest.mark.unit
def test_unknown_publication_time_is_not_reported_as_just_now():
    # NaN, not 0.0: an undated headline rendering as "0h" would read as
    # breaking news, which is the exact misreading this field prevents.
    import math

    from tradingagents.live.recommendations import Recommendation
    rec = Recommendation(id="X", issued_date="2026-08-31", symbol="X", action="BUY",
                         shares=1, reference_price=1.0, stop_price=0.5,
                         target_price=2.0, catalyst="Something", catalyst_at="")
    assert math.isnan(rec.catalyst_age_hours())
    rec.catalyst_at = "not a timestamp"
    assert math.isnan(rec.catalyst_age_hours())


@pytest.mark.unit
def test_rendered_line_omits_provenance_when_there_is_none():
    from tradingagents.live.advisor import catalyst_line
    from tradingagents.live.recommendations import Recommendation
    rec = Recommendation(id="X", issued_date="2026-08-31", symbol="X", action="BUY",
                         shares=1, reference_price=1.0, stop_price=0.5,
                         target_price=2.0, catalyst="rank #7 on the nasdaq screen")
    assert catalyst_line(rec) == "rank #7 on the nasdaq screen"   # no empty brackets


@pytest.mark.unit
def test_a_pipe_in_a_headline_cannot_break_the_markdown_table():
    from tradingagents.live.advisor import _md_catalyst
    from tradingagents.live.recommendations import Recommendation
    rec = Recommendation(id="X", issued_date="2026-08-31", symbol="X", action="BUY",
                         shares=1, reference_price=1.0, stop_price=0.5, target_price=2.0,
                         catalyst="Beats | raises guidance", catalyst_url="https://e.test/a")
    cell = _md_catalyst(rec)
    assert cell.startswith("[") and "](https://e.test/a)" in cell
    assert "\\|" in cell          # escaped, so the row keeps its columns
