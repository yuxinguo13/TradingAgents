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
    out = desk._catalyst(_cand([noise]))
    assert "76,221" not in out
    assert "Roku" not in out
    assert "no company news above filing noise" in out
    assert "rank #7" in out


@pytest.mark.unit
def test_real_news_is_preferred_over_a_filing_that_arrived_first(desk):
    # Order in the feed must not decide the reason.
    noise = _item("Meros Investment Management LP Invests $1.2 Million in BJRI",
                  NOISE_CAP, age_h=0.5)
    real = _item("FDA clears the company's lead program", 9, age_h=6.0)
    out = desk._catalyst(_cand([noise, real]))
    assert "FDA clears" in out
    assert "Meros" not in out


@pytest.mark.unit
def test_materiality_outranks_freshness_but_freshness_breaks_ties(desk):
    older_big = _item("Acquisition agreed at a premium", 9, age_h=20.0)
    newer_small = _item("Analyst nudges target", 5, age_h=0.2)
    assert "Acquisition agreed" in desk._catalyst(_cand([newer_small, older_big]))

    stale9 = _item("Nine, yesterday", 9, age_h=20.0)
    fresh9 = _item("Nine, this morning", 9, age_h=1.0)
    assert "this morning" in desk._catalyst(_cand([stale9, fresh9]))


@pytest.mark.unit
def test_the_catalyst_carries_source_and_age(desk):
    # Traceability: a claim the reader cannot date is a claim they must re-search.
    out = desk._catalyst(_cand([_item("Guidance raised", 8, age_h=3.0, source="Reuters")]))
    assert "Reuters" in out
    assert "3h" in out


@pytest.mark.unit
def test_no_news_at_all_states_the_rank_without_apology(desk):
    out = desk._catalyst(_cand([], rank=3, tilt=-0.2))
    assert out.startswith("rank #3")
    assert "sector tilt -0.20" in out
    assert "filing noise" not in out   # there was no news, not noisy news


@pytest.mark.unit
def test_an_item_missing_materiality_is_treated_as_noise(desk):
    # A stub or a future feed may not set the field; defaulting it to "usable"
    # would reopen exactly this bug.
    class Bare:
        title = "Something happened"
        source = "S"
        def age_hours(self):
            return 1.0
    out = desk._catalyst(_cand([Bare()]))
    assert "Something happened" not in out
