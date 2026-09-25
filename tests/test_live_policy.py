"""Policy event detection, the sector impact map, and the tilt it produces.

No network: every feed is a monkeypatched ``fetch_rss``. The failures these
tests exist to prevent are the ones that make a policy monitor worse than
none — a speculation column scoring as an event and spending an LLM call, a
restart replaying a fortnight of tariff coverage as if it just broke, a typo in
a sector label silently zeroing the tilt for that sector, and any single
unreachable feed taking down a loop meant to run unattended for weeks.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.live import newsfeed, policy
from tradingagents.live.policy import (
    ALL_SECTORS,
    CATEGORIES,
    CONSUMER_CYCLICAL,
    ENERGY,
    FINANCIALS,
    HEALTHCARE,
    IMPACT_RULES,
    MAX_SEVERITY,
    POLICY_SEVERITY_TRIGGER,
    REAL_ESTATE,
    TECHNOLOGY,
    UTILITIES,
    PolicyEvent,
    PolicyMonitor,
    classify,
    policy_brief,
    sector_pressure,
)

NOW = datetime.now(timezone.utc)


def iso(hours_ago: float = 0.0) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def item(title: str, published: str | None = None, link: str = "u", source: str = "") -> dict:
    """One parsed RSS row, shaped exactly as ``newsfeed.fetch_rss`` returns it."""
    return {"title": title, "link": link,
            "published": published or iso(1), "source": source}


def feed_of(*titles: str, published: str | None = None):
    """A ``fetch_rss`` stand-in that answers every URL with the same rows."""
    rows = [item(t, published) for t in titles]
    return lambda url, timeout=20: list(rows)


def event(category="trade", headline="h", severity=5, direction="neutral",
          sectors=None, published=None, source="") -> PolicyEvent:
    return PolicyEvent(
        category=category, headline=headline, url="https://example.invalid/x",
        published=published or iso(1), severity=severity, direction=direction,
        sector_impact=dict(sectors or {}), rationale="because", source=source,
        fingerprint=headline,
    )


# ---------------------------------------------------------------------------
# the map itself
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestImpactMapIntegrity:
    """A typo in this table does not fail loudly — it silently stops matching."""

    def test_every_pattern_compiles(self):
        import re
        for rule in IMPACT_RULES:
            re.compile(rule.pattern, re.IGNORECASE)   # raises if malformed

    def test_rule_keys_are_unique(self):
        keys = [r.key for r in IMPACT_RULES]
        assert len(keys) == len(set(keys))

    def test_every_sector_label_is_a_known_sector(self):
        # The tilt is applied against yfinance's sector strings. A misspelled
        # label here never matches anything and the sector's tilt is quietly
        # always zero, which looks identical to "no policy touched it".
        for rule in IMPACT_RULES:
            for sector in rule.sectors:
                assert sector in ALL_SECTORS, f"{rule.key} names unknown sector {sector!r}"

    def test_every_sign_is_minus_one_zero_or_plus_one(self):
        # sector_pressure multiplies sign by severity; a sign of 3 would let one
        # row outweigh the whole rest of the table.
        for rule in IMPACT_RULES:
            assert set(rule.sectors.values()) <= {-1, 0, 1}, rule.key

    def test_every_category_is_a_polled_category(self):
        # A rule filed under a category nothing queries is unreachable.
        for rule in IMPACT_RULES:
            assert rule.category in CATEGORIES, rule.key

    def test_severities_are_in_range(self):
        for rule in IMPACT_RULES:
            assert 1 <= rule.severity <= MAX_SEVERITY, rule.key

    def test_every_rule_states_its_mechanism(self):
        # A sign with no rationale cannot be argued with when it turns out
        # wrong. Length is not the bar — the mirror rows legitimately say only
        # "the mirror of the row above" — but a written sentence is.
        for rule in IMPACT_RULES:
            assert rule.rationale.strip().endswith("."), rule.key

    def test_every_category_has_queries(self):
        for cat, queries in CATEGORIES.items():
            assert queries, cat


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassification:
    @pytest.mark.parametrize("headline,category", [
        ("Fed cuts interest rates by 25 basis points", "monetary"),
        ("Inflation cools to 2.4% in July CPI report", "monetary"),
        ("Government shutdown begins as Congress fails to pass a budget", "fiscal"),
        ("Senate approves debt ceiling suspension", "fiscal"),
        ("US imposes new tariffs on Chinese imports", "trade"),
        ("Commerce Department expands semiconductor export controls", "trade"),
        ("Senate passes drug pricing legislation for Medicare", "regulatory"),
        ("FTC files antitrust suit over monopoly conduct", "regulatory"),
        ("OPEC+ agrees to production cuts of one million barrels", "geopolitical"),
        ("Taiwan Strait tensions rise as drills expand", "geopolitical"),
    ])
    def test_policy_headlines_are_recognised(self, headline, category):
        ev = classify(headline, published=iso(1))
        assert ev is not None
        assert ev.category == category

    @pytest.mark.parametrize("headline", [
        "",
        "Cats are cute",
        "ACME beats Q3 earnings estimates",
        "5 stocks to watch this week",
    ])
    def test_non_policy_headlines_are_not_events(self, headline):
        # None is the common answer and the point of the function: Google
        # returns whatever it likes for "sanctions".
        assert classify(headline, published=iso(1)) is None

    def test_institutional_flow_noise_is_suppressed(self):
        # newsfeed's 13F filter is imported rather than copied; a policy query
        # like "defense budget" returns these by the hundred and none of them
        # is a policy event.
        assert classify("Ninepoint Partners LP Boosts Position in Lockheed Martin "
                        "on defense budget increase", published=iso(1)) is None

    def test_category_comes_from_the_rule_not_the_query(self, tmp_path, monkeypatch):
        # A tariff story returned by the geopolitical query is a trade event.
        # Filing it under the query would scatter one theme across the brief.
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        evs = PolicyMonitor(state_path=tmp_path / "s.json").poll_category("geopolitical")
        assert [e.category for e in evs] == ["trade"]

    def test_severity_never_leaves_the_scale(self):
        # Bumps stack (action verb + newsfeed materiality) and penalties
        # subtract; the published contract is 0-10.
        for rule in IMPACT_RULES:
            for prefix in ("", "SEC opens investigation as ", "Senate votes to "):
                ev = classify(prefix + _example_for(rule.key), published=iso(1))
                if ev is not None:
                    assert 1 <= ev.severity <= MAX_SEVERITY

    def test_fingerprint_is_namespaced_away_from_ticker_news(self):
        # Otherwise a policy headline already seen by NewsMonitor would be
        # swallowed here, or vice versa.
        ev = classify("US imposes new tariffs on Chinese imports", published=iso(1))
        assert ev is not None
        assert ev.fingerprint != newsfeed._fingerprint(ev.headline, "AAPL")

    def test_missing_published_defaults_to_now(self):
        ev = classify("US imposes new tariffs on Chinese imports")
        assert ev is not None
        assert ev.age_hours() < 1

    def test_age_hours_matches_the_news_reading_of_the_same_stamp(self):
        stamp = iso(6)
        ev = classify("US imposes new tariffs on Chinese imports", published=stamp)
        assert ev is not None
        reference = newsfeed.NewsItem("", "t", "u", "s", stamp).age_hours()
        assert ev.age_hours() == pytest.approx(reference, abs=0.01)

    def test_unparseable_timestamp_is_treated_as_fresh_not_fatal(self):
        ev = classify("US imposes new tariffs on Chinese imports", published="not a date")
        assert ev is not None
        assert ev.age_hours() == 0.0

    def test_affected_sectors_tracks_the_impact_map(self):
        ev = classify("US imposes new tariffs on Chinese imports", published=iso(1))
        assert ev is not None
        assert ev.affected_sectors == list(ev.sector_impact)
        assert CONSUMER_CYCLICAL in ev.affected_sectors


def _example_for(key: str) -> str:
    """A headline that fires one named rule, for the range sweep above."""
    return {
        "rate_cut": "Fed announces rate cuts",
        "rate_hike": "Fed raises interest rates again",
        "inflation_hot": "CPI comes in hot as inflation accelerates",
        "inflation_cool": "Inflation cools sharply in the latest print",
        "fomc": "FOMC decision lands Wednesday",
        "shutdown": "Government shutdown begins",
        "debt_ceiling": "Debt ceiling standoff drags on",
        "corp_tax_up": "Congress backs a corporate tax hike",
        "corp_tax_down": "Congress backs corporate tax cuts",
        "fiscal_spend": "Infrastructure bill signed",
        "chip_controls": "New semiconductor export controls announced",
        "tariffs": "US imposes new tariffs on imports",
        "sanctions": "US announces sanctions on the central bank",
        "trade_deal": "US and China reach a trade deal",
        "trade_breakdown": "Trade talks collapse in Geneva",
        "antitrust": "FTC opens an antitrust case over monopoly conduct",
        "drug_pricing": "Medicare drug pricing negotiation expands",
        "health_programs": "Congress approves Medicaid cuts",
        "fda_policy": "FDA guidance overhauls the accelerated approval pathway",
        "sec_rules": "SEC adopts climate disclosure rules",
        "bank_capital": "Basel capital requirements finalised",
        "crypto_rules": "Stablecoin bill clears the Senate",
        "energy_permits": "LNG export permit approved",
        "energy_restrictions": "Pipeline blocked by the regulator",
        "ev_and_emissions": "EV tax credits repealed",
        "opec_cut": "OPEC+ agrees to production cuts",
        "opec_raise": "OPEC+ raises output quotas",
        "conflict": "Missile strike escalates the conflict",
        "ceasefire": "Ceasefire agreed after months of talks",
        "taiwan": "Taiwan Strait tensions escalate",
        "election": "Presidential race narrows before the election",
    }[key]


# ---------------------------------------------------------------------------
# the specific directional claims the module makes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSectorDirections:
    """The worked examples the map was written from. Heuristics, but stated ones."""

    def test_chip_export_controls_hit_technology(self):
        ev = classify("US expands semiconductor export controls on AI chips",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[TECHNOLOGY] == -1
        assert ev.direction == "bearish"

    def test_tariffs_hit_importers(self):
        ev = classify("White House imposes tariffs on imported goods", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[CONSUMER_CYCLICAL] == -1
        assert ev.sector_impact["Industrials"] == -1

    def test_rate_cut_lifts_duration_and_leaves_banks_unsigned(self):
        # Real Estate and Utilities are long-duration and levered; a bank is
        # two-sided (cheaper funding, thinner margin) so it gets no sign.
        ev = classify("Fed cuts interest rates by 25 basis points", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[REAL_ESTATE] == 1
        assert ev.sector_impact[UTILITIES] == 1
        assert ev.sector_impact[FINANCIALS] == 0

    def test_drug_pricing_hits_healthcare(self):
        ev = classify("Senate passes drug pricing legislation", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[HEALTHCARE] == -1

    def test_opec_supply_cut_helps_producers_and_hurts_burners(self):
        ev = classify("OPEC+ agrees to production cuts of one million barrels",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[ENERGY] == 1
        assert ev.sector_impact["Industrials"] == -1
        # Opposing signs must not be netted into one word about the index.
        assert ev.direction == "neutral"

    def test_an_unsigned_exposure_is_recorded_not_dropped(self):
        # "Exposed, direction unknown" is information: it should widen the
        # panel's uncertainty, not vanish.
        ev = classify("FDA guidance overhauls the accelerated approval pathway",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact == {HEALTHCARE: 0}
        assert ev.direction == "neutral"

    def test_zeros_do_not_veto_a_direction_the_signed_sectors_agree_on(self):
        ev = classify("Fed cuts interest rates by 25 basis points", published=iso(1))
        assert ev is not None
        assert 0 in ev.sector_impact.values()
        assert ev.direction == "bullish"

    def test_an_election_is_surfaced_without_tilting_anything(self):
        # The sector effect of an election is a function of the result, which is
        # not in a headline about the campaign.
        ev = classify("Presidential race tightens ahead of the election",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact == {}
        assert sector_pressure([ev]) == {}

    def test_two_rules_disagreeing_about_a_sector_resolve_to_unsigned(self):
        # Sanctions (Industrials -1) and conflict (Industrials +1) on one
        # headline is a real two-sided exposure, not an error to pick a side on.
        ev = classify("US announces sanctions and airstrikes", published=iso(1))
        assert ev is not None
        assert ev.sector_impact["Industrials"] == 0


# ---------------------------------------------------------------------------
# what happens when two rows fire on one headline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOverlappingRules:
    """A zero and a sign are not two claims about the same thing.

    Zero means "exposed, sign genuinely unknown", so it must never cancel a
    sign another row asserted. Two opposite signs are a genuine disagreement
    and must still net to zero.
    """

    def test_an_unsigned_row_does_not_erase_the_signed_row_it_overlaps(self):
        # "Powell" fires the fomc row, which marks Real Estate, Utilities and
        # Technology 0 by design. Counting that as disagreement zeroed the
        # three legs the rate_cut row exists for, so the highest-severity
        # event in the table produced no tilt at all.
        ev = classify("Powell says the Fed will cut rates twice this year",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[REAL_ESTATE] == 1
        assert ev.sector_impact[UTILITIES] == 1
        assert ev.sector_impact[TECHNOLOGY] == 1
        assert ev.direction == "bullish"
        assert sector_pressure([ev])[REAL_ESTATE] > 0

    def test_naming_the_venue_does_not_change_the_tilt(self):
        # One event, two house styles. The signs used to turn on whether the
        # publisher put "FOMC" or "Powell" in the headline.
        bare = classify("Fed cuts interest rates", published=iso(1))
        venue = classify("FOMC statement confirms a quarter-point rate cut",
                         published=iso(1))
        assert bare is not None and venue is not None
        assert bare.sector_impact == venue.sector_impact
        assert bare.direction == venue.direction == "bullish"

    def test_an_unsigned_row_does_not_erase_a_rate_hike(self):
        # The mirror of the case above: "Fed decision" fires the unsigned fomc
        # row over the top of rate_hike's signed one.
        ev = classify("Fed decision: policymakers vote to raise rates",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[REAL_ESTATE] == -1
        assert ev.sector_impact[UTILITIES] == -1
        assert ev.direction == "bearish"

    def test_an_unsigned_regulatory_row_does_not_erase_a_signed_one(self):
        # crypto_rules marks Financial Services 0, sec_rules marks it -1. The
        # zero used to erase the compliance-cost sign.
        ev = classify("SEC adopts crypto rules", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[FINANCIALS] == -1
        assert ev.direction == "bearish"

    def test_an_unsigned_health_row_does_not_erase_the_drug_pricing_sign(self):
        # fda_policy is Healthcare 0, drug_pricing is Healthcare -1.
        ev = classify("FDA guidance overhaul targets drug pricing", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[HEALTHCARE] == -1
        assert ev.direction == "bearish"

    def test_a_sector_no_row_signed_stays_unsigned(self):
        # The exemption must not invent a sign: Financial Services is 0 in
        # both rows that touch it here, and reporting it as +1 would claim
        # something no row in the table says.
        ev = classify("Powell says the Fed will cut rates twice this year",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[FINANCIALS] == 0

    def test_opposite_signs_still_resolve_to_unsigned(self):
        # The exemption is for zeros only. Sanctions (Industrials -1) and
        # conflict (Industrials +1) remain a real two-sided exposure.
        ev = classify("US announces sanctions and airstrikes", published=iso(1))
        assert ev is not None
        assert ev.sector_impact["Industrials"] == 0
        assert ev.direction == "neutral"


# ---------------------------------------------------------------------------
# the OPEC rows are about crude, and only about crude
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOilRowsRequireOilContext:
    @pytest.mark.parametrize("headline", [
        "Samsung announces memory production cuts",
        "TSMC announces output cuts amid weak demand",
        "Boeing output increase planned for 737 line",
    ])
    def test_a_manufacturer_changing_output_is_not_an_oil_event(self, headline):
        # The unanchored `(?:production|output) cuts?` alternative made a
        # chipmaker's capacity decision a severity-8 geopolitical event: over
        # the trigger, so it spent an LLM call, and it tilted Energy long.
        assert classify(headline, published=iso(1)) is None

    def test_an_oil_cut_is_still_an_energy_event_without_the_word_opec(self):
        # Anchoring must not cost the headlines the row exists for.
        ev = classify("Saudi Arabia extends crude output cuts through June",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[ENERGY] == 1

    def test_the_oil_context_may_follow_the_cut(self):
        ev = classify("Production cuts push crude prices higher", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[ENERGY] == 1


# ---------------------------------------------------------------------------
# polarity: escalation vs rollback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPolarity:
    def test_a_rollback_inverts_an_instrument_rule(self):
        # The map's signs are written for the escalation reading. "US lifts
        # tariffs" is the same instrument pointing the other way, and reading it
        # as bearish would be exactly backwards.
        ev = classify("US lifts tariffs on European steel", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[CONSUMER_CYCLICAL] == 1
        assert ev.direction == "bullish"

    def test_a_fixed_sign_rule_is_not_inverted(self):
        # "eases" is a rollback word, but "rate cut" already names its own
        # direction; flipping it would invert a sign that was correct.
        ev = classify("Fed eases policy with rate cuts", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[REAL_ESTATE] == 1

    def test_both_directions_present_makes_no_claim(self):
        # A headline carrying an escalation word and a rollback word has not
        # told us which way it goes; the declared sign stands.
        ev = classify("White House expands and eases tariffs simultaneously",
                      published=iso(1))
        assert ev is not None
        assert ev.sector_impact[CONSUMER_CYCLICAL] == -1

    def test_neither_direction_present_keeps_the_declared_sign(self):
        ev = classify("Tariffs on Chinese imports take effect", published=iso(1))
        assert ev is not None
        assert ev.sector_impact[CONSUMER_CYCLICAL] == -1


# ---------------------------------------------------------------------------
# speculation suppression
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpeculationSuppression:
    """Policy feeds are mostly columns. A column is not an event."""

    @pytest.mark.parametrize("headline", [
        "Opinion: why the tariff debate matters",
        "Analysis: the Fed is likely to cut rates in September",
        "What to expect from the FOMC decision this week",
        "Here's what the drug pricing bill means for pharma",
        "Traders brace for tariffs ahead of the deadline",
        "Explainer: how the debt ceiling works",
        "Fed may cut rates in September",
        "Congress might approve Medicaid cuts",
    ])
    def test_speculation_cannot_reach_the_trigger(self, headline):
        ev = classify(headline, published=iso(1))
        if ev is None:
            return                       # not an event at all is also fine
        assert ev.severity <= policy._SPECULATION_CAP
        assert ev.severity < POLICY_SEVERITY_TRIGGER

    def test_the_month_of_may_is_not_the_word_may(self):
        # The capital letter is the only cheap way to tell the hedge from the
        # month, and capping a real event because it happened in May would lose
        # a month of policy every year.
        ev = classify("Powell announces the decision at Jackson Hole in May",
                      published=iso(1))
        assert ev is not None
        assert ev.severity > policy._SPECULATION_CAP

    @pytest.mark.parametrize("headline", [
        "Fed May Cut Rates In September",
        "Powell Says Fed Might Pause Hikes",
        "Trump May Impose New Tariffs On EU Goods",
    ])
    def test_a_title_cased_hedge_is_capped_like_a_lowercase_one(self, headline):
        # The aggregators that fill much of Google News RSS publish in Title
        # Case, where the capital letter no longer separates the modal from
        # the month. "Fed May Cut Rates In September" scored 8 and cleared the
        # trigger while the same story in sentence case was capped at 3 — pure
        # speculation spending an LLM call on the publisher's house style.
        ev = classify(headline, published=iso(1))
        assert ev is not None
        assert ev.severity <= policy._SPECULATION_CAP
        assert ev.severity < POLICY_SEVERITY_TRIGGER

    def test_a_title_cased_month_is_still_the_month(self):
        # The title-case branch must not cap a decision that happened, or the
        # fix for the hedge costs a month of real policy every year.
        ev = classify("Fed Cuts Rates At May Meeting", published=iso(1))
        assert ev is not None
        assert ev.severity > policy._SPECULATION_CAP

    def test_proper_nouns_do_not_make_a_sentence_case_headline_title_cased(self):
        # If the detector fired on sentence case, the case-insensitive branch
        # would take over and cap every real event published in May.
        assert not policy._title_cased(
            "Powell announces the decision at Jackson Hole in May")
        assert policy._title_cased("Fed May Cut Rates In September")

    def test_a_proposal_scores_below_the_enacted_version(self):
        # A proposed rule is a real object with a real effect, just a smaller
        # one — docked rather than capped.
        proposed = classify("White House proposes new tariffs on steel", published=iso(1))
        imposed = classify("White House imposes new tariffs on steel", published=iso(1))
        assert proposed is not None and imposed is not None
        assert proposed.severity < imposed.severity

    def test_an_enacted_action_clears_the_trigger(self):
        ev = classify("US imposes new tariffs on Chinese imports", published=iso(1))
        assert ev is not None
        assert ev.severity >= POLICY_SEVERITY_TRIGGER


# ---------------------------------------------------------------------------
# the monitor
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPolicyMonitorNovelty:
    def test_second_poll_returns_nothing_new(self, tmp_path, monkeypatch):
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert len(m.poll(pause=0)) == 1
        assert m.poll(pause=0) == []

    def test_novelty_survives_a_restart(self, tmp_path, monkeypatch):
        # Without persistence, every restart replays the week's policy coverage
        # and tilts the book on Monday's tariff again on Thursday.
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        p = tmp_path / "s.json"
        assert len(PolicyMonitor(state_path=p).poll(pause=0)) == 1
        assert PolicyMonitor(state_path=p).poll(pause=0) == []

    def test_one_story_returned_by_two_categories_is_one_event(self, tmp_path, monkeypatch):
        # Every policy story shares one fingerprint namespace, so a tariff
        # headline found by both the trade and the geopolitical query does not
        # get double weight in the tilt.
        def feed(url, timeout=20):
            return [item("US imposes new tariffs on Chinese imports - Reuters"
                         if "trade" in url or "tariff" in url else
                         "US imposes new tariffs on Chinese imports - AP")]
        monkeypatch.setattr(policy, "fetch_rss", feed)
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert len(m.poll(pause=0)) == 1

    def test_non_events_are_marked_seen_too(self, tmp_path, monkeypatch):
        # Classification is deterministic: a headline that is not an event today
        # will not be one tomorrow, so re-deciding it every cycle is pure work.
        monkeypatch.setattr(policy, "fetch_rss", feed_of("Cats are cute"))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert m.poll(pause=0) == []
        assert newsfeed._fingerprint("Cats are cute", policy._NAMESPACE) in m.seen

    def test_prime_swallows_the_backlog(self, tmp_path, monkeypatch):
        # First start must not find a fortnight of tariff coverage and tilt the
        # whole book on it.
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert m.prime() == 1
        assert m.poll(pause=0) == []

    def test_stale_fingerprints_are_pruned(self, tmp_path, monkeypatch):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"deadbeef": (NOW - timedelta(days=30)).isoformat()}))
        monkeypatch.setattr(policy, "fetch_rss", lambda url, timeout=20: [])
        PolicyMonitor(state_path=p).poll(pause=0)
        assert "deadbeef" not in json.loads(p.read_text())

    def test_fresh_fingerprints_are_kept(self, tmp_path, monkeypatch):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"cafe": (NOW - timedelta(hours=1)).isoformat()}))
        monkeypatch.setattr(policy, "fetch_rss", lambda url, timeout=20: [])
        PolicyMonitor(state_path=p).poll(pause=0)
        assert "cafe" in json.loads(p.read_text())

    def test_state_is_written_atomically(self, tmp_path, monkeypatch):
        # A half-written seen set reads back as an empty one, which replays the
        # whole feed — the exact failure the tmp+replace dance prevents.
        p = tmp_path / "s.json"
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        PolicyMonitor(state_path=p).poll(pause=0)
        assert json.loads(p.read_text())
        assert not p.with_suffix(".tmp").exists()


@pytest.mark.unit
class TestPolicyMonitorAgeFilter:
    def test_old_coverage_is_dropped(self, tmp_path, monkeypatch):
        # A Google policy search is not chronological: it routinely returns last
        # year's tariff coverage alongside this morning's, and "new to this
        # process" is not "new to the world".
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports",
                                    published=iso(24 * 30)))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert m.poll(pause=0) == []

    def test_recent_coverage_is_kept(self, tmp_path, monkeypatch):
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports",
                                    published=iso(2)))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert len(m.poll(pause=0)) == 1

    def test_the_age_cut_is_configurable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports",
                                    published=iso(10)))
        assert PolicyMonitor(state_path=tmp_path / "a.json",
                             max_age_hours=4).poll(pause=0) == []
        assert PolicyMonitor(state_path=tmp_path / "b.json",
                             max_age_hours=72).poll(pause=0)

    def test_a_dropped_story_stays_seen(self, tmp_path, monkeypatch):
        # Otherwise the same stale headline is re-fetched and re-classified on
        # every cycle for as long as Google keeps returning it.
        title = "US imposes new tariffs on Chinese imports"
        monkeypatch.setattr(policy, "fetch_rss", feed_of(title, published=iso(24 * 30)))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        m.poll(pause=0)
        assert newsfeed._fingerprint(title, policy._NAMESPACE) in m.seen


@pytest.mark.unit
class TestPolicyMonitorFaultTolerance:
    """Nothing here may raise out of a loop meant to run for weeks."""

    def test_a_dead_feed_is_silent_not_fatal(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise OSError("network down")
        monkeypatch.setattr(policy, "fetch_rss", boom)
        assert PolicyMonitor(state_path=tmp_path / "s.json").poll(pause=0) == []

    def test_one_broken_category_does_not_starve_the_others(self, tmp_path, monkeypatch):
        def feed(url, timeout=20):
            if "tariffs" in url:
                raise RuntimeError("that one feed is down")
            return [item("Fed cuts interest rates by 25 basis points")]
        monkeypatch.setattr(policy, "fetch_rss", feed)
        evs = PolicyMonitor(state_path=tmp_path / "s.json").poll(pause=0)
        assert any(e.category == "monetary" for e in evs)

    def test_a_classifier_failure_skips_the_headline(self, tmp_path, monkeypatch):
        # A regex table must never take down the loop.
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        monkeypatch.setattr(policy, "classify",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
        assert PolicyMonitor(state_path=tmp_path / "s.json").poll(pause=0) == []

    def test_an_unwritable_state_path_does_not_raise(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory")
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        m = PolicyMonitor(state_path=blocker / "s.json")
        assert len(m.poll(pause=0)) == 1      # events still returned

    def test_a_corrupt_state_file_reads_as_empty(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text("{not json")
        assert PolicyMonitor(state_path=p).seen == {}

    @pytest.mark.parametrize("body", [
        "[1, 2, 3]", '{"abc": 123}', '"hello"', "null", '{"a": null}',
    ])
    def test_a_state_file_of_the_wrong_shape_reads_as_empty(self, tmp_path, body):
        # _load caught only unparseable JSON. A file that parsed but was not
        # dict[str, str] got through and detonated in _save's prune instead:
        # AttributeError on a list, TypeError comparing an int to the cutoff.
        p = tmp_path / "s.json"
        p.write_text(body)
        assert PolicyMonitor(state_path=p).seen == {}

    def test_a_partly_wrong_state_file_keeps_the_usable_fingerprints(self, tmp_path):
        # Dropping the whole file on one bad value would replay every story it
        # still remembered, which is the failure the seen set exists to stop.
        stamp = iso(1)
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"good": stamp, "bad": 123}))
        assert PolicyMonitor(state_path=p).seen == {"good": stamp}

    def test_a_wrong_shape_state_file_does_not_kill_the_poll(self, tmp_path, monkeypatch):
        # The failure prevented is a hard exit from a loop meant to run
        # unattended for weeks, on the strength of one bad state file.
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"deadbeef": 123}))
        monkeypatch.setattr(policy, "fetch_rss",
                            feed_of("US imposes new tariffs on Chinese imports"))
        assert len(PolicyMonitor(state_path=p).poll(pause=0)) == 1
        assert json.loads(p.read_text())        # rewritten in a shape it can read

    def test_a_seen_set_corrupted_in_memory_does_not_raise(self, tmp_path, monkeypatch):
        # The prune line used to sit outside the suppress, so a non-string
        # stamp reaching it raised straight out of poll().
        monkeypatch.setattr(policy, "fetch_rss", lambda url, timeout=20: [])
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        m.seen = {"deadbeef": 123}              # type: ignore[dict-item]
        assert m.poll(pause=0) == []

    def test_a_feed_row_missing_fields_is_tolerated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(policy, "fetch_rss", lambda url, timeout=20: [{}, {"title": ""}])
        assert PolicyMonitor(state_path=tmp_path / "s.json").poll(pause=0) == []


@pytest.mark.unit
class TestPolicyMonitorQuerying:
    def test_only_keyless_google_queries_are_fetched(self, tmp_path, monkeypatch):
        seen_urls: list[str] = []

        def feed(url, timeout=20):
            seen_urls.append(url)
            return []
        monkeypatch.setattr(policy, "fetch_rss", feed)
        PolicyMonitor(state_path=tmp_path / "s.json").poll(pause=0)
        assert seen_urls
        assert all(u.startswith("https://news.google.com/rss/search?q=") for u in seen_urls)
        assert len(seen_urls) == sum(len(q) for q in CATEGORIES.values())

    def test_a_category_subset_polls_only_that_category(self, tmp_path, monkeypatch):
        seen_urls: list[str] = []

        def feed(url, timeout=20):
            seen_urls.append(url)
            return []
        monkeypatch.setattr(policy, "fetch_rss", feed)
        PolicyMonitor(state_path=tmp_path / "s.json").poll(["monetary"], pause=0)
        assert len(seen_urls) == len(CATEGORIES["monetary"])

    def test_an_unknown_category_is_empty_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(policy, "fetch_rss", feed_of("US imposes new tariffs"))
        m = PolicyMonitor(state_path=tmp_path / "s.json")
        assert m.poll_category("astrology") == []

    def test_results_are_ranked_most_severe_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(policy, "fetch_rss", feed_of(
            "SEC adopts a market structure rule",
            "US imposes new tariffs on Chinese imports",
        ))
        evs = PolicyMonitor(state_path=tmp_path / "s.json").poll(pause=0)
        assert [e.severity for e in evs] == sorted((e.severity for e in evs), reverse=True)

    def test_state_defaults_under_tradingagents_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        assert PolicyMonitor().state_path == tmp_path / "policy_seen.json"


# ---------------------------------------------------------------------------
# the brief
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPolicyBrief:
    def test_no_events_says_so(self):
        assert policy_brief([]) == "(no policy or political events)"

    def test_grouped_by_category(self):
        text = policy_brief([
            event("trade", "tariffs land", 9, "bearish", {CONSUMER_CYCLICAL: -1}),
            event("monetary", "rates cut", 8, "bullish", {REAL_ESTATE: 1}),
        ])
        assert "### trade (1)" in text
        assert "### monetary (1)" in text

    def test_most_severe_category_leads(self):
        text = policy_brief([
            event("monetary", "rates cut", 4, "bullish", {REAL_ESTATE: 1}),
            event("trade", "tariffs land", 9, "bearish", {CONSUMER_CYCLICAL: -1}),
        ])
        assert text.index("### trade") < text.index("### monetary")

    def test_most_severe_event_leads_inside_a_category(self):
        text = policy_brief([
            event("trade", "small trade thing", 3, sectors={CONSUMER_CYCLICAL: -1}),
            event("trade", "big trade thing", 9, sectors={CONSUMER_CYCLICAL: -1}),
        ])
        assert text.index("big trade thing") < text.index("small trade thing")

    def test_per_category_cap_reports_what_it_hid(self):
        evs = [event("trade", f"thing {i}", 9 - i, sectors={CONSUMER_CYCLICAL: -1})
               for i in range(6)]
        text = policy_brief(evs, per_category=2)
        assert "... and 4 more in trade" in text

    def test_unsigned_sectors_print_as_unknown_not_dropped(self):
        text = policy_brief([event("monetary", "FOMC meets", 7,
                                   sectors={FINANCIALS: 0, REAL_ESTATE: 1})])
        assert "Financial Services?" in text
        assert "Real Estate+" in text

    def test_the_caveat_travels_with_the_events(self):
        # The model reading the brief is the one that needs to know how far to
        # trust it, so the caveat cannot live only in a docstring.
        text = policy_brief([event(sectors={ENERGY: 1})])
        assert "directional heuristics" in text
        assert "already be priced" in text

    def test_min_severity_filters(self):
        evs = [event("trade", "loud", 8, sectors={CONSUMER_CYCLICAL: -1}),
               event("trade", "quiet", 2, sectors={CONSUMER_CYCLICAL: -1})]
        text = policy_brief(evs, min_severity=5)
        assert "loud" in text
        assert "quiet" not in text

    def test_filtering_everything_out_says_so(self):
        assert policy_brief([event(severity=1)], min_severity=9) == \
            "(no policy or political events)"

    def test_a_sectorless_event_still_appears(self):
        text = policy_brief([event("geopolitical", "election nears", 5)])
        assert "election nears" in text
        assert "sectors:" not in text

    def test_long_headlines_are_truncated(self):
        text = policy_brief([event("trade", "T" * 400, 9, sectors={ENERGY: 1})])
        assert "T" * 141 not in text

    def test_the_brief_is_markdown_a_model_can_read(self):
        text = policy_brief([event("trade", "tariffs land", 9, "bearish",
                                   {CONSUMER_CYCLICAL: -1}, source="Reuters")])
        assert text.startswith("## Policy and political backdrop")
        assert "[9/bearish]" in text
        assert "Reuters" in text


# ---------------------------------------------------------------------------
# the tilt
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSectorPressure:
    def test_no_events_no_pressure(self):
        assert sector_pressure([]) == {}

    def test_sign_follows_the_impact(self):
        p = sector_pressure([event(sectors={ENERGY: 1, CONSUMER_CYCLICAL: -1})])
        assert p[ENERGY] > 0
        assert p[CONSUMER_CYCLICAL] < 0

    def test_bounded_to_the_unit_interval(self):
        # The tilt multiplies into a ranking; a value outside [-1, 1] would let
        # policy override the screen it is only supposed to reorder.
        evs = [event(severity=MAX_SEVERITY, sectors={ENERGY: 1}) for _ in range(200)]
        p = sector_pressure(evs)
        assert -1.0 < p[ENERGY] < 1.0
        assert p[ENERGY] > 0.9

    def test_one_headline_cannot_saturate_the_scale(self):
        p = sector_pressure([event(severity=MAX_SEVERITY, sectors={ENERGY: 1})])
        assert 0 < p[ENERGY] < 0.5

    def test_a_pile_on_accumulates(self):
        one = sector_pressure([event(severity=8, sectors={ENERGY: 1})])[ENERGY]
        five = sector_pressure([event(severity=8, sectors={ENERGY: 1})
                                for _ in range(5)])[ENERGY]
        assert five > one

    def test_severity_orders_the_tilt(self):
        big = sector_pressure([event(severity=9, sectors={ENERGY: 1})])[ENERGY]
        small = sector_pressure([event(severity=2, sectors={ENERGY: 1})])[ENERGY]
        assert big > small

    def test_a_weak_event_is_not_diluted_away_by_many_stronger_ones(self):
        # Summing rather than averaging: eight severity-1 headlines must not
        # wash out one severity-9 event.
        evs = [event(severity=9, sectors={ENERGY: 1})]
        evs += [event(severity=1, sectors={ENERGY: 1}) for _ in range(8)]
        assert sector_pressure(evs)[ENERGY] > sector_pressure(evs[:1])[ENERGY]

    def test_opposing_events_cancel(self):
        p = sector_pressure([event(severity=7, sectors={ENERGY: 1}),
                             event(severity=7, sectors={ENERGY: -1})])
        assert p[ENERGY] == pytest.approx(0.0)

    def test_a_zero_sector_is_present_and_flat(self):
        # Present-with-zero means "exposed, direction unknown"; absent means
        # "no policy touched it". Collapsing the two loses the warning.
        p = sector_pressure([event(sectors={FINANCIALS: 0})])
        assert p == {FINANCIALS: 0.0}

    def test_untouched_sectors_are_absent(self):
        p = sector_pressure([event(sectors={ENERGY: 1})])
        assert set(p) == {ENERGY}

    def test_pressure_from_classified_events_matches_the_map(self):
        evs = [e for e in (
            classify("US expands semiconductor export controls on AI chips",
                     published=iso(1)),
            classify("OPEC+ agrees to production cuts", published=iso(1)),
        ) if e is not None]
        assert len(evs) == 2
        p = sector_pressure(evs)
        assert p[TECHNOLOGY] < 0
        assert p[ENERGY] > 0


@pytest.mark.unit
class TestConflictLatching:
    """A sector two rules genuinely disputed must stay unsigned.

    The first fix made zero ("exposed, sign unknown") stop cancelling a signed
    impact. That reintroduced a subtler bug: the merge could no longer tell
    that zero from a zero meaning "two rows conflicted", so a third rule
    overwrote a settled conflict and the answer depended on rule order.
    """

    HEADLINES = [
        "Powell defends rate cut, says rate hikes possible if inflation accelerates",
        "Fed cuts rates but signals rate hikes if inflation accelerates",
        "Rate cut now, rate hikes later as tariffs push prices up",
    ]

    @pytest.mark.parametrize("headline", HEADLINES)
    def test_a_two_sided_headline_leaves_the_disputed_sector_unsigned(self, headline):
        ev = classify(headline)
        assert ev is not None
        for sector in ("Real Estate", "Technology"):
            if sector in ev.sector_impact:
                assert ev.sector_impact[sector] == 0, (
                    f"{sector} was re-signed after a genuine conflict")

    @pytest.mark.parametrize("headline", HEADLINES)
    def test_a_disputed_sector_contributes_no_tilt(self, headline):
        ev = classify(headline)
        assert sector_pressure([ev]).get("Real Estate", 0.0) == 0.0

    def test_an_unsigned_row_still_does_not_cancel_a_signed_one(self):
        # The guard the first fix installed must survive the second.
        ev = classify("Powell says the Fed will cut rates twice this year")
        assert ev.sector_impact["Real Estate"] == 1
        assert ev.sector_impact["Technology"] == 1

    def test_a_real_two_rule_disagreement_still_resolves_to_unsigned(self):
        ev = classify("US announces sanctions and airstrikes")
        assert ev.sector_impact.get("Industrials") == 0


@pytest.mark.unit
class TestTitleCasedMayIsUsuallyTheMonth:
    """Capping a real event to 3 puts it below the trigger, so it is never read.

    The title-case branch read every capitalised "May"/"Might" as the hedge.
    A modal never opens a headline and is always followed by a bare verb; the
    month and the noun "might" are followed by nouns.
    """

    @pytest.mark.parametrize("headline", [
        "May Rate Cut Confirmed By Fed Officials",
        "May Tariffs Hit Chinese Imports",
        "May Jobs Report Shows Hiring Slowdown As Fed Cuts Rates",
        "White House Confirms May Tariff Rollout",
        "Supreme Court Rules On Tariffs, May Ruling Ends Dispute",
        "Military Might Of China Grows As Taiwan Tensions Rise",
        "Fed Cuts Rates At May Meeting",
    ])
    def test_a_real_event_is_not_capped_as_speculation(self, headline):
        ev = classify(headline)
        assert ev is not None and ev.severity >= POLICY_SEVERITY_TRIGGER, (
            f"{headline!r} was demoted below the trigger")

    @pytest.mark.parametrize("headline", [
        "Fed May Cut Rates In September",
        "Powell Says Fed Might Pause Hikes",
        "Trump May Impose New Tariffs On EU Goods",
        "Fed may cut rates in September",
    ])
    def test_a_genuine_hedge_is_still_capped(self, headline):
        ev = classify(headline)
        assert ev is None or ev.severity < POLICY_SEVERITY_TRIGGER, (
            f"{headline!r} is speculation and cleared the trigger")
