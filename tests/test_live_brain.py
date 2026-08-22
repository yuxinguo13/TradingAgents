"""Triggers, evidence assembly, and the panel's consensus arithmetic.

No network and no LLM: snapshots are constructed directly and the panel's
votes are stubbed, so these assert the decision *logic* rather than a model's
opinion on a given day.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.live.brain import (
    Panel, PanelResult, Snapshot, Trigger, Vote, build_evidence, triggers,
)
from tradingagents.live.investopedia import Account, Holding
from tradingagents.live.newsfeed import NewsItem
from tradingagents.live.personas import PANEL
from tradingagents.live.secretary import Secretary, RiskLimits


def snap(**kw) -> Snapshot:
    base = dict(symbol="MU", price=100.0, prev_close=100.0, change_pct=0.0,
                atr_pct=0.03, move_atrs=0.0, rsi14=50.0, sma20=98.0, sma50=95.0,
                sma200=90.0, vol_ratio=1.0, ret_1m=0.02, ret_3m=0.05,
                off_high_52w=-0.05, ok=True)
    base.update(kw)
    return Snapshot(**base)


def acct(holdings=()):
    return Account(account_value=100_000.0, cash=50_000.0, buying_power=50_000.0,
                   holdings=list(holdings))


def news(materiality, ticker="MU", title="Something happened", age_hours=1.0):
    # Relative to now: the trigger path filters on age, so a hardcoded date
    # would quietly stop testing what it claims to once it went stale.
    published = (datetime.now(timezone.utc)
                 - timedelta(hours=age_hours)).isoformat()
    return NewsItem(ticker=ticker, title=title, link="", source="test",
                    published=published, materiality=materiality)


@pytest.mark.unit
class TestTriggers:
    def test_quiet_symbol_fires_nothing(self):
        assert triggers("MU", snap(), [], acct()) == []

    def test_material_news_fires(self):
        t = triggers("MU", snap(), [news(9)], acct())
        assert [x.kind for x in t] == ["news"]
        assert t[0].urgency == 3

    def test_low_materiality_news_ignored(self):
        assert triggers("MU", snap(), [news(3)], acct()) == []

    def test_news_for_another_symbol_ignored(self):
        assert triggers("MU", snap(), [news(10, ticker="AAPL")], acct()) == []

    def test_stale_news_does_not_fire(self):
        # Google News backfills months of coverage; "unseen by this process" is
        # not the same as "just happened".
        assert triggers("MU", snap(), [news(10, age_hours=200)], acct()) == []

    def test_news_triggers_are_capped_per_symbol(self):
        flood = [news(8, title=f"ACME earnings story {i}") for i in range(30)]
        t = [x for x in triggers("MU", snap(), flood, acct()) if x.kind == "news"]
        assert len(t) == 3

    def test_most_material_news_is_the_one_that_fires(self):
        items = [news(7, title="minor"), news(11, title="Trading halted"),
                 news(8, title="middling")]
        t = [x for x in triggers("MU", snap(), items, acct()) if x.kind == "news"]
        assert "Trading halted" in t[0].detail

    def test_large_move_fires(self):
        t = triggers("MU", snap(move_atrs=2.5, change_pct=0.07), [], acct())
        assert any(x.kind == "price_move" and x.urgency == 2 for x in t)

    def test_volume_surge_fires(self):
        assert any(x.kind == "volume"
                   for x in triggers("MU", snap(vol_ratio=3.0), [], acct()))

    def test_stop_loss_on_held_position(self):
        a = acct([Holding("MU", 100, 120.0, 100.0, 10_000.0, -2_000.0)])
        t = triggers("MU", snap(), [], a)
        assert any(x.kind == "stop_loss" and x.urgency == 3 for x in t)

    def test_take_profit_on_held_position(self):
        a = acct([Holding("MU", 100, 70.0, 100.0, 10_000.0, 3_000.0)])
        assert any(x.kind == "take_profit" for x in triggers("MU", snap(), [], a))

    def test_trend_break_only_applies_to_held_names(self):
        broken = snap(price=90.0, sma50=95.0, ret_1m=-0.09)
        assert not any(x.kind == "trend_break"
                       for x in triggers("MU", broken, [], acct()))
        a = acct([Holding("MU", 10, 95.0, 90.0, 900.0, -50.0)])
        assert any(x.kind == "trend_break" for x in triggers("MU", broken, [], a))

    def test_screen_entry_only_when_unheld_and_highly_ranked(self):
        assert any(x.kind == "screen_entry"
                   for x in triggers("MU", snap(), [], acct(), screen_rank=3))
        assert not any(x.kind == "screen_entry"
                       for x in triggers("MU", snap(), [], acct(), screen_rank=40))
        a = acct([Holding("MU", 10, 95.0, 100.0, 1_000.0, 50.0)])
        assert not any(x.kind == "screen_entry"
                       for x in triggers("MU", snap(), [], a, screen_rank=3))

    def test_news_still_fires_when_price_data_is_missing(self):
        # A data outage must not silence the news path — that is exactly when
        # something is happening.
        dead = Snapshot(symbol="MU", ok=False, error="no history")
        assert [x.kind for x in triggers("MU", dead, [news(10)], acct())] == ["news"]


@pytest.mark.unit
class TestEvidence:
    def test_states_position_when_held(self):
        a = acct([Holding("MU", 100, 90.0, 100.0, 10_000.0, 1_000.0)])
        ev = build_evidence("MU", snap(), [], a, [Trigger("MU", "news", "x")])
        assert "Position: 100 shares" in ev and "10.0% of account" in ev

    def test_states_absence_of_position(self):
        assert "No position in MU" in build_evidence("MU", snap(), [], acct(), [])

    def test_includes_trigger_reasons(self):
        ev = build_evidence("MU", snap(), [], acct(),
                            [Trigger("MU", "stop_loss", "down 9%")])
        assert "stop_loss: down 9%" in ev

    def test_says_so_when_there_is_no_news(self):
        assert "nothing new" in build_evidence("MU", snap(), [], acct(), [])

    def test_missing_price_data_is_reported_not_hidden(self):
        ev = build_evidence("MU", Snapshot(symbol="MU", ok=False, error="boom"),
                            [], acct(), [])
        assert "price data unavailable" in ev


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------

class StubPanel(Panel):
    """Panel with scripted votes and a no-op risk officer."""

    def __init__(self, spec, veto=False, scale=1.0):
        super().__init__(llm=None, secretary=Secretary(limits=RiskLimits()))
        self.spec = spec
        self._veto, self._scale = veto, scale

    def _vote(self, persona, evidence):
        action, qty, conf = self.spec[persona.name]
        return Vote(persona=persona.name, action=action, quantity=qty,
                    confidence=conf)

    def _risk_review(self, order, evidence, price):
        return self._veto, "stubbed", self._scale


NAMES = [p.name for p in PANEL]


def deliberate(spec, **kw):
    return StubPanel(spec, **kw).deliberate("MU", "evidence", acct(), 100.0)


@pytest.mark.unit
class TestConsensus:
    def test_unanimous_buy_passes(self):
        r = deliberate({n: ("Buy", 50, 0.8) for n in NAMES})
        assert r.consensus == "Buy" and r.order.quantity == 50

    def test_split_vote_holds(self):
        # Two Hold seats outweigh two Buy seats; the old confidence-weighted
        # tally let the confident minority win this.
        r = deliberate({NAMES[0]: ("Hold", 0, 0.0), NAMES[1]: ("Buy", 55, 0.9),
                        NAMES[2]: ("Hold", 0, 0.0), NAMES[3]: ("Buy", 50, 0.9)})
        assert r.consensus == "Hold" and r.order is None

    def test_agreement_without_conviction_holds(self):
        r = deliberate({NAMES[0]: ("Hold", 0, 0.0), NAMES[1]: ("Buy", 50, 0.4),
                        NAMES[2]: ("Buy", 40, 0.3), NAMES[3]: ("Buy", 60, 0.45)})
        assert r.consensus == "Hold" and "confidence" in r.concern

    def test_size_is_the_median_not_the_maximum(self):
        r = deliberate({NAMES[0]: ("Buy", 10, 0.8), NAMES[1]: ("Buy", 1_000, 0.9),
                        NAMES[2]: ("Buy", 20, 0.8), NAMES[3]: ("Buy", 30, 0.8)})
        assert r.order.quantity == 25          # median of 10/20/30/1000

    def test_veto_suppresses_the_trade(self):
        r = deliberate({n: ("Buy", 50, 0.9) for n in NAMES}, veto=True)
        assert r.consensus == "Hold" and r.order is None and r.veto

    def test_scale_shrinks_the_order(self):
        r = deliberate({n: ("Buy", 100, 0.9) for n in NAMES}, scale=0.5)
        assert r.order.quantity == 50

    def test_errored_personas_abstain_rather_than_hold(self):
        class Erroring(StubPanel):
            def _vote(self, persona, evidence):
                if persona.name in NAMES[:2]:
                    return Vote(persona=persona.name, error="api down")
                return super()._vote(persona, evidence)

        p = Erroring({n: ("Buy", 40, 0.8) for n in NAMES})
        r = p.deliberate("MU", "e", acct(), 100.0)
        assert r.consensus == "Buy"           # 2 live votes, both Buy

    def test_total_panel_failure_holds(self):
        class Dead(StubPanel):
            def _vote(self, persona, evidence):
                return Vote(persona=persona.name, error="api down")

        r = Dead({}).deliberate("MU", "e", acct(), 100.0)
        assert r.consensus == "Hold" and "failed to respond" in r.concern


@pytest.mark.unit
class TestTriggerPriority:
    """The LLM budget per cycle is small; it must go to real events first.

    Observed live: a +7.9% move (1.4 ATR) tied with "this name is on a
    shortlist" and lost the budget to it purely by list order.
    """

    def test_screen_entry_ranks_below_every_event(self):
        a = acct()
        screen = triggers("MU", snap(), [], a, screen_rank=1)
        move = triggers("MU", snap(move_atrs=1.4, change_pct=0.079), [], a)
        assert max(t.urgency for t in screen) < max(t.urgency for t in move)

    def test_stop_loss_outranks_everything(self):
        a = acct([Holding("MU", 100, 120.0, 100.0, 10_000.0, -2_000.0)])
        t = triggers("MU", snap(), [], a)
        assert max(x.urgency for x in t) == 3

    def test_material_news_outranks_a_screen_entry(self):
        a = acct()
        news_t = triggers("MU", snap(), [news(9)], a, screen_rank=1)
        assert max(t.urgency for t in news_t if t.kind == "news") > \
               max(t.urgency for t in news_t if t.kind == "screen_entry")
