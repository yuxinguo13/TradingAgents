"""Policy and political events: the repricing a ticker feed cannot see.

Company news reprices one company. Policy reprices a whole sector at once, and
it does so to names whose own fundamentals did not change at all. A tariff
schedule, a semiconductor export rule, a drug-pricing bill, an FOMC decision, a
shutdown that stops federal contractors billing — each moves every exposed name
on the same morning. An agent watching only ticker news sees the move with no
idea where it came from, and the failure is worse than a blind spot: it will
read a policy move as company-specific, decide the thesis broke, and sell a
position that nothing was wrong with.

The second reason is the calendar. Much of policy is scheduled — FOMC dates are
published a year ahead, tariff deadlines and court dates and votes have dates
on them. That is exposure a desk can position around, and it is entirely
invisible in a feed of company headlines.

What this module is *not*: a model. The sector map below is a table of
directional heuristics written down by hand. It says a tariff is bad for
importers and an OPEC supply cut is good for producers, which is true often
enough to be worth knowing and false often enough that the sign is a prior, not
a prediction. Two honest failure modes, stated here rather than buried:

* **The sign inverts.** A rate cut lifts long-duration assets — unless it is a
  cut delivered because growth is failing, in which case everything sells off
  together. Nothing here can tell those two cuts apart.
* **The sector is too coarse.** A steel tariff hurts the carmaker and helps the
  domestic steelmaker. Under the sector labels used here that is Consumer
  Cyclical down and Basic Materials up, which the map can express — but the
  same split happens *inside* single sectors constantly, and there it cannot.

The third limitation is the one most likely to mislead. This module cannot tell
a surprise from a scheduled outcome. An FOMC decision is on the calendar and
the market has already priced a distribution over it; the headline announcing
it reads exactly like a shock. Severity here measures how much a sector's
earnings power is exposed to the policy named — never how much of it was
unpriced. Separating the two needs an economic calendar with consensus
expectations attached, which this module does not have and does not pretend to.
What the speculation filter below removes is only the *preview* coverage, not
the scheduled event itself.

Suppression matters more here than in the ticker feeds. Policy queries return a
far higher share of opinion columns, explainers and "what the Fed might do
next" than a company query does, and none of that is an event. Hedged and
column-shaped headlines are capped below the trigger threshold rather than
dropped, on the same reasoning as newsfeed's institutional-flow cap: they still
belong in a listing, they must not consume an LLM call.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# _NOISE and _fingerprint are private to newsfeed and imported deliberately.
# The 13F flow headlines newsfeed suppresses come back through these queries
# too — "defense budget" returns "Fund X Boosts Position in Lockheed Martin" —
# and a second copy of that regex here would drift out of step with the first
# the next time an aggregator invents a phrasing.
from .newsfeed import _NOISE, GOOGLE_QUERY, NewsItem, _fingerprint, fetch_rss, score

# --- feeds ------------------------------------------------------------------
# Google News queries, keyless. Phrased more narrowly than the bare topic in
# several places: "war" and "election" on their own return a firehose in which
# the market-relevant story is a rounding error, so the query carries enough
# context to be about policy rather than about the world.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "monetary": (
        "Federal Reserve decision",
        "FOMC meeting interest rates",
        "inflation CPI report",
        "Powell testimony",
    ),
    "fiscal": (
        "government shutdown Congress",
        "federal budget bill",
        "debt ceiling",
        "corporate tax legislation",
    ),
    "trade": (
        "tariffs",
        "semiconductor export controls",
        "sanctions",
        "US China trade deal",
    ),
    "regulatory": (
        "SEC rules",
        "antitrust FTC DOJ",
        "drug pricing legislation",
        "FDA policy",
    ),
    "geopolitical": (
        "OPEC oil supply",
        "military conflict escalation",
        "Taiwan Strait tensions",
        "election policy markets",
    ),
}

POLICY_CATEGORIES: tuple[str, ...] = tuple(CATEGORIES)

# Every policy story shares one fingerprint namespace. NewsMonitor namespaces
# by ticker on purpose — the same chip-rally headline is a separate story for
# MU and for NVDA — but a tariff announcement returned by both the "trade" and
# the "geopolitical" query is one event, and emitting it twice would double its
# weight in the sector tilt.
_NAMESPACE = "policy"

# --- severity ---------------------------------------------------------------
MAX_SEVERITY = 10
POLICY_SEVERITY_TRIGGER = 6      # at or above this, a policy event is worth an LLM call
_SPECULATION_CAP = 3             # hedged or column-shaped headlines cannot exceed this
_PROPOSAL_PENALTY = 2            # contemplated, not enacted

# Concrete action. These verbs separate "the rule was adopted" from "the rule
# is being discussed", which is most of the difference between a repricing and
# a news cycle.
_ACTION = re.compile(
    r"\b(imposes?|imposed|announces?|announced|signs?|signed into law|unveils?|"
    r"adopts?|adopted|approves?|approved|enacts?|passes?|passed|votes? to|"
    r"orders?|issues?|finali[sz]es?|takes? effect|effective (?:immediately|today)|"
    r"goes? into effect|(?:court|judge) rules?|struck down)\b", re.IGNORECASE)

# Hedges and column furniture. Not a quality judgement: a good column about
# what the Fed might do is still not an event that happened.
_SPECULATION = re.compile(
    r"^\s*(opinion|analysis|commentary|column|explainer|editorial|op-?ed|preview)\b|"
    r"\b(likely to|expected to|set to|poised to|braces? for|ahead of|"
    r"what to (?:expect|know|watch)|here'?s what|what it means|why (?:the|a|this)|"
    r"how (?:the|a|this)|things to know|live updates?|recap|takeaways|"
    r"analysts? (?:say|expect|see)|economists? (?:say|expect)|"
    r"traders? (?:await|brace|expect)|investors? (?:await|brace)|"
    r"markets? (?:await|brace)|\beyes\b|\bawaits?\b|"
    r"could|would|may not|explained)\b", re.IGNORECASE)

# "May" is a month and "may" is a hedge, and in a headline the capital letter is
# the only cheap way to tell them apart — so this branch is deliberately not
# case-insensitive.
_SPECULATION_LOWER = re.compile(r"\b(may|might)\b")

# Contemplated rather than done. Distinct from speculation: a proposed rule is
# a real object with a real market effect, just a smaller one than an adopted
# rule, so it is docked rather than capped.
_PROPOSED = re.compile(
    r"\b(propos(?:es|ed|al)|draft|weighs?|weighing|considers?|considering|"
    r"mulls?|floats?|reportedly|is said to|plans? to|seeks? to|"
    r"pushes? for|calls? for|urges?)\b", re.IGNORECASE)

# --- direction of the policy itself ----------------------------------------
# Applied only to rules whose pattern names an *instrument* (tariffs, export
# controls, sanctions) rather than a direction. Those rules declare their
# sector signs for the escalation reading, which is what the queries surface
# most of the time, and this flips them when a headline says the opposite.
#
# This is where the module is most often wrong, so the word lists are kept
# tight. Ambiguous verbs are left out on purpose: "drop", "block" and "cut"
# each read both ways depending on what they attach to ("prices drop",
# "FTC blocks merger"), and including them bought more wrong signs than right.
# Escalation words are only a *guard*: the declared sign already is the
# escalation reading, so a missed escalation costs nothing and a false one
# blocks a flip that should have happened. That asymmetry is why the words that
# double as instrument nouns are absent — "curbs", "bans", "restrictions" and
# "blacklist" name the policy rather than its direction, and leaving "curbs" in
# made "US eases chip export curbs" read as escalation and keep the wrong sign.
_ESCALATION = re.compile(
    r"\b(imposes?|imposed|imposing|expands?|expanded|widens?|broadens?|"
    r"tightens?|tightened|tightening|hikes?|escalat\w*|threatens?|"
    r"steps? up|cracks? down|crackdown)\b", re.IGNORECASE)
_DEESCALATION = re.compile(
    r"\b(lifts?|lifted|eases?|eased|easing|exempts?|exemptions?|waivers?|"
    r"repeals?|repealed|rolls? back|rollback|scraps?|scrapped|"
    r"suspends?|suspended|delays?|delayed|postpones?|postponed|pauses?|paused|"
    r"averts?|averted|avoided|dismisses?|dismissed|struck down|strikes? down|"
    r"withdraws?|withdrawn|backs? off|spares?|spared)\b", re.IGNORECASE)

# --- sectors ----------------------------------------------------------------
# These strings must stay byte-identical to yfinance's sector labels, because
# that is what `tradingagents.trading.screener.fetch_sectors` returns and what
# the tilt is applied against. Renaming one here does not fail loudly; it just
# stops matching, and the tilt silently becomes zero for that sector.
TECHNOLOGY = "Technology"
HEALTHCARE = "Healthcare"
FINANCIALS = "Financial Services"
CONSUMER_CYCLICAL = "Consumer Cyclical"
CONSUMER_DEFENSIVE = "Consumer Defensive"
INDUSTRIALS = "Industrials"
ENERGY = "Energy"
UTILITIES = "Utilities"
REAL_ESTATE = "Real Estate"
MATERIALS = "Basic Materials"
COMMUNICATION = "Communication Services"

ALL_SECTORS: tuple[str, ...] = (
    TECHNOLOGY, HEALTHCARE, FINANCIALS, CONSUMER_CYCLICAL, CONSUMER_DEFENSIVE,
    INDUSTRIALS, ENERGY, UTILITIES, REAL_ESTATE, MATERIALS, COMMUNICATION,
)


def _broad(sign: int) -> dict[str, int]:
    """Every sector, same sign — for policy that moves the index, not a theme.

    Used sparingly. Most policy is thematic, and marking a theme as broad is
    the easiest way to turn the tilt into noise that leans the whole book one
    way for no reason.
    """
    return dict.fromkeys(ALL_SECTORS, sign)


@dataclass(frozen=True)
class PolicyRule:
    """One row of the sector impact map.

    ``sectors`` maps a sector to +1, -1, or 0. Zero is not "no effect": it
    means *exposed, sign genuinely unknown*, which is the honest answer for a
    scheduled Fed decision or an FDA pathway reform whose winners depend on
    which side of the trade the book is on. A zero contributes nothing to the
    tilt but still puts the sector in front of the panel.
    """

    key: str
    category: str
    pattern: str
    sectors: dict[str, int]
    severity: int
    rationale: str
    # True when the pattern already names its own direction ("rate cut",
    # "OPEC output cut"), in which case headline polarity must not flip it —
    # the word "cut" would otherwise read as de-escalation and invert a sign
    # that was correct.
    fixed_sign: bool = False


# ---------------------------------------------------------------------------
# the sector impact map
#
# Directional heuristics, not modelled relationships. Each rationale says which
# mechanism is being assumed, because a sign with no mechanism behind it cannot
# be argued with when it turns out wrong.
# ---------------------------------------------------------------------------
IMPACT_RULES: tuple[PolicyRule, ...] = (

    # --- monetary ----------------------------------------------------------
    PolicyRule(
        "rate_cut", "monetary",
        r"\brate cuts?\b|cuts? (?:interest )?rates|lowers? (?:interest )?rates|"
        r"\beasing cycle\b|\bdovish\b|rate[- ]cutting",
        {REAL_ESTATE: +1, UTILITIES: +1, TECHNOLOGY: +1,
         CONSUMER_CYCLICAL: +1, FINANCIALS: 0},
        8,
        "Lower policy rates raise the present value of distant cash flows and cut "
        "the cost of leverage, which is most of what a REIT or a utility is. Banks "
        "are two-sided — cheaper funding, thinner net interest margin — so no sign. "
        "The whole row inverts if the cut is a response to a recession.",
        fixed_sign=True,
    ),
    PolicyRule(
        "rate_hike", "monetary",
        r"\brate hikes?\b|raises? (?:interest )?rates|hikes? (?:interest )?rates|"
        r"\bhawkish\b|tightening cycle",
        {REAL_ESTATE: -1, UTILITIES: -1, TECHNOLOGY: -1,
         CONSUMER_CYCLICAL: -1, FINANCIALS: 0},
        8,
        "The mirror of a cut: discount rates up, leverage dearer, duration hurt "
        "most. Financials again two-sided.",
        fixed_sign=True,
    ),
    PolicyRule(
        "inflation_hot", "monetary",
        r"inflation (?:accelerat\w*|surges?|jumps?|rises?|heats? up)|"
        r"CPI (?:comes in )?(?:hot|hotter|above|higher)|"
        r"hotter[- ]than[- ]expected inflation",
        {REAL_ESTATE: -1, UTILITIES: -1, TECHNOLOGY: -1,
         CONSUMER_CYCLICAL: -1, ENERGY: +1},
        7,
        "A hot print keeps policy tight, which is the rate-hike row again at one "
        "remove. The energy leg is the cause showing up in the measurement rather "
        "than an effect of the print, so treat it as the weaker half of the row.",
        fixed_sign=True,
    ),
    PolicyRule(
        "inflation_cool", "monetary",
        r"inflation (?:cools?|eases?|slows?|falls?|retreats?)|"
        r"CPI (?:cools?|eases?|below|softer)|disinflation",
        {REAL_ESTATE: +1, UTILITIES: +1, TECHNOLOGY: +1, CONSUMER_CYCLICAL: +1},
        6,
        "A cool print buys the Fed room to ease, which is the rate-cut row at one "
        "remove.",
        fixed_sign=True,
    ),
    PolicyRule(
        "fomc", "monetary",
        r"\bFOMC\b|Federal Reserve (?:meeting|decision|statement)|"
        r"Fed (?:decision|meeting|minutes|statement)|\bPowell\b|"
        r"Jackson Hole|dot plot",
        {FINANCIALS: 0, REAL_ESTATE: 0, UTILITIES: 0, TECHNOLOGY: 0},
        7,
        "Names the exposure without a sign. The decision's direction is the entire "
        "content of the event and a headline announcing the meeting does not carry "
        "it. This is also the clearest case of a scheduled event the module cannot "
        "distinguish from a surprise.",
        fixed_sign=True,
    ),

    # --- fiscal ------------------------------------------------------------
    PolicyRule(
        "shutdown", "fiscal",
        r"government shutdown|shutdown (?:looms|deadline|begins|starts)|"
        r"continuing resolution|federal furlough",
        {INDUSTRIALS: -1, TECHNOLOGY: -1},
        6,
        "Federal contractors stop billing: defence primes and government IT. "
        "Deliberately narrow — past shutdowns moved the broad index very little, "
        "and marking this one broad would swamp the tilt for a small effect.",
    ),
    PolicyRule(
        "debt_ceiling", "fiscal",
        r"debt ceiling|debt limit|(?:sovereign|Treasury|US) default|"
        r"default on (?:its|the) debt",
        _broad(-1),
        7,
        "One of the few genuinely broad rows: a real default reprices the "
        "risk-free rate itself, which is the input to every valuation. Every "
        "episode so far has resolved, which is why the sign flips on 'averted'.",
    ),
    PolicyRule(
        "corp_tax_up", "fiscal",
        r"corporate tax (?:hike|increase|rise)|raises? (?:the )?corporate tax|"
        r"higher corporate tax|windfall (?:profits? )?tax",
        _broad(-1),
        7,
        "Tax rate straight through to net income for every domestic earner, so "
        "broad by construction rather than by theme.",
        fixed_sign=True,
    ),
    PolicyRule(
        "corp_tax_down", "fiscal",
        r"corporate tax cuts?|cuts? (?:the )?corporate tax|"
        r"lower corporate tax|tax reform bill",
        _broad(+1),
        7,
        "The mirror of the row above.",
        fixed_sign=True,
    ),
    PolicyRule(
        "fiscal_spend", "fiscal",
        r"infrastructure (?:bill|package|spending|law)|"
        r"defen[cs]e (?:budget|spending) (?:increase|boost|hike|rises?)|"
        r"stimulus (?:package|bill|checks)",
        {INDUSTRIALS: +1, MATERIALS: +1},
        5,
        "Appropriated money lands as orders at the primes and as demand for steel, "
        "cement and copper. Slow — the backlog builds over years, so this is a "
        "reason to hold rather than a reason to enter.",
        fixed_sign=True,
    ),

    # --- trade -------------------------------------------------------------
    PolicyRule(
        "chip_controls", "trade",
        r"(?:chip|semiconductor|AI chip)s?\s+(?:export|sales?)\s+"
        r"(?:control|curb|ban|restrict)\w*|"
        r"export controls? on (?:chips|semiconductors|AI)|"
        r"entity list|chip (?:curbs|restrictions|export rules)",
        {TECHNOLOGY: -1},
        8,
        "An export rule removes a market that is already booked in guidance. The "
        "affected revenue is concentrated in a handful of names, so the sector-wide "
        "sign overstates the effect for everyone who does not sell into China.",
    ),
    PolicyRule(
        "tariffs", "trade",
        r"\btariffs?\b|import dut(?:y|ies)|section 301|trade war|"
        r"customs dut(?:y|ies)",
        {CONSUMER_CYCLICAL: -1, INDUSTRIALS: -1, CONSUMER_DEFENSIVE: -1,
         TECHNOLOGY: -1, MATERIALS: -1},
        8,
        "A tariff is a cost on imported input or finished goods, paid by the "
        "importer. Note the honest limit: a steel tariff hurts the carmaker and "
        "helps the domestic steelmaker, and 'Basic Materials down' is wrong for "
        "exactly the producers the tariff protects.",
    ),
    PolicyRule(
        "sanctions", "trade",
        r"\bsanctions?\b|sanctioned|\bembargo\b|blacklists?\b",
        {ENERGY: +1, FINANCIALS: -1, INDUSTRIALS: -1},
        7,
        "Sanctions take supply out of the market, which lifts the price the "
        "unsanctioned producers receive. Banks and exporters carry the compliance "
        "cost and lose the business.",
    ),
    PolicyRule(
        "trade_deal", "trade",
        r"trade (?:deal|pact|agreement|truce)|tariff (?:truce|rollback|deal)|"
        r"trade talks? (?:progress|breakthrough)",
        {CONSUMER_CYCLICAL: +1, INDUSTRIALS: +1, TECHNOLOGY: +1, MATERIALS: +1},
        7,
        "The tariff row unwound. Fixed sign because the pattern names the good "
        "outcome; a deal collapsing is the separate row below.",
        fixed_sign=True,
    ),
    PolicyRule(
        "trade_breakdown", "trade",
        r"trade (?:talks?|negotiations?) (?:collapse|stall|break ?down|fail)|"
        r"walks? (?:away|out) of (?:the )?(?:trade )?talks",
        {CONSUMER_CYCLICAL: -1, INDUSTRIALS: -1, TECHNOLOGY: -1, MATERIALS: -1},
        7,
        "Talks failing puts the tariff schedule back on the table.",
        fixed_sign=True,
    ),

    # --- regulatory --------------------------------------------------------
    PolicyRule(
        "antitrust", "regulatory",
        r"\bantitrust\b|\bFTC\b|monopol\w*|break ?-?up of (?:Google|Amazon|Apple|"
        r"Meta|Microsoft|Big Tech)|DOJ (?:sues|lawsuit|complaint|case)|"
        r"competition (?:probe|case|authority)",
        {TECHNOLOGY: -1, COMMUNICATION: -1, CONSUMER_CYCLICAL: -1},
        6,
        "Three sectors for one theme, because under this taxonomy Alphabet and "
        "Meta are Communication Services, Amazon is Consumer Cyclical, and only "
        "Apple and Microsoft are Technology. 'Big Tech antitrust' is not a "
        "one-sector event however it reads.",
    ),
    PolicyRule(
        "drug_pricing", "regulatory",
        r"drug pric\w*|price negotiation|Medicare (?:drug|negotiat\w*)|"
        r"insulin (?:price|cap)|pharmaceutical pricing|pill penalty",
        {HEALTHCARE: -1},
        7,
        "Negotiated prices come straight out of gross margin on the drugs that "
        "earn it. Hits large-cap pharma; means little to a device maker or an "
        "insurer, both of which sit in the same sector label.",
    ),
    PolicyRule(
        "health_programs", "regulatory",
        r"Medicaid cuts?|Medicare cuts?|ACA subsid\w*|"
        r"reimbursement (?:cut|rate)s?|coverage mandate",
        {HEALTHCARE: -1},
        6,
        "Programme money is revenue for hospitals, insurers and providers. The "
        "sign is for the payers-and-providers half of the sector, not for pharma.",
    ),
    PolicyRule(
        "fda_policy", "regulatory",
        r"\bFDA\b (?:policy|guidance|reform|user fee|commissioner|overhaul)|"
        r"accelerated approval|drug approval (?:pathway|process)",
        {HEALTHCARE: 0},
        5,
        "Exposed, sign unknown. A faster pathway helps whoever is trying to get a "
        "drug through and hurts whoever already has one protected. Which of those "
        "the book holds is not in the headline.",
        fixed_sign=True,
    ),
    PolicyRule(
        "sec_rules", "regulatory",
        r"\bSEC\b (?:rules?|proposals?|adopts?|approves?)|climate disclosure|"
        r"market structure rules?|payment for order flow|short sale rules?",
        {FINANCIALS: -1},
        5,
        "Compliance cost on brokers and asset managers. Real, and small next to "
        "what rates do to the same names — hence the low severity.",
    ),
    PolicyRule(
        "bank_capital", "regulatory",
        r"\bBasel\b|capital requirements?|stress tests?|bank regulat\w*|"
        r"liquidity rules?",
        {FINANCIALS: -1},
        6,
        "Higher required capital is less balance sheet available for buybacks and "
        "lending, which is where bank returns come from.",
    ),
    PolicyRule(
        "crypto_rules", "regulatory",
        r"crypto (?:regulation|rules?|bill|framework)|stablecoin (?:bill|rules?)|"
        r"digital asset (?:rules?|framework|legislation)",
        {FINANCIALS: 0},
        5,
        "Exposed, sign unknown, and the reason is worth stating: clear rules "
        "legitimise the exchanges and threaten the deposit franchises they take "
        "float from. Both are Financial Services.",
        fixed_sign=True,
    ),
    PolicyRule(
        "energy_permits", "regulatory",
        r"drilling (?:permits?|leases?) (?:approved|expanded|resumed)|"
        r"opens? (?:up )?(?:federal )?(?:land|waters) (?:to|for) drilling|"
        r"LNG export (?:approval|permit)|pipeline approved",
        {ENERGY: +1},
        5,
        "Permission to produce is the binding constraint for US producers more "
        "often than price is.",
        fixed_sign=True,
    ),
    PolicyRule(
        "energy_restrictions", "regulatory",
        r"drilling ban|halts? (?:new )?(?:drilling|leasing)|"
        r"LNG (?:export )?(?:pause|ban)|pipeline (?:blocked|cancel\w*|rejected)",
        {ENERGY: -1},
        5,
        "The mirror of the row above.",
        fixed_sign=True,
    ),
    PolicyRule(
        "ev_and_emissions", "regulatory",
        r"EV tax credits?|electric vehicle (?:tax credit|subsid\w*|mandate)|"
        r"clean energy (?:subsid\w*|credits?)|emissions (?:rules?|standards?)|"
        r"fuel economy standards?",
        {CONSUMER_CYCLICAL: 0, UTILITIES: 0, ENERGY: 0},
        5,
        "Exposed, no net sign. Repealing an EV credit hurts the EV maker and helps "
        "the legacy carmaker; both are Consumer Cyclical, and they cancel at the "
        "sector level even though the single-name effect is large.",
        fixed_sign=True,
    ),

    # --- geopolitical ------------------------------------------------------
    PolicyRule(
        "opec_cut", "geopolitical",
        r"OPEC\+?[^.]{0,40}(?:cuts?|reduc\w*|curbs?|trims?)|"
        r"(?:production|output|supply) cuts?",
        {ENERGY: +1, INDUSTRIALS: -1, CONSUMER_CYCLICAL: -1},
        7,
        "Less crude lifts the price the producers sell at and taxes everyone who "
        "burns it — airlines and freight are Industrials, and the fuel bill comes "
        "out of the discretionary budget that Consumer Cyclical lives on.",
        fixed_sign=True,
    ),
    PolicyRule(
        "opec_raise", "geopolitical",
        r"OPEC\+?[^.]{0,40}(?:raises?|increases?|boosts?|hikes?|unwinds?)|"
        r"(?:production|output) (?:increase|hike|boost)",
        {ENERGY: -1, INDUSTRIALS: +1, CONSUMER_CYCLICAL: +1},
        7,
        "The mirror of the row above.",
        fixed_sign=True,
    ),
    PolicyRule(
        # "trade war" belongs to the tariff row, not this one, so the lookbehind
        # keeps it out. Fixed width, which is all Python allows.
        "conflict", "geopolitical",
        r"(?<!trade )\bwar\b|\binvasion\b|invades?|"
        r"military (?:strike|offensive|operation)|missile (?:strike|attack)|"
        r"airstrikes?|conflict escalat\w*",
        {ENERGY: +1, INDUSTRIALS: +1, MATERIALS: +1, CONSUMER_CYCLICAL: -1},
        7,
        "Shooting wars price in a supply risk to energy and metals and a demand "
        "pull for defence. Industrials holds both the primes that gain and the "
        "airlines that the same oil price hurts, so that leg is the least "
        "trustworthy in the table.",
        fixed_sign=True,
    ),
    PolicyRule(
        "ceasefire", "geopolitical",
        r"ceasefire|peace (?:deal|agreement|plan|talks)|\btruce\b|"
        r"(?:war|conflict) ends?",
        {ENERGY: -1, INDUSTRIALS: -1, MATERIALS: -1, CONSUMER_CYCLICAL: +1},
        6,
        "The risk premium coming back out. Lower severity than the conflict row "
        "because premia are put on faster than they are taken off.",
        fixed_sign=True,
    ),
    PolicyRule(
        "taiwan", "geopolitical",
        r"Taiwan (?:strait|tension|invasion|blockade|drills|conflict|crisis)|"
        r"(?:tensions?|blockade|drills?|exercises) (?:near|around|over) Taiwan|"
        r"cross-strait",
        {TECHNOLOGY: -1},
        7,
        "Leading-edge foundry capacity sits on one island, so a threat to it is a "
        "supply question for every fabless designer regardless of end market. The "
        "pattern requires a risk word: 'Taiwan' alone is not an event.",
    ),
    PolicyRule(
        "election", "geopolitical",
        r"\belections?\b|presidential (?:race|campaign)|\bmidterms?\b|"
        r"\bballot\b|\breferendum\b",
        {},
        5,
        "No sectors, on purpose. The sector effect of an election is a function of "
        "the result, and the result is not in a headline about the campaign. "
        "Surfaced so the panel knows the date is coming, not to tilt anything.",
        fixed_sign=True,
    ),
)

_RULES_RE: tuple[tuple[PolicyRule, re.Pattern[str]], ...] = tuple(
    (r, re.compile(r.pattern, re.IGNORECASE)) for r in IMPACT_RULES
)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

@dataclass
class PolicyEvent:
    category: str
    headline: str
    url: str
    published: str                 # ISO8601 UTC, best effort
    severity: int = 0              # 0-10
    # bullish | bearish | neutral, where neutral covers both "no signed
    # exposure" and "the affected sectors point opposite ways".
    direction: str = "neutral"
    # Sector -> +1 / -1 / 0, where 0 means exposed with no defensible sign.
    sector_impact: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    source: str = ""
    fingerprint: str = ""

    @property
    def affected_sectors(self) -> list[str]:
        """The sector names, in most-exposed-first order.

        A view over ``sector_impact`` rather than a second field: the brief
        reads the names and the tilt reads the signs, and keeping them as one
        object is what stops the two ever disagreeing about which sectors an
        event touches.
        """
        return list(self.sector_impact)

    def age_hours(self, now: datetime | None = None) -> float:
        # Same parse and the same fallbacks as a news item, by construction:
        # one implementation of "how old is this timestamp" so the two feeds
        # cannot disagree about what counts as fresh.
        return NewsItem("", self.headline, self.url, self.source,
                        self.published).age_hours(now)


def _polarity(headline: str) -> int:
    """+1 escalation, -1 rollback, 0 when the headline does not say.

    Zero is the common case and it is fine: rule signs are declared for the
    escalation reading, which is what these queries surface most of the time.
    """
    esc, de = bool(_ESCALATION.search(headline)), bool(_DEESCALATION.search(headline))
    if esc == de:                  # both or neither — no usable claim
        return 0
    return 1 if esc else -1


def classify(headline: str, url: str = "", published: str = "",
             source: str = "") -> PolicyEvent | None:
    """Turn a headline into a :class:`PolicyEvent`, or None if it is not one.

    None is the overwhelmingly common answer and the point of the function.
    Google returns whatever it likes for "sanctions"; only a headline that
    matches a row of the impact map is something this module has an opinion
    about.
    """
    if not headline or _NOISE.search(headline):
        return None

    matched = [rule for rule, rx in _RULES_RE if rx.search(headline)]
    if not matched:
        return None

    # The event's category comes from the rule that fired hardest, not from the
    # query that found the story. A tariff headline returned by the
    # "geopolitical" query is a trade event, and filing it under the query
    # would scatter one theme across the brief.
    matched.sort(key=lambda r: -r.severity)
    lead = matched[0]

    pol = _polarity(headline)
    impact: dict[str, int] = {}
    for rule in matched:
        flip = -1 if (pol < 0 and not rule.fixed_sign) else 1
        for sector, sign in rule.sectors.items():
            signed = sign * flip
            if sector in impact and impact[sector] != signed:
                # Two rows disagree about this sector — a sanctions row and a
                # conflict row on the same headline, say. That is a real
                # two-sided exposure, not an error to resolve by picking one.
                impact[sector] = 0
            else:
                impact[sector] = signed

    severity = lead.severity
    if _ACTION.search(headline):
        severity += 1
    # newsfeed's materiality table already knows what a concrete legal or
    # regulatory action reads like — a probe, a suit, an investigation. Where
    # it fires hard on a policy headline, the headline describes something that
    # has happened rather than something being contemplated.
    if score(headline)[0] >= 7:
        severity += 1
    if _PROPOSED.search(headline):
        severity -= _PROPOSAL_PENALTY
    severity = max(1, min(MAX_SEVERITY, severity))
    if _SPECULATION.search(headline) or _SPECULATION_LOWER.search(headline):
        severity = min(severity, _SPECULATION_CAP)

    # Only an event whose affected sectors all point the same way gets a
    # direction. An OPEC supply cut is bullish for the producers and bearish for
    # everyone who burns the barrel; netting that to one word would be a claim
    # about the index, which this table cannot support. The sector line carries
    # the detail either way.
    # Zeros are excluded rather than counted: an "exposed, sign unknown" sector
    # is not evidence against the direction the signed ones agree on.
    signs = {v for v in impact.values() if v}
    direction = ("bullish" if signs == {1}
                 else "bearish" if signs == {-1}
                 else "neutral")

    # Rationales in severity order, deduplicated: two monetary rows firing on
    # one headline usually say the same thing twice.
    seen_r: list[str] = []
    for rule in matched:
        if rule.rationale not in seen_r:
            seen_r.append(rule.rationale)

    return PolicyEvent(
        category=lead.category, headline=headline, url=url,
        published=published or datetime.now(timezone.utc).isoformat(),
        severity=severity, direction=direction, sector_impact=impact,
        rationale=" ".join(seen_r)[:600], source=source,
        fingerprint=_fingerprint(headline, _NAMESPACE),
    )


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------

class PolicyMonitor:
    """Stateful poller over the policy feeds. ``poll`` returns new events only.

    Same novelty discipline as :class:`~.newsfeed.NewsMonitor`, and for the same
    reason: without a persistent seen set, every restart replays the week's
    policy coverage as if it had just broken, and a tariff announced on Monday
    tilts the book again on Thursday.

    Two differences from the news monitor, both deliberate:

    * The seen set has one namespace, so a story returned by two category
      queries is one event (see ``_NAMESPACE``).
    * Age filtering happens here rather than upstream. ``brain.triggers``
      filters news by age because a ticker feed is roughly chronological;
      a Google policy *search* is not, and routinely returns last year's tariff
      coverage alongside this morning's, so the cut has to be at the source.
    """

    def __init__(self, state_path: Path | None = None, retain_hours: int = 96,
                 max_age_hours: int = 48):
        self.state_path = Path(state_path) if state_path else (
            Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
            / "policy_seen.json"
        )
        # Longer than the news retention: a policy story is re-syndicated for
        # days as each outlet writes its own version, and the fingerprint has
        # to still be there when the fourth one arrives.
        self.retain_hours = retain_hours
        self.max_age_hours = max_age_hours
        self.seen: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self.retain_hours)).isoformat()
        self.seen = {k: v for k, v in self.seen.items() if v >= cutoff}
        # Atomic: the loop can be killed at any moment, and a half-written seen
        # set is read back as an empty one, which replays the whole feed.
        with suppress(Exception):
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.seen), encoding="utf-8")
            os.replace(tmp, self.state_path)

    # --- polling ------------------------------------------------------------

    def _collect(self, urls: list[str]) -> list[PolicyEvent]:
        out: list[PolicyEvent] = []
        now = datetime.now(timezone.utc)
        stamp = now.isoformat()
        for url in urls:
            for raw in fetch_rss(url):
                title = raw.get("title", "")
                fp = _fingerprint(title, _NAMESPACE)
                if fp in self.seen:
                    continue
                # Marked before classification, and marked even when nothing
                # matches: classification is deterministic, so a headline that
                # is not an event today will not be one tomorrow, and
                # re-deciding that every cycle is pure work.
                self.seen[fp] = stamp
                try:
                    ev = classify(title, raw.get("link", ""),
                                  raw.get("published", ""), raw.get("source", ""))
                except Exception:
                    # A regex table must never take down a loop meant to run
                    # unattended for weeks. An unclassifiable headline is not
                    # an event.
                    continue
                if ev is None or ev.age_hours(now) > self.max_age_hours:
                    continue
                out.append(ev)
        return out

    def poll_category(self, category: str) -> list[PolicyEvent]:
        queries = CATEGORIES.get(category, ())
        return self._collect([GOOGLE_QUERY.format(q=urllib.parse.quote(q))
                              for q in queries])

    def poll(self, categories: list[str] | None = None,
             pause: float = 0.4) -> list[PolicyEvent]:
        """Poll every category; returns new events, most severe first.

        ``pause`` throttles between categories for the reason NewsMonitor
        throttles between tickers: Google answers a client that hammers it with
        an empty body, and an empty body is indistinguishable from "no news".
        """
        out: list[PolicyEvent] = []
        for cat in (categories or list(CATEGORIES)):
            try:
                out += self.poll_category(cat)
            except Exception:
                # One unreachable category degrades to "nothing from this
                # source", never to a dead loop.
                continue
            time.sleep(pause)
        self._save()
        return sorted(out, key=lambda e: -e.severity)

    def prime(self, categories: list[str] | None = None) -> int:
        """Mark everything currently in the feeds as seen, without acting.

        Called on first start so the desk does not wake up, find a fortnight of
        tariff coverage, and tilt the whole book on it.
        """
        return len(self.poll(categories))


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

_CAVEAT = ("_Sector effects above are directional heuristics, not modelled "
           "relationships. The sign can invert — a rate cut is bullish for "
           "duration unless it signals a recession — and a scheduled event "
           "may already be priced._")


def policy_brief(events: list[PolicyEvent], per_category: int = 4,
                 min_severity: int = 1) -> str:
    """A compact markdown block for an LLM evidence pack.

    Grouped by category and most severe first inside each, because a panel
    reading this needs "what is happening to the sectors I hold" and not a
    chronological wire. The caveat is printed with the events rather than kept
    in this docstring: the model that reads the brief is the one that needs to
    know how much to trust it.
    """
    live = [e for e in events if e.severity >= min_severity]
    if not live:
        return "(no policy or political events)"

    by_cat: dict[str, list[PolicyEvent]] = {}
    for e in live:
        by_cat.setdefault(e.category, []).append(e)
    for evs in by_cat.values():
        evs.sort(key=lambda e: -e.severity)

    lines = ["## Policy and political backdrop"]
    for cat in sorted(by_cat, key=lambda c: -by_cat[c][0].severity):
        evs = by_cat[cat]
        lines += ["", f"### {cat} ({len(evs)})"]
        for e in evs[:per_category]:
            src = f", {e.source}" if e.source else ""
            lines.append(f"- [{e.severity}/{e.direction}] {e.headline[:140]} "
                         f"({e.age_hours():.0f}h ago{src})")
            if e.sector_impact:
                # A zero prints as "?" rather than being dropped: "exposed,
                # direction unknown" is information the panel should act on by
                # widening its uncertainty, not by ignoring the sector.
                sects = ", ".join(
                    f"{s}{'+' if v > 0 else '-' if v < 0 else '?'}"
                    for s, v in e.sector_impact.items())
                lines.append(f"  sectors: {sects}")
        if len(evs) > per_category:
            lines.append(f"  ... and {len(evs) - per_category} more in {cat}")

    lines += ["", _CAVEAT]
    return "\n".join(lines)


# Two maximum-severity events pointing the same way at one sector produce a
# tilt of about 0.5. Chosen so a single headline cannot saturate the scale and
# a genuine pile-on still can.
PRESSURE_SATURATION = 20.0


def sector_pressure(events: list[PolicyEvent]) -> dict[str, float]:
    """Net directional pressure per sector, in [-1, 1].

    **This is a tilt, not a signal.** It is meant to reorder candidates that
    have already earned their place on a quantitative screen — to prefer the
    industrial over the semiconductor on the morning a chip export rule lands,
    when both rank alike on the factors. It is not a reason to enter a name, it
    is not a reason to exit one, and it carries no view on any individual
    company. Nothing in it is calibrated: the numbers come from a hand-written
    severity table, so their ordering is meaningful and their magnitude is not.

    A sector present with value ``0.0`` is not the same as a sector that is
    absent. Zero means the policy touched it with no defensible direction, or
    with two events that cancelled — both are worth knowing, and both should
    widen uncertainty rather than be read as "unaffected".
    """
    raw: dict[str, float] = {}
    for e in events:
        for sector, sign in e.sector_impact.items():
            raw[sector] = raw.get(sector, 0.0) + sign * e.severity
    # Soft saturation rather than a hard clip, so the tenth tariff headline of
    # the morning adds less than the first without any of them being discarded.
    # Averaging instead would let one severity-9 event be diluted to nothing by
    # eight severity-1 ones, which is the wrong way round.
    return {s: v / (PRESSURE_SATURATION + abs(v)) for s, v in raw.items()}
