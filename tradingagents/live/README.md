# Live desk

Two ways to use it, sharing one engine.

| | What it is | When it runs |
|---|---|---|
| **Live agent** | Autonomous loop. Watches, decides, places orders. | Continuously, all session |
| **Daily advisor** | One considered buy/sell list for you to act on. | Once a day |

Both read the same market data, the same news, and the same reasoning panel.
The difference is who pulls the trigger.

---

## How it works

```
                    ┌─ screener ──── whole exchange, ranked nightly
                    ├─ newsfeed ──── ticker news, 24h, novelty-tracked
   DATA ────────────┼─ policy ────── Fed / tariffs / regulation / geopolitics
                    ├─ technicals ── price, trend, ATR, volume
                    └─ broker ────── account, positions, quotes
                              │
                              ▼
   TRIGGERS ─── deterministic arithmetic. Free. Answers one question:
                has anything happened that could change a position?
                              │  (usually: no)
                              ▼
   EVIDENCE ─── compact pack: position, trend, fresh news, policy backdrop
                              │
                              ▼
   PANEL ────── 4 analysts with different mandates vote independently
                Conservative · Aggressive · Balanced · Growth
                              │
                              ▼
   RISK OFFICER  reviews the consensus. May veto or scale.
                              │
                              ▼
   SECRETARY ── hard limits. Not prompts — code. Cannot be argued with.
                              │
                              ▼
   VENUE ────── alpaca | paper | investopedia
```

### Why it's split this way

**Triggers are rules, decisions are judgement.** An agent that asks an LLM about
every name every cycle burns its quota by 10am concluding "nothing changed". So
arithmetic decides *when* to think, and the panel decides *what* to do. On a
quiet cycle nothing fires and nothing is spent.

**The panel votes by seat, not by conviction.** Confidence sizes the trade; it
does not win the vote. Weighting the tally by confidence lets one emphatic
member outvote two calm ones — and since "Hold" is naturally reported at low
confidence, that scheme makes a panel structurally unable to decline a trade.

A trade needs a weighted majority **and** mean backer confidence ≥ 0.55.
Agreement without conviction is a mood, not a trade. Size is the **median** of
the backers' quantities, so one enthusiastic persona cannot set the position.

**The Secretary is code, not instruction.** A limit an LLM can talk itself past
is not a limit. Oversized orders are *resized*, not rejected — the view is
valid, the size wasn't. Everything else (no cash, no shares, wrong session,
kill switch) is a flat no.

---

## Venues

Selected by `TRADINGAGENTS_BROKER`. Everything above the adapter is
venue-agnostic — swapping venues changes nothing else.

| Venue | Status | Needs |
|---|---|---|
| `paper` | **works now** | nothing |
| `alpaca` | default | free paper API keys |
| `investopedia` | not recommended | a browser session |

**`paper`** — local book, real prices, no account. Fills are immediate and
complete at the quote: no slippage, no partial fills, no queue. Good for judging
whether the *reasoning* is any good; not for judging execution.

**`alpaca`** — the sanctioned API. Free paper account, keys issued instantly,
global. Two traps handled in the adapter: every number Alpaca returns is a
*string*, and `OrderSide` has only BUY/SELL — so "Sell Short" on a name you hold
long would sell the long unless `PositionIntent` disambiguates it.

**`investopedia`** — works, but its terms forbid it. People Inc. ToS §3.3(e)
bars robots and scrapers; §3.3(f) bars using site data to develop software or
train an AI system; §5.2 permits termination "for any reason or no reason". Kept
because it proves the abstraction, not because you should use it.

---

## Running the live agent

```bash
# safest first run — reasons out loud, never submits
TRADINGAGENTS_BROKER=paper python -m tradingagents.live.cli run --dry-run

# for real
TRADINGAGENTS_BROKER=paper python -m tradingagents.live.cli run

# watch only: full monitoring and triggers, no LLM, no orders
python -m tradingagents.live.cli run --no-llm
```

Useful flags: `--once` (single cycle), `--interval 120`, `--max-panels 4`
(LLM budget per cycle), `--no-screen`, `--trade-when-closed` (paper only).

**Stop it instantly, from any shell, without touching the process:**

```bash
python -m tradingagents.live.cli stop          # or: touch ~/.tradingagents/STOP
python -m tradingagents.live.cli stop --clear
```

### Cadence follows the clock, not a timer

| Session | Every | What happens |
|---|---|---|
| Regular 09:30–16:00 ET | 120s (60s in the last half hour) | full cycle |
| Pre / after hours | 10 min | news only — nothing is tradeable, but the headlines that move the open arrive now |
| Closed | ≤30 min | nightly rescan, news, sleep toward the open |

NYSE holidays and half-days are in the calendar, so it never waits for a 16:00
close that isn't coming.

### Triggers

| Trigger | Fires when | Urgency |
|---|---|---|
| `stop_loss` | held position ≥8% below cost | 3 |
| `news` | fresh headline ≥7 materiality, **<24h old** | 2–3 |
| `trend_break` | held name loses SMA50 on a negative month | 2 |
| `take_profit` | held position ≥20% above cost | 2 |
| `price_move` | ≥1.2 ATR since prior close | 1–2 |
| `volume` | ≥2× the 20-day average | 1 |
| `screen_entry` | top-15 screen name you don't own | **0** |

Screen entries rank last deliberately: they aren't news, they'll still be true
next cycle, and they must never take the budget from a stop-loss.

### Other commands

```bash
python -m tradingagents.live.cli portfolio    # account, positions, P&L
python -m tradingagents.live.cli scan         # coverage + news + triggers, no trading
python -m tradingagents.live.cli status       # clock, limits, kill switch, today's trades
python -m tradingagents.live.cli news NVDA MU
python -m tradingagents.live.cli trade NVDA buy 10 --dry-run
```

---

## Coverage: the pool and the watchlist

Two lists feed the desk, and they answer different questions.

**The qualification pool** (`~/.tradingagents/cache/screens/universe_<exchange>.json`)
is who is worth downloading. Rebuilt about once a month, it filters only on what
cannot change inside a month — a penny price, no liquidity at all, not enough
history to score. Measured on the 2026-08-25 panel: 3215 listed names → 1465
qualify, and every one of the 534 that passed the *daily* filter was inside it.

The daily filters are deliberately **not** used to build the pool. A name below
its 200-day today is exactly the name that reappears after a base; qualifying on
trend would mean the monthly refresh could never rediscover anything. 481 of the
pool's members are below their 200-day right now, and that is the point.

```bash
# force an early rebuild (a month is the default cadence)
python -c "from tradingagents.trading.screener import qualified_universe; \
           qualified_universe('2026-08-25', refresh=True)"
```

**The watchlist** (`~/.tradingagents/watchlist.json`) is who gets looked at
whatever the screen thinks. `{"NVDA": "semi", "GOOGL": "tech", ...}` — the tag is
free text and only groups the report.

It is scored on its own path, so it survives every way the universe scan can go:
cached, skipped, or failed. And it **bypasses the hard filters on the way out**.
On 2026-08-25 nine of thirty-one — AVGO, QCOM, NXPI, ON, META, TSLA, ORCL, CRWV,
AXTI — would have been filtered out of the report entirely, eight of them for
losing their 200-day. A semiconductor in a drawdown is precisely the one worth
reading that morning, so the section names each one and why it failed rather
than dropping it.

What the watchlist is not: a buy list. Watchlist names are reported, never
injected into the ranking, never sized, and never written to the recommendation
book. Following a name daily is not a reason to own it.

### Bases

An entry tagged `base` is a yardstick rather than a candidate:

```json
{"SPY": "base", "QQQM": "base", "NVDA": "semi", "GOOGL": "tech"}
```

Bases are shown first, never judged pass/fail — asking whether SPY clears a
momentum screen is a category error — and every other row is reported as an
**excess return over each of them**.

That column is what makes the section readable, because "below the 200-day"
covers two opposite situations. On 2026-08-25:

| | trend | 1M vs SPY | reading |
|---|---|---:|---|
| ORCL | below the 200-day | **+17.1%** | climbing out of a hole (3M: −24.8%) |
| NXPI | below the 200-day | **−19.7%** | simply broken |
| AMAT | passes every filter | **−10.7%** | trend looks fine, badly lagging |

Without a base, the first two look identical and the third looks healthy.

Bases are data, not code: retag the file and the comparison changes.

### A silent universe bug, for the record

`fetch_universe` used to drop every 5-letter Nasdaq symbol, on the reasoning that
the fifth letter marks warrants and units. It does — but only for some letters.
The rule removed 920 symbols to exclude 880 that the security-name filter already
caught, and took 39 real common stocks with them: **GOOGL, CMCSA, FCNCA, RYAAY**
and the entire Liberty complex. Nothing ever errored; the names simply never
appeared in a screen.

The rule now tests the suffix that actually means something — `W` warrant,
`U` unit, `R` right, `P` first preferred — so class letters (A/B/K/L) and `Y`
(ADR) survive. `tests/test_screener_universe_pool.py` pins it.

---

## Risk limits

Every order from every source passes the Secretary. Override any of these with
`TRADINGAGENTS_RISK_<NAME>`.

| Limit | Default |
|---|---|
| max position weight | 12% |
| max **new** position weight | 8% |
| max gross exposure | 95% |
| max trades / day | 12 |
| max turnover / day | 35% of equity |
| min order value | $250 |
| max single order | 10% of equity |
| per-symbol cooldown | 45 min |
| limit-price deviation | 5% from last |
| shorting | **off** |
| min price | $3 |

---

## Position sizing

Share count comes from the van Tharp rule:

```
quantity = account_value × risk% / |entry − stop|
```

This sizes so that being stopped out costs a **fixed fraction of the account**
regardless of which stock it is — equalising *risk* rather than *exposure*. A
10%-ATR name and a 2%-ATR name given the same dollar weight are not the same
bet, and sizing them identically is how a book quietly becomes a volatility bet.

Guarded hard against `stop == entry` (division by zero → infinite size, the
classic way this formula blows up an account) and wrong-side stops.

`r_multiple` (reward ÷ risk) is preferred over `expectancy` for ranking, because
expectancy is linear in a win-probability that is an unvalidated guess.

---

## News and policy

**Ticker news** — Yahoo (ticker-scoped) + Google News (searched by *company
name*, not ticker: "AMD stock" returns age-related macular degeneration trials,
which score 9/12 and would put a decision about the wrong company in front of
the panel).

Every story is fingerprinted so only genuinely *new* headlines count, and the
seen-set persists across restarts. A cold start primes instead of acting —
otherwise the first run treats a quarter of backlog as breaking news.

Institutional 13F noise is suppressed. Aggregators publish thousands of
"Ninepoint Partners LP Makes New Investment in X" headlines that match the M&A
pattern and score 10, which is enough to preempt a real stop-loss for the
cycle's LLM budget.

**Policy news** — monetary, fiscal, trade, regulatory, geopolitical. Company
news reprices one stock; policy reprices a whole sector at once. An agent
reading only ticker news is repeatedly blindsided by moves it had no way to see.

---

## The daily report: three horizons, one page per name

The advisor used to print only the ideas issued *that morning*. It carried the
older ones in `recommendations.json` and never showed them, so the page read as
a brand-new portfolio every day even when nothing had changed. The report is now
split by holding period, and the sections are ordered by how rarely they move:

| Section | Clock | Where the list comes from | Sized? |
|---|---|---|---|
| 核心长仓 core | months–years, reviewed monthly | `core.json`, hand-maintained | no — weights are the reader's |
| 在场的波段 open swing | 1–4 weeks | `recommendations.json`, still open | already sized when issued |
| 新增波段 new buys | 1–4 weeks | today's screen, into *free slots only* | yes |
| 日内 day trade | one session | watchlist ∪ screen, filtered on range and liquidity | no — levels, not orders |

Two rules do the work:

- **Slots, not a top-N.** `swing_slots` (default 6) caps concurrent swing ideas.
  New buys fill whatever is free, so a full book proposes nothing — which is the
  intended answer, not a bug.
- **Hysteresis on the core.** A core name leaves only on a stated rule: closing
  more than 4% below its 200-day, or a twelve-month loss past 15%. It never
  leaves because it slipped in this week's ranking. Weight drift is actioned
  only on a review day (the first four days of a month).

`core.json` is seeded on the first run that finds it missing. Two passes: a free
price screen — above the 200-day, a positive year, `$100M`+ daily turnover, not
already 35% off its high, under 5% ATR — then statements are fetched for the
survivors and the two tests price cannot make are applied: **the company must
earn money**, and **no industry may take more than two slots**. Ranking is
capped twelve-month return (40%), size (30%) and ROE (30%). Both caps matter:
ranking on the raw return makes "长期" a momentum screen, and equal weights
across four names in one supply chain is the same bet placed four times.

The invested fraction is *derived*, not fixed: `slots × position_cap` is the
most the swing book can be holding, and the core gets what is left less a 10%
cash buffer. At the defaults that is 42%, so a full swing book (6 × 8% = 48%)
and a full core come to 90% rather than the 108% a flat 60% produced.

What the seeder still cannot do: the concentration that matters is a *theme*,
and an industry label does not know that a GPU designer, a foundry and a
lithography supplier are one bet on AI capex. That judgement is why the file is
hand-editable, and why the output is announced as a draft. Edit it; nothing
overwrites it.

### One page per symbol

`reports/<date>.md` links every name to `reports/<date>/<SYMBOL>.md`, written by
`live/deepdive.py`. Each page carries, in this order: two ASCII price charts
with the 50/200-day averages and the trade's own stop/entry/target drawn as
levels; the chart read (均线排列, swing structure, momentum, volatility,
position, support/resistance) from `live/charting.py`; the trade's arithmetic
spelled out — stop distance in ATRs, R, the break-even win rate `1/(1+R)`, and
what one stop costs the account; the financial statements from
`live/fundamentals.py` (quarterly and annual income, margins, valuation,
balance sheet, the earnings-surprise record); the 24-hour news with links; a
rule-generated bear case; the raw OHLCV the page computed from; and the primary
sources from `live/research.py` — Yahoo, Finviz, TradingView, SEC EDGAR,
OpenInsider, plus Chinese-language relays.

The bear case is generated rather than written on purpose: a hand-written one
gets skipped on the names where it is least convenient.

### Turning the knobs

Anything that changes *what the report proposes* is settable as
`TRADINGAGENTS_ADVISOR_<FIELD>` — `SWING_SLOTS`, `MAX_OPEN_PER_SECTOR`,
`DAYTRADE_TOP`, `TOP`, `RISK_PCT`, `MIN_R`, `ATR_STOP_MULT`, `HORIZON_DAYS`,
`WRITE_PAGES`, `WITH_FUNDAMENTALS`, `MAX_FUNDAMENTALS`, `CORE_SEED` and the
rest of `AdvisorConfig._ENV_FIELDS`. A value that will not parse logs a warning
and keeps the default rather than silently applying a zero. `--swing-slots` and
`--no-pages` are also CLI flags, the two worth reaching for mid-session.

### Language and names

Narrative analysis is in Chinese; tables, tickers, levels and the R/ATR/SMA
vocabulary stay in English. Company names resolve through
`live/zhnames.py`: `company_names_zh.json` (yours, wins) → a curated table → a
mechanical gloss off the English suffix, marked `°` so a guess can never be
mistaken for a checked name.

## State

Everything under `~/.tradingagents/` (override with `TRADINGAGENTS_HOME`).

| File | What |
|---|---|
| `live_portfolio.json` | the paper book |
| `live_ledger.json` | every order attempted, with rationale — feeds the daily budgets |
| `live_journal.jsonl` | one line per cycle |
| `live_state.json` | screen ranking and coverage |
| `recommendations.json` | the advisor's own track record |
| `watchlist.json` | names analysed every day regardless of rank |
| `cache/screens/universe_*.json` | the monthly qualification pool |
| `news_seen.json` | story fingerprints, pruned at 72h |
| `company_names.json` | ticker → company name cache |
| `company_names_zh.json` | your Chinese names; wins over the built-in table |
| `core.json` | the long-term book — hand-maintained, seeded once |
| `fundamentals.json` | statements cache, 20h TTL |
| `earnings.json` | next report date and last surprise, 20h TTL |
| `screens/` | ranked exchange scans |
| `reports/` | daily advisor reports, plus `reports/<date>/<SYMBOL>.md` |
| `STOP` | the kill switch |

---

## Honest limitations

- **Paper only.** Routing to real money means writing a new adapter — a
  deliberate act, not a config flag.
- **The panel sees a compact pack** — price, trend, position, fresh headlines.
  Not filings, not transcripts.
- **The chart read is lagging by construction.** Moving averages, RSI and swing
  structure all describe what already happened. They say what state a name is
  in, never what it does next, and the verdict line is a summary of the bullets
  above it rather than a forecast.
- **The financial statements are a vendor's transcription**, not the filing.
  They are restated, they lag, and the TTM window is Yahoo's rather than the
  company's. Every page prints the EDGAR link beside them for that reason.
- **The intraday section has no intraday data.** Daily bars cannot produce an
  entry inside a session, so that section publishes the levels the next session
  will be measured against — the prior day's high and low — and never an order.
  Nothing in it enters the recommendation book or the track record.
- **Local paper fills are optimistic.** No slippage, no partial fills, no queue
  position. A strategy that depends on getting filled at the touch will look
  better here than anywhere real.
- **Policy impact is directional heuristics**, not modelled relationships. A
  rate cut is good for growth unless it signals recession.
- **This is not financial advice.** It is a model reading public information.
  The track record is the only thing that makes it checkable.
