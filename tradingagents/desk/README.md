# The desk

Claude sits at it; the code is its hands and its risk officer. The manual
Claude follows is [`MANUAL.md`](MANUAL.md) (操盘手册); the procedures for the
scheduled sessions are in the repo's `CLAUDE.md`.

| Command | What it does |
|---|---|
| `python -m tradingagents.desk pack trade [--add A,B]` | The morning pack: account, book with theses and invalidations, post-mortem queue, unprotected positions, regime, candidate table with reference levels and charts. |
| `python -m tradingagents.desk pack facts A,B,C` | The same table and charts for names Claude picked itself. |
| `python -m tradingagents.desk order intent.json [--dry-run]` | The gate and the hands: sizes buys by the manual's formula, refuses what §3 forbids, places bracket orders, raises stops, protects, trims, closes. Logs everything. |
| `python -m tradingagents.desk log entry.json` | Append a decision or post-mortem to `desk/trade/decisions.jsonl`. |
| `python -m tradingagents.desk report` | The report pack: macro board, policy, sectors, every prominent name scored on the reference rule, charts, a page per name. Never reads an account. |
| `python -m tradingagents.desk advise` | The advice pack for a portfolio typed into a file: rule verdicts, levels, alerts. |
| `python -m tradingagents.desk trade` | The old fully rule-based trader. Kept as a fallback when nobody is reading the pack. |

State lives under `$TRADINGAGENTS_HOME/desk/<task>/` (default `~/.tradingagents/desk/`).
No task opens another task's files. The older combined flow (`live/advisor.py`
writing a book that `live/execute.py` reconciled against the account) is still
in the tree but is not what runs.

## The division of labour

| Claude | Code |
|---|---|
| reads the pack, the news, the charts | computes every indicator and the reference levels |
| decides what to buy, sell, trim, protect; where the stop and target go; why | computes the share count: `min(equity × 1% ÷ |entry − stop|, equity × 10% ÷ entry, cash ÷ entry)` |
| writes the thesis, the invalidation condition, the principles, the regime | refuses: R < 2, stop under 1 ATR or over 10%, entry > 3% from the last price, earnings within 1 day, 200-day rule, 8 positions, 3 per sector, 10% per name, 3% new risk per day, 3% daily drawdown halt, re-entry within 10 days, market closed, kill switch |
| answers the post-mortem questions | places bracket orders (stop + take-profit resting at the venue, GTC); raises stops, never lowers or cancels them; the one cancel is inside a close |
| writes the report and the advice | keeps `decisions.jsonl` and `book.json` |

## 1 · the morning (trade)

`pack trade` → Claude reads → post-mortems → intent file → `order` → reply.
The intent file format is in `orders.py`'s docstring. Every order is logged
with the reasons Claude gave and the verdict the gate returned, placed or not.

Needs `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (paper). Kill switch:
`touch ~/.tradingagents/STOP`.

Knobs for the reference levels and the old rule-based trader are
`TRADINGAGENTS_DESK_<NAME>`; the gate's limits are constants at the top of
`orders.py` and are changed there and in the manual together.

## 2 · report (the pack for the evening)

```
python -m tradingagents.desk report                 # next session's report
python -m tradingagents.desk report --use-cache     # reuse today's saved screen
python -m tradingagents.desk report --top 15        # more ideas in full
```

Writes `desk/report/<date>.md`, `desk/report/<date>.json` (the whole pack,
structured) and `desk/report/<date>/<SYMBOL>.md` (one page per name: chart
read, levels, fundamentals, earnings, news, risks, links).

Sections: macro & rates (indices, the Treasury curve and its spread, VIX,
dollar, oil, gold, bitcoin), policy & news (the policy monitor's brief and
per-sector tilt, macro headlines), sectors (each ETF's week and month, the
tilt, how many leaders sit above their 50-day), the ideas (score, ASCII chart
with the averages and the levels, reasons, cautions, entry/stop/target, next
earnings, fundamentals line, headlines), the names to avoid, and the full
scoreboard.

The score is one published rule (`report.score`): trend ≤ 40, momentum ±20,
volume ±5, relative strength vs SPY ±10, policy tilt ±10, news ±15, and
deductions for being stretched, broken or a week from earnings. The universe
is `universe.BELLWETHERS` (the largest names in every sector, always on the
page) plus the screen's leaders capped per sector.

## 3 · advise (the pack for the portfolio)

```
python -m tradingagents.desk advise --init                    # write a template portfolio file
python -m tradingagents.desk advise                           # read desk/advise/my_portfolio.json
python -m tradingagents.desk advise --portfolio ~/mine.csv    # or any json / csv / txt
```

Portfolio formats:

```
json   {"cash": 12000, "holdings": [{"symbol": "AAPL", "shares": 10, "cost": 150.5}]}
csv    symbol,shares,cost            (a row "cash,12000," is the cash)
txt    AAPL 10 150.5                 (one per line; "cash 12000" is the cash; # comments)
```

Rules (`advise.decide`), first match wins: trend broken (under both averages
with a negative quarter, or 8% under the 200-day) → **sell**; a material
bearish headline → **sell** (or **trim** half if the trend is intact); one name
over 20% of the portfolio → **trim** to 15%; 50% above the 200-day or RSI ≥ 80 →
**trim** a third; trend intact, pulled back to an average, under 5% weight,
cash on hand, no earnings within two weeks → **add**; otherwise **hold**. Every
line carries a stop and a target. Portfolio alerts: concentration, one-sector
risk, sector weight against the policy tilt, thin cash, earnings inside two
weeks, and the names to handle today.

Writes `desk/advise/<date>.md`, `.json`, and a page per name.

## Scheduling

On a Mac, `scripts/desk_cron.sh trade|report|advise` (see the header for the
three launchd traps). In a Claude Code cloud environment, a Routine per task;
the prompts are in `CLAUDE.md`. Either way the environment needs the Alpaca
keys and outbound access to `query2.finance.yahoo.com`, `fc.yahoo.com`,
`news.google.com` and `paper-api.alpaca.markets` / `data.alpaca.markets`.

## What the code does not do

- It does not place stop orders at the venue. Stops live in `desk/trade/book.json`
  and are checked once per run. A gap through the stop between runs is sold at
  the next run's price.
- It does not read filings or transcripts. News is headlines from RSS.
- Rates come from the Treasury yield tickers; there is no Fed-funds series
  unless `FRED_API_KEY` is set for the analysis framework, which the desk does
  not call.
- Charts are text. They render in any markdown viewer and in a terminal, and
  need no plotting library.
- Nothing here is back-tested and none of it is investment advice.
