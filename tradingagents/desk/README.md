# The desk

Three jobs, three commands, nothing shared but the market data.

| Command | What it does | When |
|---|---|---|
| `python -m tradingagents.desk trade` | Trades the Alpaca paper account by rule, then writes where it stands: positions, today's orders, the plan. No analysis. | every session, ~09:35 ET |
| `python -m tradingagents.desk report` | The market, sector by sector: macro board, rates, policy, news, every prominent name scored with its reasons, charts, one page per name. Never looks at an account. | after the close |
| `python -m tradingagents.desk advise` | Keep / add / trim / sell for a portfolio you typed into a file, with levels, reasons and portfolio-level alerts. | after the report |

State lives under `$TRADINGAGENTS_HOME/desk/<task>/` (default `~/.tradingagents/desk/`).
No task opens another task's files. The older combined flow (`live/advisor.py`
writing a book that `live/execute.py` reconciled against the account) is still
in the tree but is not what runs; this package replaces it.

## 1 · trade

```
python -m tradingagents.desk trade              # place what the rules say
python -m tradingagents.desk trade --dry-run    # decide and report, place nothing
python -m tradingagents.desk trade --no-entries # manage exits only
```

Needs `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper keys). Every run:

1. **Adopts** any position in the account it has no record of — gives it a stop
   under the structure and a target from the trend, dated today. Nothing in the
   account is ever unmanaged.
2. **Reviews** every position: stop hit → sell; target reached → sell half;
   past the holding horizon → sell; a headline that breaks the thesis → sell;
   up one R → stop to breakeven. Sells go to the venue first.
3. **Opens** new positions with the slots and cash left, from the top of the
   momentum screen across all exchanges: above the 50- and 200-day lines, not
   overbought, no earnings inside the horizon, stop under the structure, target
   from the trend, at least `min_r` reward per unit of risk. Sized so a stop-out
   costs `risk_pct` of equity, capped at `cap_fraction` of equity per name.
4. Every order passes the risk gate (`live/secretary.py`: position caps, daily
   turnover, price sanity) and is written to `desk/trade/ledger.json`.
5. Writes `desk/trade/<date>.md` and appends to `desk/trade/runs.jsonl`.

Knobs, all `TRADINGAGENTS_DESK_<NAME>` in the environment: `RISK_PCT` (1.0),
`CAP_FRACTION` (0.10), `MAX_POSITIONS` (8), `HORIZON_DAYS` (30), `MIN_R` (1.5),
`MIN_PRICE` (5), `EXCHANGE` (all), `TOP` (40), `PER_SECTOR` (2),
`REENTRY_DAYS` (10), `LIMIT_BUFFER` (0.005), `VENUE` (alpaca).

Kill switch: `touch ~/.tradingagents/STOP`. The run exits before reading the account.

## 2 · report

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

## 3 · advise

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
