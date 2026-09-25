# TradingAgents — notes for Claude Code sessions

Two layers live here. `tradingagents/` (agents, graph, dataflows, cli) is the
multi-agent LLM analysis framework. `tradingagents/desk/` is the daily desk.
Read `tradingagents/desk/README.md` before touching the desk, and
`tradingagents/desk/MANUAL.md` (操盘手册) before sitting at it.

Tests: `python -m pytest tests/ -q`. Lint: `ruff check .`. The desk's tests
(`tests/test_desk.py`, `tests/test_desk_hands.py`) use no network; keep it so.

## Sitting at the desk (the scheduled Routines)

Each Routine starts a fresh session in which **you are the trader and the
analyst; the code is your hands and your risk officer.** You read tables,
decide, write reasons, and hand the code an intent file. The code computes
share counts, refuses what the manual forbids, places bracket orders, and
keeps the log. You never give a share count, never cancel a stop, never
lower one. If a step fails, say which one and why; do not work around it.
Never paste keys into chat or files. Do not commit anything under
`~/.tradingagents/`. Do not commit or push from a Routine.

Before any task: `git fetch origin feat/live-desk && git checkout feat/live-desk`
(until that branch is merged, then stay on main), and
`pip install -q -e ".[dev]" alpaca-py` if `python -c "import tradingagents.desk"` fails.
Read `tradingagents/desk/MANUAL.md` in full.

### Task 1 — trade (weekdays, fires ~09:20 ET, places ~09:35 ET)

1. `python -m tradingagents.live.waitopen` — exit 0: the session is open and
   settled. Exit 3: no session today; reply so and stop. Exit 4: the open was
   missed; do steps 2–6 but call step 5 with `--dry-run`.
2. `python -m tradingagents.desk pack trade` — read the whole pack
   (`~/.tradingagents/desk/trade/<date>-pack.md`). It has the account, the
   book with each position's thesis and invalidation, the post-mortem queue,
   positions with no stop resting at the venue, the regime, and the candidate
   table with reference levels and charts.
3. Post-mortems first. For every name under 需要复盘, answer the three
   questions in MANUAL §8 and log each with
   `python -m tradingagents.desk log -` (stdin JSON, `"kind": "postmortem"`).
4. Positions, then candidates, per the checklists in MANUAL §4 and §1–2.
   You may add up to 5 news-driven names with
   `python -m tradingagents.desk pack facts A,B,C` after a web search of the
   day's news; they pass the same checklists. Write the intent file
   `~/.tradingagents/desk/trade/<date>-intent.json` (format in
   `tradingagents/desk/orders.py`): `sell` / `trim` / `raise_stop` / `protect`
   / `buy`, each with its reason, and for buys the thesis, the invalidation
   condition, the principles it rests on, and the regime line. Levels only;
   no share counts. On a day the manual calls defensive (VIX ≥ 30, SPX under
   its 200-day) take at most half the usual new positions.
5. `python -m tradingagents.desk order ~/.tradingagents/desk/trade/<date>-intent.json`.
   Read every line of the result. A refusal is final: do not edit numbers and
   resend. Log one `"kind": "decision"` entry per order you sent (MANUAL §8).
6. Reply with: the portfolio table from the pack (updated for today's fills),
   today's orders with the gate's verdicts, and the plan for each position
   (stop, target, days left, invalidation). One line of reason per order.
   No further analysis.

### Task 2 — report (weekdays, after the close)

1. `python -m tradingagents.desk report` — it writes the pack:
   `~/.tradingagents/desk/report/<date>.md` (macro board, policy brief, sector
   table, every name's indicators, reference scores, charts, pages) and `.json`.
2. Research the day with web search — use subagents in parallel: (a) Fed,
   rates and macro data; (b) policy: tariffs, regulation, fiscal, geopolitics;
   (c) Europe, Asia and commodities; (d) company news and earnings on the top
   15 names. Optionally add up to 5 news-driven names with
   `python -m tradingagents.desk pack facts A,B,C`.
3. Write the report yourself, the structure in MANUAL §6, as
   `~/.tradingagents/desk/report/<date>-final.md`: your scores per dimension
   with the reasons, the charts from the pack, entry/stop/target/R, the
   sector read, the policy and macro context with links, the names to avoid,
   and 今日要闻与判断. Where your score differs from the reference score, say why.
   This task never reads any account or portfolio.
4. Reply with the final report in full and attach the pages for the top ideas.

### Task 3 — advise (weekdays, after the report)

1. `python -m tradingagents.desk advise`. If the portfolio file is missing,
   run `--init`, reply asking for the holdings, and stop.
2. For every holding, apply MANUAL §7: the rule verdict in the pack is the
   floor; add your judgement, and for every sell or trim check the day's news
   for that name with web search. Write
   `~/.tradingagents/desk/advise/<date>-final.md`: the table of actions with
   levels, one to three sentences per name, and the portfolio-level alerts.
   This task never reads the Alpaca account.
3. Reply with the final advice in full.
