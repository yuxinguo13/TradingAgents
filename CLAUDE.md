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
`~/.tradingagents/`. A Routine commits and pushes to `main` only what its
错误反思 step (below) calls for: code, tests, `CLAUDE.md`, `MANUAL.md`. Before
that push, `python -m pytest tests/test_desk.py tests/test_desk_hands.py -q`
must be green and `ruff check <files you touched>` must show no new error.

Before any task: `git fetch origin main && git checkout main && git pull -q --ff-only origin main`
(the desk is merged; `feat/live-desk` is only where new work is staged), and
`python -m pip install -q -e ".[dev]" alpaca-py` if `python -c "import tradingagents.desk.pack, alpaca"`
fails (2026-10-26: `import tradingagents.desk` passed on a machine with no numpy
or alpaca, and bare `pip` installed into a different Python than `python`).
Read `tradingagents/desk/MANUAL.md` in full.

State survives between sessions on the `desk-state` branch (`tradingagents/desk/state.py`):
`pack trade`, `report` and `advise` pull it before reading; `order` and `log`
push after writing. **The last desk step of every task is
`python -m tradingagents.desk state push`** so the day's final files are kept
(the push lays local files onto the branch and merges `*.jsonl` logs; it never
deletes). If a pull or push reports a failure, say so in the reply; do not
retry by hand.

### 错误反思 (owner, 2026-10-13: every day, in every report to the owner)

Every trade run and every report ends with a reflection, written before the
reply and acted on the same day. Write `~/.tradingagents/desk/reflect/<date>.md`
(the trade run) or `~/.tradingagents/desk/reflect/<date>-report.md` (the
report) with five parts, each a short list, each item one line of fact and one
line of consequence: **我的错** (judgement: a level set wrong, a rule misread, a
number trusted without checking, a name skipped for a bad reason); **代码的错**
(what the hands or the pack did that the manual does not say: a wrong price, a
lost record, a refusal that should not have been, a line that misled);
**规则的错** (a manual rule that cost money or blocked a trade the method
wanted, with the counts from `python -m tradingagents.desk review`; MANUAL §11
decides whether it may change today); **今天改了什么** (each change, the file,
the one-line why); **没改的** (what you saw and left, and why). Log it with
`python -m tradingagents.desk log -` as `{"kind": "reflection", ...}` carrying
the same five parts. Then fix the code the reflection names: edit, add a
regression test for each fault, run the tests and `ruff` on the files touched,
commit on `main` with a message starting `desk: reflect <date> —`, push with
`git push -u origin main`, and say in the reply if the push was refused. A
parameter (a limit in `orders.py` or a level in the manual) changes only when
§11's threshold is met, at most one per day, with the counts quoted in the
commit message. Put the reflection in the reply under `## 错误反思`, after the
day's table and before the equity line.

### Task 1 — trade (weekdays, fires ~15:12 ET, places ~15:30 ET, before the close)

The owner's method (2026-10-12): analysis at the open, trading at the close.
The morning analysis is the 开盘前简报 Routine (~08:45 ET, writes
`~/.tradingagents/desk/brief/<date>.md`); the evening report (Task 2) is
written after the close for the next session. The trade run reads both
before it decides, and is not conservative: it fills the momentum sleeve's
seats when the table has qualifying names, holds winners, and sells or trims
only on a rule trigger (stop, target, horizon, invalidation).

0. `python -m tradingagents.desk state pull`, then read in full today's
   `~/.tradingagents/desk/brief/<date>.md` and the latest
   `~/.tradingagents/desk/report/<date>-final.md`. Every order must answer to
   both. If today's brief is missing, say so and go on with the report.
1. `python -m tradingagents.desk pack trade` — it reads the last completed
   session's closes; the gate checks every level against the live price. On
   a fresh machine it takes ~15 min. Do not run `pack facts` after it (that
   pull overwrites the reconciled book); run it before, or skip it. Read the whole pack
   (`~/.tradingagents/desk/trade/<date>-pack.md`): the account, the book with
   each position's thesis and invalidation, the post-mortem queue, positions
   with no stop resting at the venue, the regime, and the candidate table
   with reference levels and charts.
2. `python -m tradingagents.live.waitopen` — exit 0: the session is open (at
   this hour it returns at once). Exit 3: no session, or the close has
   passed; reply so and stop. Exit 4: do steps 3–6 but call step 5 with
   `--dry-run`; also use `--dry-run` if it is past 15:50 ET at step 5.
   Levels come from the pack and the live prints; the gate checks each entry
   against the live price at order time.
3. Post-mortems first. For every name under 需要复盘, answer the three
   questions in MANUAL §8 (the line carries the trade's move relative to SPY;
   judge against that, not the raw P&L) and log each with
   `python -m tradingagents.desk log -` (stdin JSON, `"kind": "postmortem"`);
   that push also carries the reconciled book to the branch before `order`
   pulls it.
   Before writing any `"kind": "proposal"`, run
   `python -m tradingagents.desk review` and quote its counts (MANUAL §11).
4. Positions, then candidates, per the checklists in MANUAL §4 and §1–2.
   You may add up to 5 news-driven names with
   `python -m tradingagents.desk pack facts A,B,C` after a web search of the
   day's news; they pass the same checklists. Write the intent file
   `~/.tradingagents/desk/trade/<date>-intent.json` (format in
   `tradingagents/desk/orders.py`): `sell` / `trim` / `raise_stop` / `protect`
   / `cancel_buy` (a resting buy only; a stop is never cancelled) /
   `stage_half` (rest half the shares under a take-profit at 1R, same stop) / `buy`,
   each with its reason, and for buys the thesis, the invalidation
   condition, the principles it rests on, and the regime line. Levels only;
   no share counts. **The momentum sleeve comes first (MANUAL §10):** go
   down the pack's 进攻仓候选 table (the relative-strength leaders, with
   their triggers, breakout-day low and 20-day line) and fill the sleeve's
   4 seats with the names that meet two of `volume` / `catalyst` /
   `pattern` / `rs`, bought on strength (entry at the price or within 1 ATR
   above the 20-day line, stop at the breakout-day low or the 20-day line
   whichever is higher, ≤ 8%), as `"sleeve": "aggressive"` with
   `"triggers"`; read live prices before setting entries. Then the core
   pullback candidates. A position the pack marks unprotected gets a
   `protect` only after the venue confirms no stop leg is resting (the
   wrapper cannot see the held stop leg of a filled bracket). On a day the manual calls defensive (VIX ≥ 30, SPX
   under its 200-day) take at most half the usual new positions.
5. `python -m tradingagents.desk order ~/.tradingagents/desk/trade/<date>-intent.json`.
   Read every line of the result. A refusal is final: do not edit numbers and
   resend. Log one `"kind": "decision"` entry per order you sent (MANUAL §8).
6. `python -m tradingagents.desk state push`.
7. 错误反思 (above): write `~/.tradingagents/desk/reflect/<date>.md`, log it,
   fix and commit what it names, `state push` again, then reply with: the
   portfolio table from the pack (updated for today's fills), today's orders
   with the gate's verdicts, the plan for each position (stop, target, days
   left, invalidation), one line of reason per order, the `## 错误反思`
   section, and last the equity line. No further analysis.

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
   sector read, the policy and macro context with links, a
   进攻仓候选与突破跟踪 section (the pack's breakout table plus which of the
   three triggers each name meets and the breakout-day low), a 昨日复盘
   section (the pack's table of the previous report's calls against today's
   closes with the SPY-relative column, plus the pack's 五日结算 table of the
   calls from five sessions ago scored in R, plus three sentences: which call
   the market has falsified, why, and which rule you would change), the names to avoid, and 今日要闻与判断. Where your score differs from the reference score, say why.
   This task never reads any account or portfolio.
4. `python -m tradingagents.desk site --date <date> --out ./site-<date>` —
   builds the report as a web site in that folder inside the repo checkout
   (the Artifact tool only publishes from under the working directory; the
   folder is gitignored): an index plus a page per name with the chart, your
   write-up, the code's reading and the full analysis. Publish it with the
   Artifact tool: `file_path` is the folder's `index.html`, `root` is the
   folder, `files` is the list in its `files.json`; icon `chart`, a
   one-sentence description. If this session has no Artifact tool, say so
   and skip the publish.
5. `python -m tradingagents.desk state push`.
6. 错误反思 (above) for the analysis side: which of yesterday's calls the
   market falsified and why, which score or level you set wrong, what the pack
   or the site builder got wrong, what you changed in the code today. Write
   `~/.tradingagents/desk/reflect/<date>-report.md`, log it, fix and commit
   what it names, `state push` again, and put the same text in the final
   report as `## 错误反思` after 昨日复盘. Then reply with the artifact link
   first, followed by the final report in full.

### Task 4 — stock (on demand: the user names a symbol and asks for its analysis)

1. `python -m tradingagents.desk stock <SYM>` — writes the pack
   `~/.tradingagents/desk/stock/<date>-<SYM>.md` (+ `.json`, and the full
   `-deepdive.md` page): price and chart, the code's score and reference
   levels, breakout triggers, earnings date, insiders, the sector's week and
   month, headlines, and every earlier call the desk made on the name,
   settled over five sessions.
2. Research with web search, in parallel: (a) the company's last two
   earnings, guidance and the next report date; (b) the past two weeks' news,
   analyst actions with firm and target, legal or regulatory items; (c) the
   sector's policy backdrop and one or two peers for context.
3. Write `~/.tradingagents/desk/stock/<date>-<SYM>-final.md` in the structure
   of MANUAL §12, including one ranking-table row in the daily report's
   format so the page and later settlement can read your levels.
4. `python -m tradingagents.desk stock site <SYM> --date <date> --out ./site-<SYM>-<date>`
   and publish that folder with the Artifact tool (`file_path` its
   `index.html`, `root` the folder, `files` from its `files.json`; icon
   `chart`). Reply with the link first, then the final in full.
5. `python -m tradingagents.desk state push`. This task reads no account;
   if the user holds the name, say what to do with the holding under the
   horizon their portfolio file declares.

### Task 3 — advise (weekdays, after the report)

1. `python -m tradingagents.desk advise`. If the portfolio file is missing,
   run `--init`, reply asking for the holdings, and stop.
2. For every holding, apply MANUAL §7: the rule verdict in the pack is the
   floor; add your judgement, and for every sell or trim check the day's news
   for that name with web search. Write
   `~/.tradingagents/desk/advise/<date>-final.md`: the table of actions with
   levels, one to three sentences per name, and the portfolio-level alerts.
   This task never reads the Alpaca account.
3. `python -m tradingagents.desk state push`, then reply with the final advice in full.
