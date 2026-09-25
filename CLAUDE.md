# TradingAgents — notes for Claude Code sessions

Two layers live here. `tradingagents/` (agents, graph, dataflows, cli) is the
multi-agent LLM analysis framework. `tradingagents/desk/` is the daily desk:
three rule-based commands over the `tradingagents/live/` engine. Read
`tradingagents/desk/README.md` before touching the desk.

Tests: `python -m pytest tests/ -q`. Lint: `ruff check .`. The desk's tests are
`tests/test_desk.py` and use no network; keep it that way.

## Running the desk on a schedule (cloud Routines)

Each Routine starts a fresh session. The steps for each are below; do them in
order, and if a step fails say exactly which one and why rather than working
around it. Never paste keys into chat or into files. Do not commit anything
under `~/.tradingagents/`.

Before any task: `pip install -q -e ".[dev]" alpaca-py` if imports fail, and
check `git branch --show-current`; the desk lives on `feat/live-desk` until
that branch is merged (`git fetch origin feat/live-desk && git checkout feat/live-desk`).

### Task 1 — trade (weekdays, fires ~09:20 ET, trades ~09:35 ET)

1. `python -m tradingagents.live.waitopen` — exit 0 means the session is open
   and settled; 3 means no session today (stop, say so); 4 means the open was
   missed (run step 2 with `--dry-run` instead).
2. `python -m tradingagents.desk trade`
3. Reply with the note it printed (`~/.tradingagents/desk/trade/<date>.md`)
   verbatim: the portfolio, today's orders, the plan. Nothing else. No analysis,
   no commentary on the picks.

### Task 2 — report (weekdays, after the close)

1. `python -m tradingagents.desk report`
2. Read `~/.tradingagents/desk/report/<date>.json`. Then, with web search,
   add what the pack cannot know: the day's actual policy and macro news
   (Fed, tariffs, regulation, geopolitics), rate expectations, the global
   picture (Europe, Asia, commodities), and any earnings or company news on
   the top ideas. Use subagents for the search if it helps. Keep every number
   from the pack; add context, do not re-score.
3. Append a section `## 七、今日要闻与判断` to the report's `.md`: what changed
   in the world today, how it bears on the sectors and the top ideas, and which
   idea you would rank first and why. Cite sources with links.
4. Reply with the full report (the `.md`) and attach the per-symbol pages for
   the top ideas.

### Task 3 — advise (weekdays, after the report)

1. `python -m tradingagents.desk advise`. If the portfolio file is missing,
   create the template (`--init`) and reply asking for the holdings; do nothing else.
2. For every line marked sell or trim, check the news for that name with web
   search and say in one sentence whether the headline supports the rule's
   verdict or argues against it. Do not change the verdicts.
3. Reply with the advice (`~/.tradingagents/desk/advise/<date>.md`) and those
   sentences under a heading `## 补充`.
