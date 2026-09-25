"""``python -m tradingagents.desk {trade,report,advise} [options]``."""

from __future__ import annotations

import sys

USAGE = """usage: python -m tradingagents.desk <task> [options]

  pack     the tables Claude reads before deciding (pack trade | pack facts A,B,C)
  order    place what an intent file asks for, through the gate (order intent.json)
  log      append a decision / post-mortem entry to the decision log
  state    pull | push the desk's state directory from / to the desk-state branch
  report   the market, sector by sector, with no account in view
  advise   what to do with a portfolio you typed in
  trade    the rule-based trader (fallback when nobody is reading the pack)

Each task takes --help. State: $TRADINGAGENTS_HOME/desk/<task>/
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    task, rest = argv[0], argv[1:]
    if task == "trade":
        from .trade import main as run
    elif task == "pack":
        from .pack import main as run
    elif task == "order":
        from .orders import main as run
    elif task == "log":
        from .orders import main_log as run
    elif task == "state":
        from .state import main as run
    elif task == "report":
        from .report import main as run
    elif task == "advise":
        from .advise import main as run
    else:
        print(f"unknown task {task!r}\n\n{USAGE}")
        return 2
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
