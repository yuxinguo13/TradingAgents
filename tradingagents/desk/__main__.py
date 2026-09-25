"""``python -m tradingagents.desk {trade,report,advise} [options]``."""

from __future__ import annotations

import sys

USAGE = """usage: python -m tradingagents.desk <task> [options]

  trade    paper-trade the Alpaca account by rule, then say where it stands
  report   the market, sector by sector, with no account in view
  advise   what to do with a portfolio you typed in

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
