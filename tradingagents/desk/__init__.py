"""The desk: three jobs, three commands, nothing shared but the data.

    python -m tradingagents.desk trade    # paper-trade the Alpaca account, then say where it stands
    python -m tradingagents.desk report   # the market, sector by sector, with no account in view
    python -m tradingagents.desk advise   # what to do with a portfolio you typed in

The older ``live/advisor`` + ``live/execute`` pair put all three behind one
recommendation book: the report wrote buys into it, the executor read them back
out, and a holding you already had could veto an idea the report was about to
make. That is the entanglement this package undoes. Each task here reads the
same market data through :class:`~.market.Market`, keeps its own state under
``$TRADINGAGENTS_HOME/desk/<task>/``, and never opens another task's files.

Everything below the desk — the screener, the bars, the chart reader, the news
and policy feeds, the sizing arithmetic, the Alpaca adapter and the risk gate —
is the existing ``live`` engine, untouched.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """``$TRADINGAGENTS_HOME/desk`` (default ``~/.tradingagents/desk``)."""
    root = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
    return root / "desk"


def task_dir(task: str) -> Path:
    d = home() / task
    d.mkdir(parents=True, exist_ok=True)
    return d
