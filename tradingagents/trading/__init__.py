"""Paper-trading layer: portfolio state, two-sleeve sizing, and the desk.

The primary interface is the desk (`python -m tradingagents.trading.desk`),
which never touches an LLM. The graph-driven AutoTrader remains available for
API-based runs.
"""

from .execution import PaperBroker, plan_trade
from .portfolio import Portfolio, Position

__all__ = ["AutoTrader", "Desk", "PaperBroker", "Portfolio", "Position", "plan_trade"]

_LAZY = {"AutoTrader": ".autotrader", "Desk": ".desk"}


def __getattr__(name):
    # Lazy so `python -m tradingagents.trading.<mod>` doesn't import the module
    # twice (RuntimeWarning), and so importing the desk never pulls langgraph.
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
