"""Live trading against the Investopedia simulator.

An always-on agent: it watches a dynamic universe and the news wire
continuously, reasons only when something actually happens, and routes every
resulting order through a hard risk gate before it reaches the venue.

Layers, bottom to top:

* :mod:`clock`        — what session the US market is in right now
* :mod:`newsfeed`     — keyless RSS polling with persistent novelty detection
* :mod:`investopedia` — browser-driven adapter for the simulator
* :mod:`secretary`    — order validation and the hard risk limits
* :mod:`personas`     — the analyst panel's mandates
* :mod:`brain`        — triggers, evidence packs, panel deliberation
* :mod:`monitor`      — the loop that ties them together

Everything is imported lazily so `python -m tradingagents.live.cli` does not
pull Playwright or an LLM SDK for a command that needs neither.
"""

_LAZY = {
    "InvestopediaBroker": ".investopedia",
    "Account": ".investopedia",
    "Secretary": ".secretary",
    "RiskLimits": ".secretary",
    "Order": ".secretary",
    "NewsMonitor": ".newsfeed",
    "LiveDesk": ".monitor",
    "MonitorConfig": ".monitor",
    "Panel": ".brain",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
