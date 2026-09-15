"""A snapshot names the session its last bar belongs to. No network."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.live import brain


def frame(last="2026-09-11", n=60):
    dates = pd.bdate_range(end=last, periods=n)
    close = np.linspace(10.0, 20.0, n)
    return pd.DataFrame({"Date": dates, "Open": close, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close, "Volume": np.full(n, 1e6)})


@pytest.mark.unit
def test_the_last_bar_date_is_carried(monkeypatch):
    """Asked for 2026-09-14 and handed a frame ending on the 11th, it says the 11th."""
    monkeypatch.setattr(brain, "load_ohlcv", lambda sym, when: frame("2026-09-11"))
    s = brain.snapshot("OMER", "2026-09-14")
    assert s.ok and s.bar_date == "2026-09-11"


@pytest.mark.unit
def test_a_frame_without_dates_leaves_it_empty(monkeypatch):
    """Empty means unknown, which the advisor does not read as stale."""
    monkeypatch.setattr(brain, "load_ohlcv", lambda sym, when: frame().drop(columns=["Date"]))
    s = brain.snapshot("OMER", "2026-09-14")
    assert s.bar_date == ""
