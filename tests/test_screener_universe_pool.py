"""Universe membership, the monthly qualification pool, and the watchlist.

The bug these pin down is the silent kind: `fetch_universe` dropped every
5-letter Nasdaq symbol to exclude warrants and units, which also deleted
GOOGL, CMCSA, FCNCA and the whole Liberty complex from the screen. Nothing
errored — the names simply never appeared, so no run ever looked wrong.
"""

import json

import pandas as pd
import pytest

from tradingagents.trading import screener as sc

# One row per case, in the shape `_read_pipe_table` returns for nasdaqlisted.txt.
_ROWS = [
    # symbol,  security name,                              why it is here
    ("NVDA",   "NVIDIA Corporation - Common Stock"),       # 4-letter control
    ("GOOGL",  "Alphabet Inc. - Class A Common Stock"),    # 5-letter class share
    ("CMCSA",  "Comcast Corporation - Class A Common Stock"),
    ("RYAAY",  "Ryanair Holdings plc - American Depositary Shares"),
    ("BATRK",  "Atlanta Braves Holdings, Inc. - Series C Common Stock"),
    ("PSNYW",  "Polestar Automotive Holding UK Limited - Class C-1 ADS (ADW)"),
    ("CDZIP",  "Cadiz, Inc. - Depositary Shares"),
    ("ABCDU",  "Some Acquisition Corp - Unit"),
    ("ABCDR",  "Some Acquisition Corp - Right"),
    ("PLAIN",  "Plain Five Letter Inc. - Common Stock"),
]


@pytest.fixture
def _fake_nasdaq(monkeypatch):
    df = pd.DataFrame({
        "Symbol": [s for s, _ in _ROWS],
        "Security Name": [n for _, n in _ROWS],
        "ETF": ["N"] * len(_ROWS),
        "Test Issue": ["N"] * len(_ROWS),
    })
    monkeypatch.setattr(sc, "_read_pipe_table", lambda url: df.copy())
    return df


@pytest.mark.unit
def test_five_letter_class_shares_survive(_fake_nasdaq):
    # The regression: these are common stock that happens to need five letters.
    syms = set(sc.fetch_universe("nasdaq")["symbol"])
    for keep in ("GOOGL", "CMCSA", "RYAAY", "BATRK", "PLAIN"):
        assert keep in syms, f"{keep} was dropped from the universe"


@pytest.mark.unit
def test_instrument_suffixes_are_still_excluded(_fake_nasdaq):
    # W/U/R/P mean warrant/unit/right/preferred. PSNYW is the case the name
    # filter misses — its name says "ADW", never "warrant" — so the suffix
    # rule is doing real work and not merely duplicating _EXCLUDE_NAME.
    syms = set(sc.fetch_universe("nasdaq")["symbol"])
    for drop in ("PSNYW", "CDZIP", "ABCDU", "ABCDR"):
        assert drop not in syms, f"{drop} should not be in the universe"


@pytest.mark.unit
def test_four_letter_symbols_are_untouched_by_the_suffix_rule(_fake_nasdaq):
    # The rule is scoped to length 5; a 4-letter symbol ending in P/W/U/R is a
    # normal ticker and must not be caught (e.g. a real "XYZP" common stock).
    df = _fake_nasdaq
    df.loc[len(df)] = ["ABCP", "Four Letter Ending In P - Common Stock", "N", "N"]
    monkey = sc.fetch_universe("nasdaq")
    assert "ABCP" in set(monkey["symbol"])


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_watchlist_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_WATCHLIST_PATH", str(tmp_path / "nope.json"))
    assert sc.load_watchlist() == {}


@pytest.mark.unit
def test_watchlist_malformed_file_degrades_to_empty(tmp_path, monkeypatch):
    # A watchlist that cannot be parsed must cost the watchlist section, not
    # the whole scan.
    p = tmp_path / "watchlist.json"
    p.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setenv("TRADINGAGENTS_WATCHLIST_PATH", str(p))
    assert sc.load_watchlist() == {}
    p.write_text('["a", "list", "not", "a", "map"]', encoding="utf-8")
    assert sc.load_watchlist() == {}


@pytest.mark.unit
def test_watchlist_symbols_are_normalised(tmp_path, monkeypatch):
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps({" nvda ": "semi", "Googl": "tech"}), encoding="utf-8")
    monkeypatch.setenv("TRADINGAGENTS_WATCHLIST_PATH", str(p))
    assert sc.load_watchlist() == {"NVDA": "semi", "GOOGL": "tech"}


# ---------------------------------------------------------------------------
# qualification pool freshness
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("built,on,age,expected", [
    ("2026-08-01", "2026-08-25", 30, True),    # inside the window
    ("2026-07-01", "2026-08-25", 30, False),   # older than a month
    ("2026-08-25", "2026-08-25", 30, True),    # built today
    ("2026-09-01", "2026-08-25", 30, False),   # built in the future
    (None,         "2026-08-25", 30, False),   # never built
    ("garbage",    "2026-08-25", 30, False),   # unparseable
])
def test_pool_freshness(built, on, age, expected):
    # A pool built after the date being screened would leak future liquidity
    # into a backtest, so it is treated as stale rather than as valid.
    assert sc._pool_is_fresh({"built": built}, on, age) is expected


@pytest.mark.unit
def test_pool_is_rebuilt_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "fetch_universe", lambda exchange="nasdaq": pd.DataFrame(
        {"symbol": ["AAA", "BBB", "PENNY"], "name": ["a", "b", "c"],
         "exchange": ["NASDAQ"] * 3}))
    monkeypatch.setattr(sc, "download_panel", lambda *a, **k: {"Close": pd.DataFrame()})
    monkeypatch.setattr(sc, "compute_factors", lambda panel, **k: pd.DataFrame({
        "price":          [50.0, 20.0, 0.40],
        "dollar_vol_50":  [9e6,  5e6,  9e6],
        "rows":           [250,  250,  250],
    }, index=["AAA", "BBB", "PENNY"]))

    syms, meta = sc.qualified_universe("2026-08-25", log=lambda m: None)
    assert syms == ["AAA", "BBB"]          # PENNY fails the price floor
    assert meta["rebuilt"] is True
    assert meta["listed"] == 3

    saved = json.loads((tmp_path / "universe_nasdaq.json").read_text())
    assert saved["symbols"] == ["AAA", "BBB"]
    assert saved["built"] == "2026-08-25"

    # Second call the same month reads the file instead of rebuilding.
    monkeypatch.setattr(sc, "fetch_universe", lambda exchange="nasdaq": (_ for _ in ()).throw(
        AssertionError("must not re-fetch the universe while the pool is fresh")))
    syms2, meta2 = sc.qualified_universe("2026-08-26", log=lambda m: None)
    assert syms2 == ["AAA", "BBB"]
    assert not meta2.get("rebuilt")


@pytest.mark.unit
def test_pool_keeps_names_below_their_200_day(tmp_path, monkeypatch):
    # The whole point of a separate pool: a name out of favour today is the
    # one that comes back next month. The pool must not apply the trend
    # filter, or the monthly refresh could never rediscover anything.
    monkeypatch.setattr(sc, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "fetch_universe", lambda exchange="nasdaq": pd.DataFrame(
        {"symbol": ["FALLEN"], "name": ["x"], "exchange": ["NASDAQ"]}))
    monkeypatch.setattr(sc, "download_panel", lambda *a, **k: {"Close": pd.DataFrame()})
    monkeypatch.setattr(sc, "compute_factors", lambda panel, **k: pd.DataFrame({
        "price": [40.0], "dollar_vol_50": [8e6], "rows": [250],
        "above_200": [False], "rvol_20": [0.9], "ext_200": [-0.2],
    }, index=["FALLEN"]))
    syms, _ = sc.qualified_universe("2026-08-25", log=lambda m: None)
    assert syms == ["FALLEN"]


# ---------------------------------------------------------------------------
# screen(): the watchlist is reported, never promoted
# ---------------------------------------------------------------------------

def _screen_env(monkeypatch, tmp_path, factors):
    monkeypatch.setattr(sc, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "fetch_universe", lambda exchange="nasdaq": pd.DataFrame(
        {"symbol": list(factors.index), "name": list(factors.index),
         "exchange": ["NASDAQ"] * len(factors)}))
    monkeypatch.setattr(sc, "qualified_universe",
                        lambda *a, **k: (sorted(factors.index), {"qualified": len(factors),
                                                                 "built": "2026-08-25"}))
    monkeypatch.setattr(sc, "download_panel", lambda *a, **k: {"Close": pd.DataFrame()})
    monkeypatch.setattr(sc, "compute_factors", lambda panel, **k: factors.copy())


def _factor_frame():
    # WINNER passes everything; FALLEN is a healthy, liquid name that has lost
    # its 200-day — the exact case the watchlist exists to keep visible.
    idx = ["WINNER", "FALLEN"]
    return pd.DataFrame({
        "price":        [100.0, 100.0],
        "dollar_vol_50": [5e8, 5e8],
        "rows":         [250, 250],
        "rvol_20":      [0.5, 0.5],
        "ext_200":      [0.1, -0.1],
        "above_200":    [True, False],
        "mom_12_1":     [0.5, -0.1],
        "ret_6m":       [0.4, -0.1],
        "rs_3m":        [0.3, -0.1],
        "ud_vol_50":    [1.2, 0.9],
        "obv_slope_50": [1.0, -1.0],
        "sma50_slope":  [1.0, -1.0],
        "off_high":     [-0.01, -0.30],
        "dist_days_25": [1, 6],
        "above_50":     [True, False],
    }, index=idx)


@pytest.mark.unit
def test_watchlist_name_below_its_200_day_is_still_reported(monkeypatch, tmp_path):
    factors = _factor_frame()
    _screen_env(monkeypatch, tmp_path, factors)
    g, stats, wl = sc.screen("2026-08-25", top=10, max_per_sector=None,
                             watchlist={"FALLEN": "semi"}, return_watchlist=True,
                             log=lambda m: None)
    assert "FALLEN" not in g.index, "a filtered name must not enter the ranking"
    assert "FALLEN" in wl.index, "a watchlist name must be reported anyway"
    assert bool(wl.loc["FALLEN", "passes_filter"]) is False
    assert wl.loc["FALLEN", "tag"] == "semi"


@pytest.mark.unit
def test_watchlist_does_not_change_the_ranking(monkeypatch, tmp_path):
    # Following a name daily is not a reason to buy it. The ranked frame must
    # be byte-identical whether or not a watchlist was supplied.
    factors = _factor_frame()
    _screen_env(monkeypatch, tmp_path, factors)
    plain, _ = sc.screen("2026-08-25", top=10, max_per_sector=None,
                         watchlist={}, log=lambda m: None)
    _screen_env(monkeypatch, tmp_path, factors)
    withwl, _, _ = sc.screen("2026-08-25", top=10, max_per_sector=None,
                             watchlist={"FALLEN": "semi"}, return_watchlist=True,
                             log=lambda m: None)
    assert list(plain.index) == list(withwl.index)


@pytest.mark.unit
def test_screen_keeps_the_two_tuple_shape_by_default(monkeypatch, tmp_path):
    # monitor.py and advisor.py unpack two values; adding the watchlist must
    # not break either caller.
    factors = _factor_frame()
    _screen_env(monkeypatch, tmp_path, factors)
    result = sc.screen("2026-08-25", top=10, max_per_sector=None, log=lambda m: None)
    assert len(result) == 2


@pytest.mark.unit
def test_watchlist_symbol_with_no_price_data_is_named_not_dropped(monkeypatch, tmp_path):
    # A watchlist entry that yfinance could not price should be reported as
    # missing rather than silently vanish — the silent-drop failure mode this
    # whole module exists to prevent.
    factors = _factor_frame()
    _screen_env(monkeypatch, tmp_path, factors)
    _, stats, wl = sc.screen("2026-08-25", top=10, max_per_sector=None,
                             watchlist={"GHOST": "semi"}, return_watchlist=True,
                             log=lambda m: None)
    assert "GHOST" not in wl.index
    assert stats["watchlist_missing"] == ["GHOST"]


# ---------------------------------------------------------------------------
# screen_watchlist(): independent of the universe scan
# ---------------------------------------------------------------------------

def _wl_factors():
    return pd.DataFrame({
        "price":         [100.0, 100.0, 100.0, 100.0],
        "dollar_vol_50": [5e8,   5e8,   4e6,   5e8],
        "rows":          [250,   250,   250,   250],
        "rvol_20":       [0.5,   0.5,   0.5,   1.9],
        "ext_200":       [0.1,  -0.1,   0.1,   0.1],
        "above_200":     [True,  False, True,  True],
        "above_50":      [True,  False, True,  True],
    }, index=["FINE", "FALLEN", "THIN", "WILD"])


@pytest.fixture
def _wl_env(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "download_panel", lambda *a, **k: {"Close": pd.DataFrame()})
    monkeypatch.setattr(sc, "compute_factors", lambda panel, **k: _wl_factors())


@pytest.mark.unit
def test_watchlist_scoring_needs_no_universe_scan(_wl_env, monkeypatch):
    # The point of the separate path: it must work on days the screen was
    # cached, skipped or blew up.
    monkeypatch.setattr(sc, "fetch_universe", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("screen_watchlist must not touch the universe")))
    monkeypatch.setattr(sc, "qualified_universe", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("screen_watchlist must not build the pool")))
    f = sc.screen_watchlist("2026-08-25", watchlist=["FINE"], log=lambda m: None)
    assert list(f.index) == ["FINE"]


@pytest.mark.unit
def test_every_watchlist_name_is_returned_pass_or_fail(_wl_env):
    f = sc.screen_watchlist("2026-08-25", log=lambda m: None,
                            watchlist={"FINE": "a", "FALLEN": "a", "THIN": "b", "WILD": "b"})
    assert set(f.index) == {"FINE", "FALLEN", "THIN", "WILD"}
    assert bool(f.loc["FINE", "passes_filter"]) is True
    assert bool(f.loc["FALLEN", "passes_filter"]) is False


@pytest.mark.unit
def test_failure_reason_names_the_binding_filter(_wl_env):
    # "Below the bar" makes the reader go and look; naming the filter does not.
    # The reason is the FIRST binding filter in a fixed order, not a list, so a
    # name failing two of them reports the one checked first.
    f = sc.screen_watchlist("2026-08-25", log=lambda m: None,
                            watchlist=["FINE", "FALLEN", "THIN", "WILD"])
    assert f.loc["FINE", "fail_reason"] == ""
    assert f.loc["FALLEN", "fail_reason"] == "below the 200-day"
    assert "thin" in f.loc["THIN", "fail_reason"]
    assert f.loc["WILD", "fail_reason"] == "too volatile"


@pytest.mark.unit
def test_empty_watchlist_returns_an_empty_frame_not_an_error(_wl_env):
    assert sc.screen_watchlist("2026-08-25", watchlist={}, log=lambda m: None).empty


@pytest.mark.unit
def test_unpriceable_watchlist_names_are_recorded_in_attrs(_wl_env):
    f = sc.screen_watchlist("2026-08-25", watchlist=["FINE", "GHOST"], log=lambda m: None)
    assert list(f.index) == ["FINE"]
    assert f.attrs["missing"] == ["GHOST"]


# ---------------------------------------------------------------------------
# bases: the yardsticks in the watchlist
# ---------------------------------------------------------------------------

def _base_factors():
    # SPY +5% on the month; LEADER beats it, LAGGARD trails it, FALLEN is
    # below its 200-day but outperforming — the case the excess column exists
    # to separate from a name that is simply broken.
    return pd.DataFrame({
        "price":         [500.0, 300.0, 100.0, 100.0, 100.0],
        "ret_1m":        [0.05,  0.04,  0.15,  -0.05, 0.20],
        "ret_3m":        [0.10,  0.08,  0.30,  -0.10, -0.25],
        "dollar_vol_50": [5e8]  * 5,
        "rows":          [250]  * 5,
        "rvol_20":       [0.4,   0.4,   0.5,   0.5,   0.5],
        "ext_200":       [0.05,  0.05,  0.20, -0.02, -0.15],
        "above_200":     [True,  True,  True,  True,  False],
        "above_50":      [True,  True,  True,  False, True],
    }, index=["SPY", "QQQM", "LEADER", "LAGGARD", "FALLEN"])


@pytest.fixture
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "download_panel", lambda *a, **k: {"Close": pd.DataFrame()})
    monkeypatch.setattr(sc, "compute_factors", lambda panel, **k: _base_factors())


_WL = {"SPY": "base", "QQQM": "base", "LEADER": "semi",
       "LAGGARD": "semi", "FALLEN": "tech"}


@pytest.mark.unit
def test_a_base_in_the_watchlist_is_not_dropped_as_scaffolding(_base_env):
    # SPY is downloaded for the relative-strength factor and normally dropped.
    # When it is also a name being watched, dropping it deletes a row that was
    # explicitly asked for.
    f = sc.screen_watchlist("2026-08-25", watchlist=_WL, log=lambda m: None)
    assert "SPY" in f.index
    assert sorted(f.attrs["bases"]) == ["QQQM", "SPY"]


@pytest.mark.unit
def test_excess_return_is_measured_against_every_base(_base_env):
    f = sc.screen_watchlist("2026-08-25", watchlist=_WL, log=lambda m: None)
    assert f.loc["LEADER", "vs_SPY_1m"] == pytest.approx(0.10)    # 15% - 5%
    assert f.loc["LEADER", "vs_QQQM_1m"] == pytest.approx(0.11)   # 15% - 4%
    assert f.loc["LAGGARD", "vs_SPY_1m"] == pytest.approx(-0.10)  # -5% - 5%


@pytest.mark.unit
def test_a_base_is_not_measured_against_itself(_base_env):
    # Zero would read as "in line with the market"; it is not a comparison.
    import math as _math
    f = sc.screen_watchlist("2026-08-25", watchlist=_WL, log=lambda m: None)
    assert _math.isnan(f.loc["SPY", "vs_SPY_1m"])
    assert f.loc["SPY", "vs_QQQM_1m"] == pytest.approx(0.01)


@pytest.mark.unit
def test_a_base_is_never_judged_by_the_screen(_base_env):
    # Reporting "SPY failed the momentum screen" says nothing about SPY and
    # inflates the count of watched names in trouble.
    f = sc.screen_watchlist("2026-08-25", watchlist=_WL, log=lambda m: None)
    assert bool(f.loc["SPY", "passes_filter"]) is True
    assert f.loc["SPY", "fail_reason"] == ""
    assert bool(f.loc["SPY", "is_base"]) is True
    assert bool(f.loc["LEADER", "is_base"]) is False


@pytest.mark.unit
def test_outperforming_while_below_the_200_day_is_distinguishable(_base_env):
    # The whole reason for the column: FALLEN fails the trend filter and is
    # still beating the market. That is a different situation from LAGGARD,
    # which passes the filter and is losing to it.
    f = sc.screen_watchlist("2026-08-25", watchlist=_WL, log=lambda m: None)
    assert bool(f.loc["FALLEN", "passes_filter"]) is False
    assert f.loc["FALLEN", "vs_SPY_1m"] > 0
    assert bool(f.loc["LAGGARD", "passes_filter"]) is True
    assert f.loc["LAGGARD", "vs_SPY_1m"] < 0


@pytest.mark.unit
def test_changing_the_watchlist_does_not_reuse_the_old_panel(monkeypatch, tmp_path):
    # Cached on the date alone, adding a symbol returned the earlier panel and
    # the new name silently never appeared.
    monkeypatch.setattr(sc, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(sc, "compute_factors", lambda panel, **k: _base_factors())
    seen = []

    def fake_download(tickers, date, **kw):
        seen.append(kw.get("cache_key"))
        return {"Close": pd.DataFrame()}

    monkeypatch.setattr(sc, "download_panel", fake_download)
    sc.screen_watchlist("2026-08-25", watchlist=["LEADER"], log=lambda m: None)
    sc.screen_watchlist("2026-08-25", watchlist=["LEADER", "LAGGARD"], log=lambda m: None)
    assert seen[0] != seen[1], "a changed watchlist must not share a cache key"
