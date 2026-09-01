"""Price shape, drawn and read — the half of a report a table cannot carry.

A row of numbers says where a name is. It does not say how it got there, and
those are different facts: ``+2.5% 1M`` is the same cell for a name that ground
up every day and for one that fell 12% and then took it all back in four
sessions. The second is a very different trade, and the table hides it.

So two things live here, and they are deliberately separate:

* **The drawing.** :func:`sparkline` for a table cell, :func:`line_chart` for a
  page. Both are plain text, because the report is a markdown file a human
  reads in a terminal or on a phone, and a PNG in that pipeline is a file that
  does not open. The chart draws the stop and the target as levels on the same
  axis as the price, which is the only place in the whole report where the
  reader can see the distance they are being asked to accept.
* **The reading.** :func:`read_trend` turns the same bars into named facts —
  均线排列, swing structure, 动能, 波动, 位置, 支撑/阻力 — and then one sentence.

The reading is mechanical and says so. Every threshold in it (RSI 55, a pivot's
±3 bars, "加速" at 1.2× the quarter's pace) is a convention written down in this
file, not a result anyone established, and the verdict line is a summary of the
bullets above it rather than a forecast. A chart that is read out loud with
confidence is worth less than one read out loud with its assumptions attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

# Eight levels, low to high. Wider than this and the cell stops being scannable
# at a glance, which is the only thing a sparkline is for.
_BLOCKS = "▁▂▃▄▅▆▇█"

# Box-drawing set used by line_chart. Kept as a table so a terminal that cannot
# render them can be given an ASCII fallback in one place.
_GLYPH = {"flat": "─", "up_end": "╭", "up_start": "╯",
          "down_end": "╰", "down_start": "╮", "riser": "│"}


def is_num(value) -> bool:
    """True for anything that converts to a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _f(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clean(values) -> list[float | None]:
    """A series with gaps preserved. A hole is not a zero and must not plot as one."""
    return [float(v) if is_num(v) else None for v in (values or [])]


def bucket(values, width: int) -> list[float | None]:
    """Downsample to ``width`` points by averaging within each bucket.

    Averaging rather than sampling every n-th bar on purpose: a sampled series
    can miss the one bar that made the move, and a chart that omits the gap it
    is being read for is worse than no chart. The cost is that a single spike is
    flattened into its neighbours, which is the right trade for a shape.
    """
    xs = _clean(values)
    if width <= 0 or len(xs) <= width:
        return xs
    out: list[float | None] = []
    n = len(xs)
    for i in range(width):
        lo = int(i * n / width)
        hi = max(lo + 1, int((i + 1) * n / width))
        chunk = [x for x in xs[lo:hi] if x is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def sma(values, window: int) -> list[float | None]:
    """Simple moving average, aligned to the input, None until the window fills."""
    xs = _clean(values)
    out: list[float | None] = [None] * len(xs)
    if window <= 0:
        return out
    run = 0.0
    filled = 0
    for i, x in enumerate(xs):
        if x is not None:
            run += x
            filled += 1
        if i >= window:
            old = xs[i - window]
            if old is not None:
                run -= old
                filled -= 1
        if i >= window - 1 and filled == window:
            out[i] = run / window
    return out


def sparkline(values, width: int = 0) -> str:
    """A one-cell shape. Empty when there is not enough to draw.

    Empty rather than a flat line: a table cell showing ``▄▄▄▄▄▄▄▄`` for a name
    with two bars of history reads as "went nowhere", which is a claim the data
    does not support.
    """
    xs = bucket(values, width) if width else _clean(values)
    pts = [x for x in xs if x is not None]
    if len(pts) < 2:
        return ""
    lo, hi = min(pts), max(pts)
    span = hi - lo
    steps = len(_BLOCKS) - 1
    out = []
    for x in xs:
        if x is None:
            out.append(" ")
        elif span <= 0:
            out.append(_BLOCKS[steps // 2])
        else:
            out.append(_BLOCKS[int(round((x - lo) / span * steps))])
    return "".join(out)


# ---------------------------------------------------------------------------
# the chart
# ---------------------------------------------------------------------------

def _fmt_axis(value: float, span: float) -> str:
    """Enough decimals to separate adjacent rows, and no more."""
    if span >= 100:
        return f"{value:,.0f}"
    if span >= 10:
        return f"{value:,.1f}"
    if span >= 1:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def line_chart(closes, *, height: int = 12, width: int = 72,
               overlays: dict | None = None, levels: dict | None = None,
               dates=None, title: str = "") -> list[str]:
    """The price line, with moving averages under it and levels across it.

    ``overlays`` are drawn beneath the price (a moving average that erases the
    close it is averaging is a chart that lies about its own subject) and
    ``levels`` — stop, entry, target — are drawn as dashed rules with the label
    on the right. Those three are why this function exists in a report that
    already has a table of them: on the page the stop is a number, and on the
    chart it is a distance the reader can see against the name's own range.

    Returns a list of lines to drop inside a fenced block. Empty when there is
    nothing plottable, which the caller must render as an absence rather than
    an empty box.
    """
    price = bucket(closes, width)
    pts = [x for x in price if x is not None]
    if len(pts) < 2:
        return []

    over = {}
    for name, series in (overlays or {}).items():
        got = bucket(series, width)
        if any(x is not None for x in got):
            over[name] = got

    lvl = {k: _f(v) for k, v in (levels or {}).items()}
    lvl = {k: v for k, v in lvl.items() if math.isfinite(v)}

    seen = list(pts)
    for series in over.values():
        seen += [x for x in series if x is not None]
    seen += list(lvl.values())
    lo, hi = min(seen), max(seen)
    if hi <= lo:                       # a perfectly flat window still deserves a line
        pad = abs(hi) * 0.01 or 1.0
        lo, hi = lo - pad, hi + pad
    span = hi - lo

    rows = max(2, height - 1)
    n = len(price)
    ratio = rows / span

    def y_of(value: float) -> int:
        return max(0, min(rows, int(round((value - lo) * ratio))))

    grid = [[" "] * n for _ in range(rows + 1)]

    def put(y: int, x: int, ch: str, force: bool = False) -> None:
        if 0 <= x < n and (force or grid[rows - y][x] == " "):
            grid[rows - y][x] = ch

    # Levels first, then averages, then the close: later layers overwrite.
    right: dict[int, list[str]] = {}
    for label, value in sorted(lvl.items(), key=lambda kv: -kv[1]):
        y = y_of(value)
        for x in range(n):
            put(y, x, "┈")
        right.setdefault(y, []).append(f"{label} {_fmt_axis(value, span)}")

    for mark, series in zip("·:", over.values()):
        for x, v in enumerate(series):
            if v is not None:
                put(y_of(v), x, mark)

    ys = [None if v is None else y_of(v) for v in price]
    for x, y in enumerate(ys):
        if y is None:
            continue
        prev = ys[x - 1] if x else None
        if prev is None or prev == y:
            put(y, x, _GLYPH["flat"], force=True)
            continue
        if prev < y:
            put(y, x, _GLYPH["up_end"], force=True)
            put(prev, x, _GLYPH["up_start"], force=True)
        else:
            put(y, x, _GLYPH["down_end"], force=True)
            put(prev, x, _GLYPH["down_start"], force=True)
        for mid in range(min(prev, y) + 1, max(prev, y)):
            put(mid, x, _GLYPH["riser"], force=True)

    labels = [_fmt_axis(lo + (rows - r) * span / rows, span) for r in range(rows + 1)]
    pad = max(len(s) for s in labels)
    out = []
    if title:
        out.append(" " * (pad + 2) + title)
    for r in range(rows + 1):
        line = f"{labels[r]:>{pad}} ┤{''.join(grid[r])}"
        tail = right.get(rows - r)
        if tail:
            line += "  ← " + " / ".join(tail)
        out.append(line.rstrip())

    out.append(" " * pad + " └" + "─" * n)
    stamps = [str(d) for d in (dates or []) if str(d).strip()]
    if len(stamps) >= 2:
        first, last = stamps[0], stamps[-1]
        gap = max(0, n - len(first) - len(last))
        # Three stamps only when the middle one has room to sit between the
        # other two; two dates on a narrow chart is a legible axis, three
        # overlapping ones is not.
        mid = stamps[len(stamps) // 2] if len(stamps) >= 5 else ""
        body = first + (f"{mid:^{gap}}" if mid and gap > len(mid) + 2
                        else " " * gap) + last
        out.append(" " * (pad + 2) + body[:n])
    # Only name the overlays that were actually drawn, and in the order the
    # glyphs were assigned above.
    if over:
        key = "   ".join(f"{m} {name}" for m, name in zip("·:", over.keys()))
        out.append(" " * (pad + 2) + key + "   ─ close")
    return out


# ---------------------------------------------------------------------------
# swing structure
# ---------------------------------------------------------------------------

def pivots(highs, lows, k: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Fractal turning points: a bar that is the extreme of its ±k neighbours.

    ``k`` is a convention. Smaller finds more turns and calls noise a structure;
    larger finds fewer and misses the turn that matters. Three is the smallest
    value that ignores a single outlier bar, which is the failure worth avoiding
    on daily data.
    """
    hs, ls = _clean(highs), _clean(lows)
    ph: list[tuple[int, float]] = []
    pl: list[tuple[int, float]] = []
    for i in range(k, len(hs) - k):
        h = hs[i]
        if h is not None:
            window = [x for x in hs[i - k:i + k + 1] if x is not None]
            if window and h >= max(window) and h > min(window):
                ph.append((i, h))
    for i in range(k, len(ls) - k):
        l = ls[i]
        if l is not None:
            window = [x for x in ls[i - k:i + k + 1] if x is not None]
            if window and l <= min(window) and l < max(window):
                pl.append((i, l))
    return ph, pl


def _thin(points: list[tuple[int, float]], gap: int, *, highs: bool
          ) -> list[tuple[int, float]]:
    """Collapse pivots closer than ``gap`` bars, keeping the more extreme one.

    ``highs`` picks the direction of "more extreme": the higher of two swing
    highs, the lower of two swing lows. Getting this backwards produces a swing
    low that is not the low of its own cluster, and every support printed off it
    is then a level the price already traded through.
    """
    out: list[tuple[int, float]] = []
    for idx, px in points:
        if out and idx - out[-1][0] < gap:
            better = px > out[-1][1] if highs else px < out[-1][1]
            if better:
                out[-1] = (idx, px)
            continue
        out.append((idx, px))
    return out


def structure(highs, lows, k: int = 3, gap: int = 5) -> tuple[str, str]:
    """The last two swing highs and lows, named. ``(label, detail)``.

    This is the only part of the read that looks at shape rather than at a
    single latest value, and it is the part that separates "up 8% this month"
    from "up 8% this month on a series of higher lows".
    """
    ph, pl = pivots(highs, lows, k)
    ph, pl = _thin(ph, gap, highs=True), _thin(pl, gap, highs=False)
    if len(ph) < 2 or len(pl) < 2:
        return "结构未成形", "摆动高低点不足两组，无法判断结构（历史太短或波动太小）"
    h1, h0 = ph[-2][1], ph[-1][1]
    l1, l0 = pl[-2][1], pl[-1][1]
    hh, hl = h0 > h1, l0 > l1
    detail = (f"前高 {h1:,.2f} → {h0:,.2f}（{(h0 / h1 - 1) * 100:+.1f}%）、"
              f"前低 {l1:,.2f} → {l0:,.2f}（{(l0 / l1 - 1) * 100:+.1f}%）"
              if h1 and l1 else "")
    if hh and hl:
        return "上升结构", f"更高的高点 + 更高的低点；{detail}"
    if not hh and not hl:
        return "下降结构", f"更低的高点 + 更低的低点；{detail}"
    if hh and not hl:
        return "扩张震荡", f"高点抬高但低点下移，波动在放大；{detail}"
    return "收敛整理", f"高点走低但低点抬高，区间在收窄；{detail}"


def levels_near(price: float, highs, lows, k: int = 3, gap: int = 5
                ) -> tuple[float, float]:
    """Nearest swing low below and swing high above. NaN when there is none.

    A support below the price and a resistance above it are the two numbers a
    reader checks a stop and a target against, and they are the two the table
    never prints.
    """
    ph, pl = pivots(highs, lows, k)
    ph, pl = _thin(ph, gap, highs=True), _thin(pl, gap, highs=False)
    below = [p for _, p in pl if p < price]
    above = [p for _, p in ph if p > price]
    return (max(below) if below else float("nan"),
            min(above) if above else float("nan"))


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------

@dataclass
class TrendRead:
    """One name's shape, in named facts. Every field is prose, already formatted."""

    symbol: str = ""
    price: float = float("nan")
    ma_stack: str = ""
    ma_detail: str = ""
    structure: str = ""
    structure_detail: str = ""
    momentum: str = ""
    volatility: str = ""
    position: str = ""
    volume: str = ""
    relative: str = ""
    support: float = float("nan")
    resistance: float = float("nan")
    verdict: str = ""
    spark: str = ""
    # Set when the bars were too short to say anything. The caller renders the
    # absence; it must never render an empty read as a neutral one.
    error: str = ""

    def bullets(self) -> list[tuple[str, str]]:
        """``(label, text)`` pairs, in reading order, skipping what is absent."""
        pairs = [
            ("均线排列", f"{self.ma_stack} — {self.ma_detail}" if self.ma_detail else self.ma_stack),
            ("形态结构", f"{self.structure} — {self.structure_detail}"
             if self.structure_detail else self.structure),
            ("动能", self.momentum),
            ("波动", self.volatility),
            ("位置", self.position),
            ("量能", self.volume),
            ("相对强度", self.relative),
        ]
        return [(k, v) for k, v in pairs if v and v.strip(" —")]


def _ma_stack(price, s20, s50, s200) -> tuple[str, str]:
    have = [x for x in (s20, s50, s200) if is_num(x)]
    if not is_num(price) or not have:
        return "均线不足", ""
    bits = []
    for label, value in (("20 日", s20), ("50 日", s50), ("200 日", s200)):
        if is_num(value) and value:
            bits.append(f"{label} {value:,.2f}（{(price / value - 1) * 100:+.1f}%）")
    detail = "现价距 " + "、".join(bits) if bits else ""
    if all(is_num(x) for x in (s20, s50, s200)):
        if price > s20 > s50 > s200:
            return "多头排列", detail
        if price < s20 < s50 < s200:
            return "空头排列", detail
    if is_num(s200) and price > s200 and is_num(s50) and s50 > s200:
        return "均线偏多", detail
    if is_num(s200) and price < s200:
        return "跌破 200 日线", detail
    return "均线纠缠", detail


def _momentum(rsi, ret_1m, ret_3m) -> str:
    parts = []
    if is_num(rsi):
        zone = ("超买区" if rsi >= 70 else "偏强" if rsi >= 55
                else "中性" if rsi >= 45 else "偏弱" if rsi >= 30 else "超卖区")
        parts.append(f"RSI(14) {rsi:.0f}，{zone}")
    if is_num(ret_1m) and is_num(ret_3m):
        # Pace, not level: a quarter's return spread over 63 sessions against a
        # month's over 21. Comparing the raw returns would call every rising
        # name "accelerating" merely because the quarter is longer.
        p1, p3 = ret_1m / 21.0, ret_3m / 63.0
        if p3 > 0 and p1 > p3 * 1.2:
            parts.append(f"近一月的斜率快于近三月（月 {ret_1m * 100:+.1f}% vs 季 {ret_3m * 100:+.1f}%），趋势在加速")
        elif p3 > 0 and p1 < p3 * 0.8:
            parts.append(f"近一月的斜率慢于近三月（月 {ret_1m * 100:+.1f}% vs 季 {ret_3m * 100:+.1f}%），涨势在放缓")
        elif p3 <= 0 and p1 > 0:
            parts.append(f"季线仍为负（{ret_3m * 100:+.1f}%）但近一月已转正（{ret_1m * 100:+.1f}%），属于反弹初期而非既成趋势")
        else:
            parts.append(f"月 {ret_1m * 100:+.1f}%、季 {ret_3m * 100:+.1f}%，节奏平稳")
    return "；".join(parts)


def _volatility(atr_pct, k: float = 2.0) -> str:
    if not is_num(atr_pct) or atr_pct <= 0:
        return ""
    return (f"日均真实波幅约为价格的 {atr_pct * 100:.1f}%；按本报告 {k:g} ATR 的止损惯例，"
            f"一个 R 相当于 {atr_pct * k * 100:.1f}% 的价格距离")


def _position(off_high, off_low) -> str:
    parts = []
    if is_num(off_high):
        parts.append(f"距 52 周高点 {off_high * 100:+.1f}%")
    if is_num(off_low):
        parts.append(f"距 52 周低点 {off_low * 100:+.1f}%")
    return "，".join(parts)


def _volume(vol_ratio) -> str:
    if not is_num(vol_ratio) or vol_ratio <= 0:
        return ""
    if vol_ratio >= 2:
        return f"最新一日成交量为 20 日均量的 {vol_ratio:.1f} 倍，属于放量"
    if vol_ratio <= 0.6:
        return f"最新一日成交量仅为 20 日均量的 {vol_ratio:.1f} 倍，缩量"
    return f"最新一日成交量为 20 日均量的 {vol_ratio:.1f} 倍，量能正常"


def _verdict(stack: str, struct: str, off_high, rsi) -> str:
    """One sentence, assembled from the bullets above. Not a forecast.

    The wording is deliberately conditional. Every input is a lagging
    description of what already happened, and a chart read that says "will"
    about any of them is claiming something none of these numbers contain.
    """
    if struct == "结构未成形":
        # No contradiction to report: the swing test simply had nothing to
        # measure. Calling that "均线与结构互相矛盾" would invent a conflict.
        return (f"均线读数是{stack}，但摆动高低点不足两组，形态结构无从判断——"
                f"只有均线一个依据时，不要把仓位当作趋势确认后的仓位来下")
    up = stack in ("多头排列", "均线偏多") and struct in ("上升结构", "收敛整理")
    down = stack in ("空头排列", "跌破 200 日线") or struct == "下降结构"
    if up:
        head = "趋势向上且结构完整"
        if is_num(off_high) and off_high > -0.05:
            head += "，且贴近 52 周高点——追高的代价是止损离得远"
        elif is_num(rsi) and rsi >= 70:
            head += "，但动能已进入超买区，短期回撤的概率高于起涨时"
        return head + "；顺势的一侧在上方，风险在于结构一旦破位就要认"
    if down:
        return ("趋势向下或已跌破长期均线；在结构重新出现更高的低点之前，"
                "任何买入都是在与自己的图形对赌")
    return "均线与结构互相矛盾，属于震荡；这种形态里止损容易被扫，仓位应比顺势时更小"


def read_trend(symbol: str, closes, highs=None, lows=None, volumes=None, *,
               rsi=float("nan"), atr_pct=float("nan"), vol_ratio=float("nan"),
               ret_1m=float("nan"), ret_3m=float("nan"),
               benchmark: dict | None = None, atr_stop_mult: float = 2.0,
               min_bars: int = 30) -> TrendRead:
    """Bars in, named facts out. Never raises; an unreadable series sets ``error``."""
    out = TrendRead(symbol=symbol)
    cs = [x for x in _clean(closes) if x is not None]
    if len(cs) < min_bars:
        out.error = f"历史不足 {min_bars} 根 K 线（拿到 {len(cs)} 根），不做图形判断"
        return out

    price = cs[-1]
    out.price = price
    out.spark = sparkline(cs[-63:], width=24)

    s20 = sma(cs, 20)[-1]
    s50 = sma(cs, 50)[-1]
    s200 = sma(cs, 200)[-1] if len(cs) >= 200 else None
    out.ma_stack, out.ma_detail = _ma_stack(price, s20, s50, s200)

    hs = list(highs) if highs is not None else cs
    ls = list(lows) if lows is not None else cs
    window = 126                        # half a year: long enough for two swings
    out.structure, out.structure_detail = structure(hs[-window:], ls[-window:])
    out.support, out.resistance = levels_near(price, hs[-window:], ls[-window:])

    if not is_num(ret_1m) and len(cs) > 22:
        ret_1m = cs[-1] / cs[-22] - 1.0
    if not is_num(ret_3m) and len(cs) > 64:
        ret_3m = cs[-1] / cs[-64] - 1.0
    out.momentum = _momentum(rsi, ret_1m, ret_3m)
    out.volatility = _volatility(atr_pct, atr_stop_mult)

    year = cs[-252:]
    hi, low = max(year), min(year)
    out.position = _position(price / hi - 1 if hi else float("nan"),
                             price / low - 1 if low else float("nan"))
    out.volume = _volume(vol_ratio)

    if benchmark:
        bits = []
        for name, excess in benchmark.items():
            if is_num(excess):
                verb = "跑赢" if excess >= 0 else "跑输"
                bits.append(f"近一月{verb} {name} {abs(excess) * 100:.1f} 个百分点")
        out.relative = "，".join(bits)

    out.verdict = _verdict(out.ma_stack, out.structure, price / hi - 1 if hi else float("nan"), rsi)
    return out
