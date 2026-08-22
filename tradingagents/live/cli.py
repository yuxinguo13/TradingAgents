"""Command surface for the live desk.

    python -m tradingagents.live.cli login       # one-time: sign in by hand
    python -m tradingagents.live.cli calibrate   # dump the ticket's controls
    python -m tradingagents.live.cli portfolio   # read the account
    python -m tradingagents.live.cli news NVDA MU
    python -m tradingagents.live.cli scan        # triggers only, no trading
    python -m tradingagents.live.cli trade NVDA Buy 5 --dry-run
    python -m tradingagents.live.cli run --dry-run
    python -m tradingagents.live.cli stop        # engage the kill switch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from . import clock


def _log(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_login(a) -> int:
    from .investopedia import InvestopediaBroker
    with InvestopediaBroker(headless=False, slow_mo=50) as b:
        if b.is_logged_in():
            print("Already signed in — the saved browser profile is still valid.")
            return 0
        return 0 if b.login_interactive(a.wait) else 1


def cmd_calibrate(a) -> int:
    from .investopedia import InvestopediaBroker, artifact_dir
    with InvestopediaBroker(headless=a.headless) as b:
        out = b.calibrate()
    print(f"logged in: {out['logged_in']}   url: {out['url']}")
    print(f"{len(out['controls'])} interactive controls → {artifact_dir()/'calibration.json'}")
    for c in out["controls"][:40]:
        bits = [f"{c['tag']}{'/' + c['type'] if c['type'] else ''}"]
        for k in ("name", "id", "placeholder", "aria", "testid", "text"):
            if c.get(k):
                bits.append(f"{k}={c[k]!r}")
        print("  " + "  ".join(bits))
    return 0


def cmd_discover(a) -> int:
    from .investopedia import InvestopediaBroker, artifact_dir
    with InvestopediaBroker(headless=a.headless) as b:
        if not b.is_logged_in():
            print("Not signed in. Run: python -m tradingagents.live.cli login")
            return 1
        out = b.discover_api(a.seconds)
    print(f"{out['captured']} XHR calls, {len(out['endpoints'])} unique "
          f"→ {artifact_dir()/'api_discovery.json'}\n")
    for e in out["endpoints"]:
        auth = " [bearer]" if e["bearer"] else ""
        print(f"  {e['status']} {e['method']:<5}{e['url'][:120]}{auth}")
    return 0


def cmd_portfolio(a) -> int:
    from .broker import configured_venue, open_broker
    venue = configured_venue()
    with open_broker(venue, headless=a.headless) as b:
        if not b.is_logged_in():
            print(_not_ready(venue))
            return 1
        acct = b.account()
        orders = b.open_orders() if hasattr(b, "open_orders") else []
    invested = sum(h.market_value for h in acct.holdings)
    unreal = sum(h.unrealized for h in acct.holdings)
    print(f"[{venue}]  account ${acct.account_value:,.2f}   cash ${acct.cash:,.2f}   "
          f"buying power ${acct.buying_power:,.2f}")
    if not acct.holdings:
        print("(no open positions)")
    else:
        print(f"\n{'Symbol':<8}{'Side':<7}{'Qty':>10}{'Cost':>11}{'Last':>11}"
              f"{'Value':>13}{'P&L':>13}{'P&L%':>9}")
        print("-" * 82)
        for h in sorted(acct.holdings, key=lambda x: -x.market_value):
            pct = (h.unrealized / (h.avg_cost * h.quantity)) if h.avg_cost and h.quantity else 0.0
            print(f"{h.symbol:<8}{h.side:<7}{h.quantity:>10g}{h.avg_cost:>11,.2f}"
                  f"{h.last:>11,.2f}{h.market_value:>13,.2f}{h.unrealized:>+13,.2f}"
                  f"{pct:>+9.2%}")
        print("-" * 82)
        print(f"{'TOTAL':<8}{'':<7}{'':>10}{'':>11}{'':>11}{invested:>13,.2f}"
              f"{unreal:>+13,.2f}{(unreal/(invested-unreal) if invested-unreal else 0):>+9.2%}")
    if orders:
        print(f"\nOpen orders ({len(orders)}):")
        for o in orders[:10]:
            print(f"  {str(o.get('symbol','?')):<8}{str(o.get('side','')):<6}"
                  f"{o.get('qty',0):>8g}  {o.get('type',''):<8}{o.get('status','')}")
    return 0


def _not_ready(venue: str) -> str:
    if venue == "alpaca":
        return ("Alpaca is not configured.\n"
                "  1. Sign up (free, instant): https://app.alpaca.markets/signup\n"
                "  2. Switch to Paper Trading, then Home > API Keys > Generate\n"
                "  3. Add to .env:  ALPACA_API_KEY=PK...   ALPACA_SECRET_KEY=...")
    return "Not signed in. Run: python -m tradingagents.live.cli login"


def cmd_news(a) -> int:
    from .newsfeed import NewsMonitor, format_items
    m = NewsMonitor()
    items = m.poll([t.upper() for t in a.tickers], macro=not a.no_macro)
    print(format_items(items, a.limit))
    print(f"\n{len(items)} new item(s)")
    return 0


def cmd_scan(a) -> int:
    """Everything the loop does except deciding and trading."""
    from .brain import build_evidence, snapshot, triggers
    from .broker import configured_venue, open_broker
    from .monitor import LiveDesk, MonitorConfig
    from .newsfeed import format_items

    desk = LiveDesk(MonitorConfig(headless=a.headless, screen_top=a.top,
                                  auto_screen=not a.no_screen))
    data_date = clock.last_trading_day().isoformat()
    with open_broker(headless=a.headless) as b:
        if not b.is_logged_in():
            print(_not_ready(configured_venue()))
            return 1
        acct = b.account()
    desk.refresh_screen(data_date)
    cover = desk.build_coverage(acct)
    print(f"coverage ({len(cover)}): {', '.join(cover)}\n")

    items = desk.news.poll(cover, macro=True)
    print(format_items(items, 15), "\n")

    any_fired = False
    for sym in cover:
        snap = snapshot(sym, data_date)
        trg = triggers(sym, snap, items, acct, desk.screen_rank.get(sym))
        if trg:
            any_fired = True
            print(f"** {sym}")
            for t in trg:
                print(f"   [{t.urgency}] {t.kind}: {t.detail}")
    if not any_fired:
        print("no triggers")
    return 0


def cmd_trade(a) -> int:
    from .broker import configured_venue, open_broker
    from .secretary import Secretary
    action = {"buy": "Buy", "sell": "Sell", "short": "Sell Short",
              "cover": "Buy to Cover"}.get(a.action.lower(), a.action.title())
    with open_broker(headless=a.headless) as b:
        if not b.is_logged_in():
            print(_not_ready(configured_venue()))
            return 1
        acct = b.account()
        price = b.quote(a.symbol.upper()) or 0.0
        sec = Secretary()
        from .secretary import Order
        order = Order(symbol=a.symbol.upper(), action=action, quantity=a.quantity,
                      source="manual", rationale="manual CLI order")
        state = clock.market_state()
        v = sec.check(order, acct, price, market_open=state.is_tradeable)
        if not v.ok:
            print(f"BLOCKED: {v.reason}")
            return 1
        print(f"approved: {v.reason} → {v.order.action} {v.order.quantity} "
              f"{v.order.symbol} @ ~{price:,.2f}")
        res = b.place_order(v.order.symbol, v.order.action, v.order.quantity,
                            dry_run=a.dry_run)
        sec.ledger.record(v.order, price, res.ok, res.message)
    print(f"{'OK' if res.ok else 'FAILED'}: {res.message}")
    if res.artifact:
        print(f"artifact: {res.artifact}")
    return 0 if res.ok else 1


def cmd_run(a) -> int:
    from .monitor import LiveDesk, MonitorConfig, build_llm
    from .secretary import RiskLimits
    cfg = MonitorConfig(
        open_interval=a.interval, dry_run=a.dry_run, headless=a.headless,
        screen_top=a.top, max_panels_per_cycle=a.max_panels,
        auto_screen=not a.no_screen, trade_when_closed=a.trade_when_closed,
    )
    llm = None
    if not a.no_llm:
        try:
            llm = build_llm()
        except Exception as exc:
            print(f"LLM unavailable ({exc}); running in monitor-only mode.")
    desk = LiveDesk(cfg, limits=RiskLimits.from_env(), llm=llm)
    desk.run(once=a.once, max_cycles=a.max_cycles)
    return 0


def cmd_status(a) -> int:
    from .secretary import RiskLimits, TradeLedger, kill_switch_engaged, kill_switch_path
    st = clock.market_state()
    print(f"market: {st.session} ({st.phase()})   {st.now:%Y-%m-%d %H:%M %Z}")
    if not st.is_open:
        print(f"next open: {clock.next_open():%Y-%m-%d %H:%M %Z} "
              f"(in {clock.seconds_until_open()/3600:.1f}h)")
    print(f"kill switch: {'ENGAGED' if kill_switch_engaged() else 'clear'} "
          f"({kill_switch_path()})")
    led = TradeLedger()
    print(f"today: {led.trades_today()} trades, ${led.turnover_today():,.0f} turnover")
    lim = RiskLimits.from_env()
    print("\nrisk limits:")
    for k, v in lim.__dict__.items():
        print(f"  {k:<26}{v}")
    for e in led.today()[-10:]:
        print(f"  {e['at'][11:19]}  {e['action']:<12}{e['quantity']:>6} {e['symbol']:<7}"
              f"@{e['price']:>9,.2f}  {'ok' if e['ok'] else e['message'][:40]}")
    return 0


def cmd_stop(a) -> int:
    from .secretary import kill_switch_path
    p = kill_switch_path()
    if a.clear:
        p.unlink(missing_ok=True)
        print(f"kill switch cleared ({p})")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"engaged {date.today()}\n", encoding="utf-8")
        print(f"kill switch ENGAGED — no orders will be placed ({p})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="live", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--show-browser", dest="headless", action="store_false",
                   default=True, help="run the browser visibly")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("login", help="sign in to Investopedia (opens a window)")
    s.add_argument("--wait", type=int, default=300)
    s.set_defaults(fn=cmd_login)

    s = sub.add_parser("calibrate", help="dump the trade ticket's controls")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("discover", help="capture the simulator's own JSON API calls")
    s.add_argument("--seconds", type=int, default=25)
    s.set_defaults(fn=cmd_discover)

    s = sub.add_parser("portfolio", help="read the simulator account")
    s.set_defaults(fn=cmd_portfolio)

    s = sub.add_parser("news", help="poll the news feeds")
    s.add_argument("tickers", nargs="*", default=[])
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--no-macro", action="store_true")
    s.set_defaults(fn=cmd_news)

    s = sub.add_parser("scan", help="coverage + news + triggers, no trading")
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--no-screen", action="store_true")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("trade", help="place one order by hand (through the risk gate)")
    s.add_argument("symbol")
    s.add_argument("action", choices=["buy", "sell", "short", "cover"])
    s.add_argument("quantity", type=int)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_trade)

    s = sub.add_parser("run", help="start the always-on loop")
    s.add_argument("--interval", type=int, default=120, help="seconds between cycles")
    s.add_argument("--dry-run", action="store_true", help="never submit an order")
    s.add_argument("--once", action="store_true", help="one cycle, then exit")
    s.add_argument("--max-cycles", type=int, default=None)
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--max-panels", type=int, default=4)
    s.add_argument("--no-screen", action="store_true")
    s.add_argument("--no-llm", action="store_true", help="monitor only, never decide")
    s.add_argument("--trade-when-closed", action="store_true",
                   help="fill at the last close outside market hours "
                        "(local paper book only; a real venue will reject)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="clock, limits, kill switch, today's trades")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("stop", help="engage (or --clear) the kill switch")
    s.add_argument("--clear", action="store_true")
    s.set_defaults(fn=cmd_stop)

    a = p.parse_args(argv)
    _log(a.verbose)
    try:
        return a.fn(a)
    except Exception as exc:
        # Missing credentials is the expected first-run state, not a crash. A
        # traceback here would bury the one line that says what to do about it.
        if type(exc).__name__ == "MissingCredentials":
            print(f"\n{exc}\n")
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
