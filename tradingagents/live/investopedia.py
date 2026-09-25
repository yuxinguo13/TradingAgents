"""Investopedia simulator adapter, driven through a real browser.

Investopedia has no public API, blocks non-browser HTTP clients outright
(a bare request to the trade page returns 403 behind a JS challenge), and gates
the simulator behind a login. The published Python "simulator APIs" on GitHub
all predate the 2021 rewrite and target endpoints that no longer exist. That
leaves exactly one honest option: drive the real site in a real browser.

Three consequences shape this module.

**Login is manual, once.** The browser runs against a persistent profile
directory, so you sign in by hand the first time and the session cookie is
reused thereafter. The agent never sees or stores your password — there is no
credential field anywhere in this codebase, and adding one would mean handling
a password the site is entitled to protect with a captcha anyway.

**The ticket is Vuetify, so nothing is a plain form control.** The simulator is
a Vue + Vuetify app: its "dropdowns" are ``<div class="v-select__selections">``
that open a floating menu, not ``<select>`` elements, and its buttons are
``<span class="v-btn__content">`` inside a div. An earlier version of this file
tried ``select[name='action']`` and ``select_option()`` first. That cannot ever
match — Playwright's ``select_option`` only works on a real ``<select>`` — so
the action and order-type steps silently fell through to a bare text click and
the ticket was submitted with whatever the defaults happened to be. Submitting
a Buy when the panel asked for a Sell Short is the worst bug this module can
have, so both selections are now made through the Vuetify path and *read back*
before the order advances.

**Selectors are tiered, and calibrated when they break.** Every lookup tries a
priority list: ``data-cy`` first (Investopedia ships Cypress test hooks, which
are written for the site's own test suite and therefore survive the CSS
refactors that kill class selectors), then accessible role/label, then the
exact placeholder string, then Vuetify's own classes, and only last a
positional index. Positional selectors are marked as such everywhere they
appear: they encode "the 7th button on the page", which a single new button
above them silently breaks. ``calibrate()`` dumps the live page's controls,
its ``data-cy`` inventory and its ordered dropdown/button lists, so a broken
selector is repaired by reading reality instead of guessing. Any failure writes
a screenshot and an HTML dump next to the log, because a trading bot that fails
silently against a changed page is worse than one that stops.

The DOM facts above were read off a working Selenium bot
(github.com/bassel27/Investopedia-Bot); the tiers below note which strings came
from there and which are guesses that merely cost nothing when absent.

    python -m tradingagents.live.cli login       # one-time, opens a window
    python -m tradingagents.live.cli calibrate   # dump the ticket's controls
    python -m tradingagents.live.cli portfolio
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

TRADE_URL = "https://www.investopedia.com/simulator/trade/stocks"
PORTFOLIO_URL = "https://www.investopedia.com/simulator/portfolio"
OPEN_ORDERS_URL = "https://www.investopedia.com/simulator/trade/open-orders"
HOME_URL = "https://www.investopedia.com/simulator"
# The simulator SPA authenticates against a Keycloak realm; a signed-out
# request to any account page redirects here, which is the crispest
# available signal for "not logged in".
AUTH_HOST = "auth.investopedia.com"

# The order vocabulary and the account/order types live in broker.py, which is
# the venue contract shared with the Alpaca adapter. Re-exported here so the
# existing `from .investopedia import Account, BUY, ...` call sites keep working.
from .broker import (  # noqa: E402  (kept beside the other module constants)
    ACTIONS, BUY, COVER, LIMIT, MARKET, ORDER_TYPES, SELL, SHORT, STOP,
    Account, Holding, OrderResult,
)

# A real UA string: the headless default is fingerprinted and challenged.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


def profile_dir() -> Path:
    """Persistent Chromium profile — this is what keeps you logged in."""
    p = Path(os.getenv("TRADINGAGENTS_BROWSER_PROFILE", _home() / "browser_profile"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifact_dir() -> Path:
    p = _home() / "browser_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _money(text: str | None) -> float:
    """Parse '$12,345.67', '(1,234.00)' (negative), '—' → float."""
    if not text:
        return 0.0
    t = text.strip().replace("−", "-")
    neg = t.startswith("(") and t.endswith(")")
    t = re.sub(r"[^0-9.\-]", "", t)
    if not t or t in {"-", "."}:
        return 0.0
    try:
        v = float(t)
    except ValueError:
        return 0.0
    return -v if neg else v


class SelectionNotApplied(RuntimeError):
    """A Vuetify dropdown read back a different choice than the one clicked.

    Deliberately not a LookupError: the two failures want opposite responses.
    A control that cannot be *found* may still be reachable through a
    lower-tier strategy, so it is worth falling through. A control that
    demonstrably holds the wrong value must stop the order there and then —
    this is the case where carrying on means buying when the panel asked to
    sell short.
    """


def _norm(text: str | None) -> str:
    """Collapse whitespace and case so two rendered labels can be compared."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


class InvestopediaBroker:
    """Browser-backed adapter. Use as a context manager.

    ``headless=False`` is the default for :meth:`login` only; the monitor runs
    headless. Note that a persistent profile cannot be shared by two browsers
    at once, so the monitor and a manual login window must not run together.
    """

    def __init__(self, headless: bool = True, slow_mo: int = 0, timeout: int = 30_000):
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = timeout
        self._pw = None
        self._ctx = None
        self.page = None

    # --- lifecycle ----------------------------------------------------------

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._ctx = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir()),
            headless=self.headless,
            slow_mo=self.slow_mo,
            user_agent=_UA,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._ctx.set_default_timeout(self.timeout)
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        return self

    def __exit__(self, *exc):
        with suppress(Exception):
            self._ctx.close()
        with suppress(Exception):
            self._pw.stop()
        return False

    # --- diagnostics --------------------------------------------------------

    def snapshot(self, tag: str) -> str:
        """Screenshot + HTML dump. Returns the screenshot path."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = artifact_dir() / f"{tag}_{stamp}"
        png = f"{base}.png"
        with suppress(Exception):
            self.page.screenshot(path=png, full_page=True)
        with suppress(Exception):
            Path(f"{base}.html").write_text(self.page.content(), encoding="utf-8")
        logger.warning("saved browser artifact: %s", png)
        return png

    def _goto(self, url: str, wait: str = "domcontentloaded") -> None:
        self.page.goto(url, wait_until=wait, timeout=self.timeout)
        # The simulator is a client-rendered app: DOM-ready fires long before
        # the ticket or the holdings table exists. Settling on network idle is
        # what makes the subsequent locators deterministic instead of racy.
        with suppress(Exception):
            self.page.wait_for_load_state("networkidle", timeout=15_000)

    # --- auth ---------------------------------------------------------------

    def is_logged_in(self) -> bool:
        """True when the portfolio page renders instead of redirecting to auth.

        The simulator authenticates through Keycloak (an OIDC realm at
        ``auth.investopedia.com``), so a signed-out request to any account page
        bounces to that host. Checking the *landing host* is therefore an exact
        test, where scraping the page for the words "sign in" is a guess that
        breaks the moment a marketing banner mentions signing in.
        """
        self._goto(PORTFOLIO_URL)
        if AUTH_HOST in self.page.url:
            return False
        body = (self.page.inner_text("body") or "").lower()
        return not ("continue with email" in body or "sign in to investopedia" in body)

    def login_interactive(self, wait_seconds: int = 300) -> bool:
        """Open a real window and wait for you to sign in by hand.

        Deliberately manual: Investopedia protects sign-in with a challenge,
        and scripting a password entry would mean this tool storing a
        credential it has no business holding.
        """
        self._goto(HOME_URL)
        print("\nA browser window is open. Sign in to Investopedia, then return here.")
        print(f"Waiting up to {wait_seconds}s for the session to appear...\n")
        for _ in range(wait_seconds // 5):
            self.page.wait_for_timeout(5000)
            with suppress(Exception):
                if self.is_logged_in():
                    print("Signed in. Session saved to the browser profile.")
                    return True
        print("Timed out waiting for sign-in.")
        return False

    # --- resilient locators -------------------------------------------------
    #
    # Strategy kinds, in the order they are allowed to appear in a list:
    #
    #   "cy"                Cypress hook, e.g. 'input[data-cy="quantity-input"]'.
    #                       Best tier: these exist to make the site's own tests
    #                       pass, so they outlive class and layout churn.
    #   "role" / "label"    Accessible name. Survives a CSS refactor; only
    #                       breaks when the visible wording changes.
    #   "exact_placeholder" The placeholder string, matched whole.
    #   "placeholder"/"text"/"exact_text"   Loose visible-string matches.
    #   "vuetify"           A Vuetify component class (.v-select__selections,
    #                       .v-btn__content). Framework-stable, app-fragile.
    #   "css"               Anything else.
    #   "nth"               (selector, 1-based index) — POSITIONAL, last resort
    #                       only. It names a place, not a thing, and a single
    #                       new element above it points it at the wrong control.
    #
    # Within a tier a *verified* string (read off a working bot against the
    # live DOM) is listed ahead of a guessed one, so a guess that happens to
    # match some other visible element cannot outrank a known-good selector.

    def _locate(self, kind: str, value):
        """Build a single-element locator for one strategy pair."""
        page = self.page
        if kind == "role":
            return page.get_by_role(value[0], name=value[1]).first
        if kind == "label":
            return page.get_by_label(value, exact=False).first
        if kind == "exact_placeholder":
            return page.get_by_placeholder(value, exact=True).first
        if kind == "placeholder":
            return page.get_by_placeholder(value, exact=False).first
        if kind == "exact_text":
            return page.get_by_text(value, exact=True).first
        if kind == "text":
            return page.get_by_text(value, exact=False).first
        if kind == "nth":
            # The reference bot's XPath indexes were 1-based; keep that
            # numbering so a selector copied from a calibration dump or from
            # devtools transfers without an off-by-one.
            selector, one_based = value
            return page.locator(selector).nth(max(one_based - 1, 0))
        # "cy", "vuetify", "css" are all plain selector strings; they are
        # distinct kinds only so a call site reads as a priority order.
        return page.locator(value).first

    def _first_visible(self, strategies: list, what: str, timeout: int = 6000):
        """Try each locator strategy in order; return the first visible match.

        The order of the list is the whole design — see the tier table above.
        A missing strategy is not an error, only exhaustion is: that is what
        lets the same call site work against a redesigned page.
        """
        for kind, value in strategies:
            with suppress(Exception):
                loc = self._locate(kind, value)
                if loc.is_visible(timeout=timeout):
                    return loc
        raise LookupError(f"could not locate {what} on {self.page.url}")

    def _read_money(self, selector: str) -> float | None:
        """Currency text from a selector, or None when it is not on the page.

        None and 0.0 mean different things here: 0.0 is a real reading, None
        means "this hook is gone, try the next strategy". Collapsing the two
        would make a renamed ``data-cy`` look like an empty account.
        """
        with suppress(Exception):
            txt = self.page.locator(selector).first.inner_text(timeout=2500)
            if txt and any(ch.isdigit() for ch in txt):
                return _money(txt)
        return None

    def _dismiss_overlay(self) -> None:
        """Best-effort click on a welcome/tour overlay, if one is up.

        An overlay does not hide the ticket, it *intercepts the clicks* meant
        for it, so the symptom is a timeout on a control that is plainly
        visible in the failure screenshot. The reference bot dismissed this
        with ``//span[@style="color: rgb(255, 255, 255)"]``, which is not used
        here: an inline white span is a styling coincidence and could just as
        easily be the ticket's own primary button, and clicking Preview on a
        half-filled ticket is worse than leaving a popup up. The wording below
        is a guess, not a verified string; when none of it matches, nothing is
        clicked and the ticket steps report their own failure as usual.
        """
        with suppress(Exception):
            btn = self._first_visible([
                ("role", ("button", re.compile(
                    r"^\s*(got it|no thanks|dismiss|close|maybe later)\s*$", re.I))),
            ], "overlay dismiss", timeout=600)
            btn.click()
            self.page.wait_for_timeout(300)

    # --- Vuetify dropdowns --------------------------------------------------

    def _choose_dropdown(self, option: str, trigger_strategies: list,
                         vocabulary: tuple[str, ...], what: str) -> None:
        """Set a Vuetify select to ``option``.

        Vuetify renders a select as a div that opens a floating menu; there is
        no ``<select>`` to call ``select_option`` on. The sequence is: read the
        current selection, click the trigger to open the menu, click the item
        by its visible text, then read the selection back.

        The read-back is the point. A click that lands on the overlay instead
        of the item leaves the ticket on its default — Buy, Market — and the
        order goes through as something nobody asked for. ``vocabulary`` is the
        set of choices this dropdown offers, so a read-back that shows a
        *different valid choice* is proof of a mis-click and raises. A
        read-back that shows nothing is only a failed read and is logged.
        """
        trigger = self._first_visible(trigger_strategies, f"{what} dropdown")

        before = ""
        with suppress(Exception):
            before = trigger.inner_text(timeout=2000)
        if _norm(before) == _norm(option):
            # Already right (Buy and Market are the ticket's defaults). Not
            # opening the menu also removes the one case where the option text
            # and the closed trigger text are identical and a text click could
            # land on the trigger itself.
            return

        trigger.click()
        self.page.wait_for_timeout(350)
        self._click_menu_item(option, what)
        self.page.wait_for_timeout(350)

        after = ""
        with suppress(Exception):
            after = trigger.inner_text(timeout=2000)
        if _norm(after) == _norm(option):
            return
        if any(_norm(after) == _norm(v) for v in vocabulary):
            raise SelectionNotApplied(
                f"{what} reads {after.strip()!r} after selecting {option!r}")
        logger.warning("could not confirm %s is %r (reads %r)", what, option, after)

    def _click_menu_item(self, option: str, what: str) -> None:
        """Click an item in the open Vuetify menu by its visible text."""
        self._first_visible([
            ("role", ("option", re.compile(rf"^\s*{re.escape(option)}\s*$", re.I))),
            # Vuetify detaches the open menu into an overlay appended to
            # <body>, so scoping to it keeps the match inside the list that
            # was just opened rather than anywhere on the ticket. Class names
            # are Vuetify 2's; this app's exact version is not verified, so
            # the unscoped exact-text match below stays as the backstop (it is
            # what the reference bot used: //*[text()='Buy']).
            ("vuetify", f'.v-menu__content .v-list-item >> text="{option}"'),
            ("vuetify", f'.v-list-item >> text="{option}"'),
            ("css", f'[role="listbox"] >> text="{option}"'),
            ("exact_text", option),
        ], f"{what} option {option!r}", timeout=4000).click()

    # --- calibration --------------------------------------------------------

    def calibrate(self) -> dict:
        """Dump every interactive control on the trade ticket.

        Run this once against your logged-in account and after any site change:
        it is the ground truth the selector lists above are meant to match, and
        reading it beats guessing at a DOM you cannot see.

        Three things are dumped beyond the controls themselves, because they
        are what the tiers are built on: the page's ``data-cy`` inventory (tier
        one), and the *ordered* lists of Vuetify selects and buttons — the
        positional last-resort strategies are indexes into exactly those two
        lists, so this is where you read off a corrected index.
        """
        self._goto(TRADE_URL)
        js = """() => {
            const out = [];
            document.querySelectorAll('input,select,button,textarea,[role="button"],[role="combobox"]')
              .forEach(el => {
                const r = el.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) return;
                out.push({
                  tag: el.tagName.toLowerCase(),
                  type: el.getAttribute('type') || '',
                  name: el.getAttribute('name') || '',
                  id: el.id || '',
                  cls: (el.getAttribute('class') || '').slice(0, 90),
                  placeholder: el.getAttribute('placeholder') || '',
                  aria: el.getAttribute('aria-label') || '',
                  role: el.getAttribute('role') || '',
                  text: (el.innerText || el.value || '').trim().slice(0, 60),
                  cy: el.getAttribute('data-cy') || '',
                  testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || '',
                });
              });
            return out;
        }"""
        hooks_js = """() => Array.from(document.querySelectorAll('[data-cy]')).map(el => ({
            cy: el.getAttribute('data-cy'),
            tag: el.tagName.toLowerCase(),
            text: (el.innerText || el.value || '').trim().slice(0, 60),
        }))"""
        # 1-based to match the "nth" strategy and the XPath indexes these
        # positions were originally read from.
        ordered_js = """(sel) => Array.from(document.querySelectorAll(sel)).map((el, i) => ({
            index: i + 1, text: (el.innerText || '').trim().slice(0, 40),
        }))"""

        def dump(fn, *args) -> list:
            with suppress(Exception):
                return self.page.evaluate(fn, *args)
            return []

        controls = dump(js)
        out = {
            "url": self.page.url, "title": self.page.title(),
            "logged_in": "sign in" not in (self.page.content() or "").lower(),
            "controls": controls,
            "data_cy": dump(hooks_js),
            "v_selects": dump(ordered_js, ".v-select__selections"),
            "v_buttons": dump(ordered_js, ".v-btn__content"),
        }
        path = artifact_dir() / "calibration.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        self.snapshot("calibrate")
        logger.info("calibration written to %s (%d controls)", path, len(controls))
        # Logged rather than only written, because these three lists are what
        # a human actually needs to repair a selector, and the CLI's control
        # table does not show them.
        logger.info("data-cy hooks: %s",
                    ", ".join(sorted({h["cy"] for h in out["data_cy"] if h.get("cy")}))
                    or "(none — the Cypress hooks are gone, fall back a tier)")
        for name in ("v_selects", "v_buttons"):
            logger.info("%s: %s", name,
                        " | ".join(f'[{e["index"]}] {e["text"]}' for e in out[name]))
        return out

    def discover_api(self, seconds: int = 25) -> dict:
        """Record the XHR/fetch calls the simulator's own front end makes.

        The simulator is a single-page app talking to a JSON backend behind an
        OIDC bearer token. Driving the DOM works, but it is the slow and
        fragile path; the calls captured here are the fast and stable one, and
        they can only be observed from inside an authenticated session. Run
        this once while signed in and the result is a map of the real API.

        Request bodies are captured for reads only. Nothing here places an
        order, and the bearer token is redacted before it is written to disk.
        """
        seen: list[dict] = []

        def on_response(resp):
            try:
                req = resp.request
                if req.resource_type not in ("xhr", "fetch"):
                    return
                url = resp.url
                if any(s in url for s in ("google", "doubleclick", "analytics",
                                          "segment", "sentry", "chartbeat",
                                          "permutive", "adsystem", "facebook")):
                    return
                headers = dict(req.headers)
                had_auth = "authorization" in {k.lower() for k in headers}
                seen.append({
                    "method": req.method, "url": url, "status": resp.status,
                    "bearer": had_auth,
                    "content_type": resp.headers.get("content-type", ""),
                })
            except Exception:
                pass

        self.page.on("response", on_response)
        try:
            for url in (PORTFOLIO_URL, TRADE_URL, OPEN_ORDERS_URL):
                with suppress(Exception):
                    self._goto(url)
                    self.page.wait_for_timeout(int(seconds * 1000 / 3))
        finally:
            with suppress(Exception):
                self.page.remove_listener("response", on_response)

        # Collapse per-symbol and per-id variants so the shape of the API is
        # readable rather than buried under a hundred near-identical rows.
        uniq: dict[str, dict] = {}
        for e in seen:
            key = f"{e['method']} {re.sub(r'[0-9A-Fa-f-]{8,}|=[^&]+', '*', e['url'])}"
            uniq.setdefault(key, e)
        out = {"captured": len(seen), "endpoints": sorted(uniq.values(),
                                                          key=lambda e: e["url"])}
        path = artifact_dir() / "api_discovery.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        logger.info("captured %d API calls (%d unique) → %s",
                    len(seen), len(uniq), path)
        return out

    # --- account ------------------------------------------------------------

    def account(self) -> Account:
        """Scrape account value, cash, buying power, and open positions."""
        self._goto(PORTFOLIO_URL)
        acct = Account(fetched_at=datetime.now().isoformat())

        def near(label: str) -> float:
            """Value rendered adjacent to a label, in DOM order.

            The simulator lays out summary tiles as label/value sibling pairs
            with no stable class, so anchoring on the visible label text and
            walking to the nearest currency string is the durable read. Kept
            behind the ``data-cy`` reads below rather than replaced by them:
            a Cypress hook can be renamed in a single commit, the visible word
            "Cash" cannot.
            """
            js = """(label) => {
                const els = Array.from(document.querySelectorAll('*'));
                for (const el of els) {
                    if (el.children.length) continue;
                    const t = (el.innerText || '').trim();
                    if (t.toLowerCase().replace(/\\s+/g,' ') !== label.toLowerCase()) continue;
                    let n = el, hops = 0;
                    while (n && hops < 5) {
                        const txt = (n.parentElement?.innerText || '');
                        const m = txt.match(/\\$[\\d,]+\\.?\\d*/);
                        if (m) return m[0];
                        n = n.parentElement; hops++;
                    }
                }
                return null;
            }"""
            with suppress(Exception):
                return _money(self.page.evaluate(js, label))
            return 0.0

        # data-cy first: 'account-value' and 'cash' are verified hooks on the
        # portfolio page. 'buying-power' is a guess by analogy — it costs one
        # miss when absent and the label walk below covers that case.
        acct.account_value = (self._read_money('div[data-cy="account-value"]')
                              or near("Account Value") or near("Portfolio Value")
                              or near("Total Value"))
        acct.cash = (self._read_money('div[data-cy="cash"]')
                     or near("Cash") or near("Cash Balance") or near("Available Cash"))
        acct.buying_power = (self._read_money('div[data-cy="buying-power"]')
                             or near("Buying Power") or acct.cash)
        acct.holdings = self._scrape_holdings()
        return acct

    def _scrape_holdings(self) -> list[Holding]:
        """Open positions, by Cypress hook where present, else by header text."""
        rows = self._holdings_by_cy()
        return rows if rows else self._holdings_by_header()

    @staticmethod
    def _col(names: list[str], *wanted: str) -> int | None:
        """Index of the first entry in ``names`` containing any wanted word."""
        for w in wanted:
            for i, n in enumerate(names):
                if w in n:
                    return i
        return None

    def _holdings_by_cy(self) -> list[Holding]:
        """Read holdings from per-cell ``data-cy`` attributes, if there are any.

        The hooks on the *ticket* are verified; the ones on the portfolio table
        are not — this is a bet that the same test suite covers both pages, and
        it is written to find nothing rather than to find something wrong. A
        row only counts when one of its hooks names a symbol, so a table with
        unrelated hooks yields [] and the header-text scrape below runs.
        """
        js = """() => {
            const out = [];
            document.querySelectorAll('tr,[role="row"]').forEach(row => {
                // A header cell carrying a data-cy naming the symbol column
                // would otherwise be scraped as a position called "SYMBOL".
                if (row.closest('thead')) return;
                if (row.querySelector('th') && !row.querySelector('td')) return;
                const rec = {};
                row.querySelectorAll('[data-cy]').forEach(cell => {
                    const k = (cell.getAttribute('data-cy') || '').toLowerCase();
                    if (k && !(k in rec)) rec[k] = (cell.innerText || '').trim();
                });
                const keys = Object.keys(rec);
                if (keys.some(k => k.includes('symbol') || k.includes('ticker'))) {
                    out.push(rec);
                }
            });
            return out;
        }"""
        try:
            rows = self.page.evaluate(js)
        except Exception:
            rows = None
        if not rows:
            return []

        out: list[Holding] = []
        for rec in rows:
            keys = list(rec.keys())

            def val(*wanted: str) -> str:
                i = self._col(keys, *wanted)
                return rec[keys[i]] if i is not None else ""

            holding = self._holding(
                raw_symbol=val("symbol", "ticker"), raw_qty=val("quantity", "shares", "qty"),
                raw_cost=val("purchase-price", "purchase", "avg", "cost"),
                raw_last=val("current-price", "current", "last", "price"),
                raw_value=val("market-value", "value"),
                raw_pl=val("gain", "loss", "p&l", "unrealized"),
            )
            if holding:
                out.append(holding)
        return out

    def _holdings_by_header(self) -> list[Holding]:
        """Read the holdings table, mapping columns by their header text."""
        js = """() => {
            const tables = Array.from(document.querySelectorAll('table'));
            for (const tb of tables) {
                const heads = Array.from(tb.querySelectorAll('th'))
                    .map(th => (th.innerText || '').trim().toLowerCase());
                if (!heads.some(h => h.includes('symbol'))) continue;
                const rows = Array.from(tb.querySelectorAll('tbody tr')).map(tr =>
                    Array.from(tr.querySelectorAll('td')).map(td => (td.innerText || '').trim()));
                return {heads, rows};
            }
            return null;
        }"""
        try:
            data = self.page.evaluate(js)
        except Exception:
            data = None
        if not data or not data.get("rows"):
            return []

        heads = data["heads"]
        i_sym = self._col(heads, "symbol", "ticker")
        i_qty = self._col(heads, "quantity", "shares", "qty")
        i_cost = self._col(heads, "purchase price", "avg", "cost")
        i_last = self._col(heads, "current price", "last", "price")
        i_val = self._col(heads, "market value", "value")
        i_pl = self._col(heads, "gain", "loss", "p&l", "unrealized")

        out: list[Holding] = []
        for row in data["rows"]:
            def cell(i) -> str:
                return row[i] if i is not None and i < len(row) else ""
            holding = self._holding(
                raw_symbol=cell(i_sym), raw_qty=cell(i_qty), raw_cost=cell(i_cost),
                raw_last=cell(i_last), raw_value=cell(i_val), raw_pl=cell(i_pl),
            )
            if holding:
                out.append(holding)
        return out

    @staticmethod
    def _holding(raw_symbol: str, raw_qty: str, raw_cost: str, raw_last: str,
                 raw_value: str, raw_pl: str) -> Holding | None:
        """Build one Holding from rendered cell text, or None if unusable."""
        # The symbol cell often renders the company name on a second line and
        # a chevron or badge alongside the ticker; take the first line and keep
        # only ticker characters.
        sym = re.sub(r"[^A-Z.\-]", "", (raw_symbol or "").upper().split("\n")[0])
        if not sym:
            return None
        qty = _money(raw_qty)
        return Holding(
            symbol=sym, quantity=abs(qty), avg_cost=_money(raw_cost),
            last=_money(raw_last), market_value=_money(raw_value),
            unrealized=_money(raw_pl),
            # A short position renders as a negative quantity; the sign is the
            # only place the side is stated.
            side="short" if qty < 0 else "long",
        )

    # --- quotes -------------------------------------------------------------

    def quote(self, symbol: str) -> float:
        """Live simulator quote for ``symbol``, via the ticket's lookup.

        Worth reading from the site rather than yfinance: Investopedia fills
        against *its* quote, and reconciling a signal to a price the venue does
        not agree with is how paper books drift from their own P&L.
        """
        self._goto(TRADE_URL)
        self._enter_symbol(symbol)
        js = """() => {
            const m = document.body.innerText.match(/\\$\\s?([\\d,]+\\.\\d{2})/);
            return m ? m[1] : null;
        }"""
        self.page.wait_for_timeout(1500)
        with suppress(Exception):
            return _money(self.page.evaluate(js))
        return 0.0

    # --- order entry --------------------------------------------------------

    def _enter_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        box = self._first_visible([
            # Verified exact placeholder on the live ticket. It leads the list
            # ahead of the accessible-name guesses below it because it is a
            # fact and they are not: get_by_label('Symbol') matching some other
            # visible input would otherwise outrank a known-good selector.
            ("exact_placeholder", "Look up Symbol/Company Name"),
            ("role", ("combobox", re.compile(r"symbol|company", re.I))),
            ("label", "Symbol"),
            ("placeholder", "Symbol"),
            ("placeholder", "Look up"),
            ("css", "input[name='symbol']"),
            ("css", "input[id*='symbol' i]"),
            ("css", "input[type='search']"),
        ], "symbol input")
        box.click()
        box.fill("")
        box.type(symbol, delay=60)
        self.page.wait_for_timeout(1200)
        # Take the autocomplete suggestion when one appears: typing alone often
        # leaves the ticket unbound to an instrument, and submitting then
        # silently does nothing. The suggestion carries a verified hook,
        # data-cy="symbol-description"; the :has-text filter on the first entry
        # is what keeps a stale or partial list from binding the ticket to a
        # different company than the one that was asked for.
        for strategy in (
            ("cy", f'span[data-cy="symbol-description"]:has-text("{symbol}")'),
            ("cy", 'span[data-cy="symbol-description"]'),
            ("role", ("option", re.compile(rf"\b{re.escape(symbol)}\b", re.I))),
            ("css", f"li:has-text('{symbol}')"),
            ("css", f"[role='option']:has-text('{symbol}')"),
            ("css", f"[class*='result' i]:has-text('{symbol}')"),
        ):
            with suppress(Exception):
                opt = self._locate(*strategy)
                if opt.is_visible(timeout=2500):
                    opt.click()
                    self.page.wait_for_timeout(800)
                    self._dismiss_overlay()
                    return
        with suppress(Exception):
            box.press("Enter")
        self.page.wait_for_timeout(800)
        self._dismiss_overlay()

    # Trigger for the action ("Buy" / "Sell" / "Sell Short" / "Buy to Cover")
    # dropdown. No data-cy hook has been observed on it, so the accessible
    # name leads and the Vuetify class follows.
    _ACTION_TRIGGER = [
        ("role", ("combobox", re.compile(r"action|transaction|order side", re.I))),
        ("label", "Action"),
        ("label", "Transaction"),
        # Vuetify class tier, and positional in effect: the action select is
        # the FIRST .v-select__selections on the ticket (order type is the
        # second). A dropdown added above it would silently capture this.
        ("vuetify", ".v-select__selections"),
    ]

    _ORDER_TYPE_TRIGGER = [
        ("role", ("combobox", re.compile(r"order type|type of order", re.I))),
        ("label", "Order Type"),
        # Last resort, POSITIONAL: the order-type select is the second
        # .v-select__selections on the ticket. This is a place, not a thing.
        ("nth", (".v-select__selections", 2)),
    ]

    def _select_action(self, action: str) -> None:
        """Set Buy / Sell / Sell Short / Buy to Cover on the ticket."""
        try:
            self._choose_dropdown(action, self._ACTION_TRIGGER, ACTIONS, "action")
            return
        except LookupError:
            # Kept only for the day Investopedia ships a real <select> or an
            # accessible fallback: on the Vuetify ticket this has never matched
            # anything, and it used to run *first*, which is why a failed
            # selection went unnoticed for so long.
            if self._try_native_select(action, "action"):
                return
            raise

    _QUANTITY_INPUT = [
        ("cy", 'input[data-cy="quantity-input"]'),
        ("label", "Quantity"),
        ("placeholder", "Quantity"),
        ("css", "input[name='quantity']"),
        ("css", "input[id*='quantity' i]"),
        ("css", "input[type='number']"),
    ]

    def _set_quantity(self, qty: float) -> None:
        box = self._first_visible(self._QUANTITY_INPUT, "quantity input")
        box.click()
        # Clear first: the Max button and a re-used ticket both leave a value
        # in this field, and typing would append to it.
        box.fill("")
        # Whole shares only: the simulator rejects fractional quantities, and a
        # rejection surfaces as a form that simply refuses to advance.
        box.type(str(int(qty)), delay=40)

    def _set_order_type(self, order_type: str, limit_price: float | None) -> None:
        # Market is the ticket's default, so leaving the dropdown alone is both
        # correct and one less click to mis-land.
        if order_type == MARKET:
            return
        try:
            self._choose_dropdown(order_type, self._ORDER_TYPE_TRIGGER,
                                  ORDER_TYPES, "order type")
        except LookupError:
            if not self._try_native_select(order_type, "order type"):
                raise
        self.page.wait_for_timeout(500)
        if limit_price:
            # data-cy="limit-input" is verified for a Limit order. A Stop order
            # very likely has its own field; its hook has NOT been observed, so
            # the guess below is listed first and the limit strategies remain
            # underneath it rather than being replaced.
            price_field = [("cy", 'input[data-cy="limit-input"]')]
            if order_type == STOP:
                price_field = [("cy", 'input[data-cy="stop-input"]')] + price_field
            box = self._first_visible(price_field + [
                ("label", "Limit"), ("placeholder", "Limit"),
                ("css", "input[name*='limit' i]"), ("css", "input[name*='price' i]"),
            ], f"{order_type.lower()} price input")
            box.click()
            box.fill("")
            box.type(f"{limit_price:.2f}", delay=40)

    def _try_native_select(self, value: str, what: str) -> bool:
        """Last-ditch: drive a real ``<select>``, if the site ever ships one.

        Retained as a fallback and nothing more. ``select_option`` throws on
        anything that is not a native ``<select>``, so against the Vuetify
        ticket this cannot succeed — which is exactly why having it run first
        was a bug rather than a redundancy.
        """
        key = what.replace(" ", "")
        for sel in (f"select[name='{key}' i]", f"select[id*='{key}' i]",
                    f"select[aria-label*='{what}' i]"):
            with suppress(Exception):
                el = self.page.locator(sel).first
                if el.is_visible(timeout=1000):
                    el.select_option(label=value)
                    return True
        return False

    def max_quantity(self, symbol: str, action: str = BUY) -> float:
        """The venue's own view of the largest fillable size for ``symbol``.

        The ticket has a Max button that fills the quantity field with what the
        simulator thinks the account can afford — margin, settled cash and its
        own price all folded in. Reading it back is the only way to see that
        number; the portfolio page's buying power is a different figure that
        does not account for the instrument.

        Returns 0.0 when it cannot be read. That is "no opinion from the
        venue", not "you can afford nothing" — a caller that treats 0.0 as a
        hard cap will stop trading the moment this button is renamed, so treat
        a zero as missing data.
        """
        try:
            self._goto(TRADE_URL)
            self._enter_symbol(symbol)
            if action != BUY:
                self._select_action(action)
            self._first_visible([
                ("role", ("button", re.compile(r"^\s*max\s*$", re.I))),
                ("vuetify", 'span.v-btn__content:text-is("Max")'),
                # Last resort, POSITIONAL: the reference bot found Max as the
                # 7th .v-btn__content on the page. Any button added above it
                # points this at something else entirely, so it is only here
                # to keep a class rename from stopping the read.
                ("nth", ("span.v-btn__content", 7)),
            ], "Max button").click()
            # The field is repopulated by an async affordability call, not on
            # the click, so reading immediately returns the old value.
            self.page.wait_for_timeout(2000)
            box = self._first_visible(self._QUANTITY_INPUT, "quantity input")
            return _money(box.input_value(timeout=4000))
        except Exception as exc:
            logger.warning("could not read max quantity for %s: %s", symbol, exc)
            return 0.0

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str = MARKET,
        limit_price: float | None = None,
        dry_run: bool = False,
    ) -> OrderResult:
        """Fill and submit the trade ticket.

        ``dry_run`` fills every field and screenshots the completed ticket
        without pressing the final confirm — the only safe way to verify the
        selectors against a live account.

        ``quantity`` is submitted as given and never silently trimmed to
        :meth:`max_quantity`. The size here has already been through the
        secretary's risk gate and is what the ledger records; shrinking it in
        the browser layer would leave the local book claiming a position size
        the venue never filled.
        """
        symbol = symbol.upper()
        res = OrderResult(ok=False, symbol=symbol, action=action, quantity=quantity,
                          order_type=order_type, limit_price=limit_price,
                          submitted_at=datetime.now().isoformat())
        if action not in ACTIONS:
            res.message = f"unknown action {action!r}; expected one of {ACTIONS}"
            return res
        if int(quantity) <= 0:
            res.message = f"quantity rounds to zero ({quantity})"
            return res

        try:
            self._goto(TRADE_URL)
            self._enter_symbol(symbol)
            self._select_action(action)
            self._set_quantity(quantity)
            self._set_order_type(order_type, limit_price)

            if dry_run:
                res.artifact = self.snapshot(f"dryrun_{symbol}")
                res.ok = True
                res.message = "DRY RUN — ticket filled, not submitted"
                return res

            # Investopedia uses a two-step ticket: preview, then confirm.
            preview = self._first_visible([
                ("role", ("button", re.compile(r"preview|review|continue", re.I))),
                ("vuetify", "span.v-btn__content:has-text('Preview')"),
                ("css", "button[type='submit']"),
                # Last resort, POSITIONAL: 9th .v-btn__content on the filled
                # ticket, as the reference bot had it. Run `calibrate` and read
                # v_buttons before trusting this number again.
                ("nth", ("span.v-btn__content", 9)),
            ], "preview button")
            preview.click()
            self.page.wait_for_timeout(2000)

            confirm = self._first_visible([
                ("role", ("button", re.compile(r"^(confirm|submit|place order)", re.I))),
                ("vuetify", "span.v-btn__content:has-text('Submit')"),
                ("css", "button[type='submit']"),
                # Last resort, POSITIONAL: 11th .v-btn__content on the preview
                # step. A mis-click here does not place a wrong order — it
                # places none, and the confirmation check below reports that.
                ("nth", ("span.v-btn__content", 11)),
            ], "confirm button", timeout=8000)
            confirm.click()
            self.page.wait_for_timeout(2500)

            body = (self.page.inner_text("body") or "").lower()
            if any(k in body for k in ("order received", "successfully", "order placed",
                                       "your order has been", "order confirmation")):
                res.ok = True
                res.message = "submitted"
            elif any(k in body for k in ("insufficient", "error", "not enough", "rejected",
                                         "cannot be", "invalid")):
                res.message = "venue rejected the order"
                res.artifact = self.snapshot(f"reject_{symbol}")
            else:
                # Ambiguous: neither confirmation nor error text. Report it as
                # unknown rather than as success — a false "filled" corrupts
                # the local book against the real one.
                res.message = "submitted but no confirmation text found — verify manually"
                res.artifact = self.snapshot(f"unknown_{symbol}")
        except Exception as exc:
            res.message = f"{type(exc).__name__}: {exc}"
            res.artifact = self.snapshot(f"error_{symbol}")
            logger.exception("order failed for %s", symbol)
        return res

    def open_orders(self) -> list[dict]:
        """Pending orders, as raw row dicts (shape varies with the site)."""
        self._goto(OPEN_ORDERS_URL)
        js = """() => {
            const tb = document.querySelector('table');
            if (!tb) return [];
            const heads = Array.from(tb.querySelectorAll('th')).map(t => (t.innerText||'').trim());
            return Array.from(tb.querySelectorAll('tbody tr')).map(tr => {
                const tds = Array.from(tr.querySelectorAll('td')).map(t => (t.innerText||'').trim());
                return Object.fromEntries(heads.map((h,i) => [h, tds[i] ?? '']));
            });
        }"""
        with suppress(Exception):
            return self.page.evaluate(js)
        return []
