"""Keeping the desk alive when nobody is watching it.

``python -m tradingagents.live.cli run`` is an ordinary foreground process. It
dies with the terminal, at logout, at reboot, and — the one that actually bites
— when the Mac goes to sleep. On this machine that is not hypothetical:
``pmset -g custom`` reports ``sleep 1`` on both AC and battery, so the display
dims after ten minutes and the system follows a minute later, in the middle of
the session, with an unmanaged position open.

Three tiers, ranked honestly:

1. **nohup** — survives the terminal and nothing else. Fine for one afternoon
   of watching it work; not a deployment.
2. **launchd** — the right answer on a Mac. A LaunchAgent restarts the desk
   when it crashes, brings it back at login and after a reboot, and holds a
   power assertion for as long as it runs.
3. **A Linux VPS** — the only genuinely machine-independent option, and the
   awkward one, because the logged-in Investopedia session lives in a Chromium
   profile directory that has to travel with it. See :data:`CLOUD_NOTES`.

Two failure modes dominate every launchd install, and both are silent:

**A launchd job does not read your shell profile.** No ``.zshrc``, no
``.zprofile``, no ``export``. A job that runs perfectly from a terminal and
does nothing under launchd is nearly always missing an API key or a PATH entry
the shell was quietly supplying. So every variable the desk needs is written
into the plist's ``EnvironmentVariables`` at install time, captured from the
environment doing the installing.

**The repo is not the installed package.** Verified on this machine: the copy
of ``tradingagents`` in site-packages has no ``live`` subpackage at all, so
``python -m tradingagents.live.cli`` only resolves from the checkout. The plist
therefore sets both ``WorkingDirectory`` and ``PYTHONPATH`` to the repo — one
of them would do, and having both means a stray ``cd`` cannot produce a
``ModuleNotFoundError`` at 09:30.

Nothing here raises out of a loop. The launchctl wrappers report a failed step
instead of throwing, and every check degrades to a reported-unavailable row.
"""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

LABEL = "com.tradingagents.livedesk"
# Spelled out rather than read from __spec__, which is None when this file
# is run as a plain script rather than with -m.
_MODULE = "tradingagents.live.deploy"

# Absolute paths: the job's PATH is whatever the plist says it is, and these
# two are the parts that must work before the plist's PATH matters.
LAUNCHCTL = "/bin/launchctl"
CAFFEINATE = "/usr/bin/caffeinate"


# --- locations --------------------------------------------------------------

def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


def log_dir() -> Path:
    p = _home() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path:
    """The checkout this file lives in — deploy.py is at <repo>/tradingagents/live/."""
    return Path(__file__).resolve().parents[2]


def launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(label: str = LABEL) -> Path:
    return launch_agents_dir() / f"{label}.plist"


def out_log(label: str = LABEL) -> Path:
    return log_dir() / f"{label}.out.log"


def err_log(label: str = LABEL) -> Path:
    return log_dir() / f"{label}.err.log"


def domain() -> str:
    """launchctl's modern service target for the logged-in user's agents."""
    return f"gui/{os.getuid()}"


# --- environment capture ----------------------------------------------------

# Swept up by pattern so a provider added to the registry later needs no edit
# here: every *_API_KEY in the installing environment travels into the job,
# which is the shape every entry in llm_clients.api_key_env already has.
_ENV_SUFFIXES = ("_API_KEY",)
_ENV_PREFIXES = ("TRADINGAGENTS_",)
_ENV_EXACT = ("PLAYWRIGHT_BROWSERS_PATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "no_proxy")

# Deliberately narrow. An earlier version swept every AWS_* variable and
# promptly wrote an unrelated set of cloud credentials, which happened to be
# exported in the installing shell, into the plist in cleartext. A generated
# file should carry what this job needs and nothing it merely found lying
# around, so the AWS chain travels only when the configured provider is the
# one that authenticates with it.
_BEDROCK_ENV = ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
_VERTEX_ENV = ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT")


def _provider() -> str:
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        return str(DEFAULT_CONFIG.get("llm_provider", "")).lower()
    except Exception:
        return ""


def _provider_env(provider: str) -> tuple[str, ...]:
    """Extra variables that only one provider needs to authenticate."""
    if provider == "bedrock":
        return _BEDROCK_ENV
    if provider.startswith("google"):
        return _VERTEX_ENV
    return ()

# launchd hands a job a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin). The
# interpreter's own bin directory goes first so a subprocess that shells out to
# `python` or `playwright` finds the same interpreter that started the job.
_PATH_TAIL = ("/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin",
              "/usr/sbin", "/sbin")


def captured_env(cfg: DeployConfig | None = None,
                 source: dict[str, str] | None = None) -> dict[str, str]:
    """Everything the job needs, resolved now, because launchd resolves nothing.

    ``source`` defaults to ``os.environ``. Note that importing this module has
    already run ``tradingagents/__init__.py``, which loads the repo's ``.env``
    through python-dotenv — so keys kept in ``.env`` are visible here even
    though they were never exported by a shell.
    """
    cfg = cfg or DeployConfig()
    src = dict(os.environ if source is None else source)
    env: dict[str, str] = {}

    if cfg.include_secrets:
        extra = _provider_env(cfg.provider or _provider())
        for k, v in src.items():
            if not v:
                continue
            if (k.endswith(_ENV_SUFFIXES) or k.startswith(_ENV_PREFIXES)
                    or k in _ENV_EXACT or k in extra):
                env[k] = v

    env["HOME"] = src.get("HOME", str(Path.home()))
    env["PATH"] = os.pathsep.join([str(Path(cfg.python).parent), *_PATH_TAIL])
    env["LANG"] = src.get("LANG", "en_US.UTF-8")
    # Without this, stdout to a file is block-buffered and the log looks dead
    # for hours — the single most common reason a working job is believed dead.
    env["PYTHONUNBUFFERED"] = "1"
    # See the module docstring: site-packages has no `live` subpackage.
    env["PYTHONPATH"] = str(cfg.repo)
    env.update(cfg.env)
    return env


# --- configuration ----------------------------------------------------------

@dataclass
class DeployConfig:
    """Everything the plist is generated from. All of it is overridable so the
    generator can be exercised in a test without touching the real LaunchAgents
    directory or the real interpreter."""

    label: str = LABEL
    # sys.executable, never "python": launchd resolves argv[0] against the
    # plist's PATH, and picking up a different interpreter than the one holding
    # playwright is a failure that only shows up at the first browser launch.
    python: str = field(default_factory=lambda: sys.executable)
    repo: Path = field(default_factory=repo_root)
    run_args: list[str] = field(default_factory=list)   # extra flags for `cli run`
    caffeinate: bool = True
    run_at_load: bool = True
    keep_alive: bool = True
    restart_on_crash_only: bool = False
    # A desk that exits immediately (expired Investopedia session, say) would
    # otherwise be respawned by launchd every few seconds forever. Ten minutes
    # of throttle turns a spin into something a log tail can read.
    throttle_interval: int = 600
    # launchd.plist(5): with no ProcessType the system "will apply light
    # resource limits to the job, throttling its CPU usage and I/O bandwidth".
    # The cycle drives a real browser under a 30s page timeout, so it is
    # classified Interactive to opt out of that throttling.
    process_type: str = "Interactive"
    include_secrets: bool = True
    # Which provider's auth variables to carry. Empty means "read it from
    # DEFAULT_CONFIG at install time"; set it to pin the capture in a test.
    provider: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.repo = Path(self.repo)


def program_arguments(cfg: DeployConfig | None = None) -> list[str]:
    """The exact argv launchd will exec.

    ``caffeinate -is`` holds two power assertions for exactly as long as the
    child runs: ``-i`` blocks idle sleep, ``-s`` blocks system sleep. Two
    honest limits. caffeinate(8) documents ``-s`` as "valid only when system is
    running on AC power", so on battery only the idle assertion applies; and a
    closed lid sleeps the machine regardless — no caffeinate assertion covers
    lid-close. (The lid behaviour is not something this code verified on your
    machine; treat it as the reason to leave the lid open or stay on power.)

    caffeinate execs the utility and exits with its status — verified:
    ``caffeinate -is /bin/sh -c 'exit 7'`` returns 7 — so KeepAlive still sees
    a crash for what it is rather than seeing caffeinate exit cleanly.
    """
    cfg = cfg or DeployConfig()
    cmd = [cfg.python, "-m", "tradingagents.live.cli", "run", *cfg.run_args]
    return [CAFFEINATE, "-is", *cmd] if cfg.caffeinate else cmd


def build_plist(cfg: DeployConfig | None = None) -> dict:
    cfg = cfg or DeployConfig()
    # A dict restarts only on a non-zero exit. The tradeoff is real, and worth
    # stating: the desk exits 0 when it finds no Investopedia session, so under
    # restart_on_crash_only an expired login stops the agent silently instead
    # of spinning against the throttle. Plain True is the default because a
    # desk that is down and quiet is worse than one that is down and noisy.
    keep: bool | dict = ({"SuccessfulExit": False} if cfg.restart_on_crash_only
                         else cfg.keep_alive)

    return {
        "Label": cfg.label,
        "ProgramArguments": program_arguments(cfg),
        "RunAtLoad": cfg.run_at_load,      # comes back after a reboot, at login
        "KeepAlive": keep,                 # comes back after a crash
        "ThrottleInterval": cfg.throttle_interval,
        "ProcessType": cfg.process_type,
        "WorkingDirectory": str(cfg.repo),
        "EnvironmentVariables": captured_env(cfg),
        "StandardOutPath": str(out_log(cfg.label)),
        "StandardErrorPath": str(err_log(cfg.label)),
    }


def render_plist(cfg: DeployConfig | None = None, redact: bool = False) -> bytes:
    """Serialise through plistlib rather than a text template.

    An API key containing ``&`` or ``<`` silently corrupts a hand-written XML
    template, and the resulting job fails to load with a parse error that names
    a line number rather than the key.

    ``redact=True`` blanks the secret values so the plist can be printed,
    pasted into an issue, or read over someone's shoulder. The file written to
    disk is never redacted.
    """
    d = build_plist(cfg)
    if redact:
        d["EnvironmentVariables"] = {
            k: (f"<redacted, {len(v)} chars>" if _is_secret(k) else v)
            for k, v in d["EnvironmentVariables"].items()}
    return plistlib.dumps(d, fmt=plistlib.FMT_XML)


def _is_secret(name: str) -> bool:
    return (name.endswith(("_API_KEY", "_SECRET_ACCESS_KEY", "_SESSION_TOKEN",
                           "_ACCESS_KEY_ID", "_TOKEN"))
            or "SECRET" in name or "PASSWORD" in name)


def write_plist(cfg: DeployConfig | None = None, path: str | Path | None = None,
                dry_run: bool = False) -> Path:
    cfg = cfg or DeployConfig()
    p = Path(path) if path else plist_path(cfg.label)
    data = render_plist(cfg)
    if dry_run:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    # The plist holds API keys in cleartext. 0600 is the least that is owed;
    # `include_secrets=False` is the alternative, and works only because
    # WorkingDirectory makes the repo's .env discoverable to python-dotenv.
    os.chmod(p, 0o600)
    return p


# --- launchctl --------------------------------------------------------------

@dataclass
class Step:
    """One shelled-out command and what came back. ``rc is None`` means the
    command could not be run at all (missing binary, timeout)."""
    cmd: list[str]
    rc: int | None = None
    out: str = ""
    skipped: bool = False
    # Failed, but with the failure that means "there was nothing to do". Kept
    # distinct from success so a report does not claim it stopped a job that
    # was never running, and distinct from an error so a clean first install
    # does not print what looks like a fault.
    benign: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def __str__(self) -> str:
        head = " ".join(self.cmd)
        if self.skipped:
            return f"[dry-run] {head}"
        if self.benign:
            return f"[none] {head}"
        return f"[{'ok' if self.ok else self.rc}] {head}" + (
            f"\n    {self.out.strip()}" if self.out.strip() else "")


def _run(cmd: list[str], dry_run: bool = False, timeout: int = 30) -> Step:
    if dry_run:
        return Step(cmd, skipped=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return Step(cmd, r.returncode, (r.stdout or "") + (r.stderr or ""))
    except (OSError, subprocess.SubprocessError) as exc:
        # Reported, not raised: install() reports a bad step and leaves the
        # plist on disk so the user can load it by hand.
        return Step(cmd, None, f"could not run: {exc}")


def _unsupported(step: Step) -> bool:
    """True when this macOS predates the bootstrap/bootout subcommands."""
    t = step.out.lower()
    return "unrecognized" in t or "unknown subcommand" in t or "usage:" in t


def bootstrap(path: Path, dry_run: bool = False) -> list[Step]:
    """Load the job. ``bootstrap`` is the modern spelling; ``load -w`` is the
    fallback for macOS old enough not to have it."""
    steps = [_run([LAUNCHCTL, "bootstrap", domain(), str(path)], dry_run)]
    if not dry_run and not steps[-1].ok and _unsupported(steps[-1]):
        steps.append(_run([LAUNCHCTL, "load", "-w", str(path)]))
    return steps


def bootout(label: str = LABEL, path: Path | None = None,
            dry_run: bool = False) -> list[Step]:
    """Unload the job. Failure is normal — it means it was not loaded."""
    steps = [_run([LAUNCHCTL, "bootout", f"{domain()}/{label}"], dry_run)]
    # launchd reports "No such process" (errno 3) when the label is not loaded.
    # On a first install that is the normal path, not a fault.
    if not steps[-1].ok and "no such process" in steps[-1].out.lower():
        steps[-1].benign = True
        return steps
    if not dry_run and not steps[-1].ok and _unsupported(steps[-1]):
        target = str(path or plist_path(label))
        steps.append(_run([LAUNCHCTL, "unload", "-w", target]))
    return steps


def install(cfg: DeployConfig | None = None, path: str | Path | None = None,
            dry_run: bool = False, force: bool = False,
            check_session: bool = True) -> dict:
    """Write the plist and load it. Returns a report; never raises.

    Preflight runs first and blocks the install on a hard failure, because the
    two conditions that matter — no Investopedia session, no LLM key — both
    produce a job that launchd will restart forever without ever trading.
    """
    cfg = cfg or DeployConfig()
    p = Path(path) if path else plist_path(cfg.label)
    report: dict = {"label": cfg.label, "plist": str(p), "dry_run": dry_run,
                    "checks": [], "steps": [], "ok": False,
                    "out_log": str(out_log(cfg.label)),
                    "err_log": str(err_log(cfg.label))}

    checks = preflight_check(cfg, check_session=check_session)
    report["checks"] = checks
    blocking = [c for c in checks if not c.ok]
    if blocking and not force:
        report["blocked"] = [c.name for c in blocking]
        return report

    log_dir()
    written = write_plist(cfg, p, dry_run=dry_run)
    report["written"] = not dry_run

    # Boot out first, unconditionally. bootstrap refuses a label that is
    # already bootstrapped, and an unload/reload is also the only way a plist
    # edit takes effect — launchd caches the loaded copy, not the file.
    stopped = bootout(cfg.label, written, dry_run=dry_run)
    started = bootstrap(written, dry_run=dry_run)
    report["steps"] = stopped + started
    report["ok"] = dry_run or any(s.ok for s in started)
    report["status"] = status(cfg.label, path=written, dry_run=dry_run)
    return report


def uninstall(label: str = LABEL, path: str | Path | None = None,
              dry_run: bool = False, remove_plist: bool = True) -> dict:
    p = Path(path) if path else plist_path(label)
    steps = bootout(label, p, dry_run=dry_run)
    removed = False
    if remove_plist and p.exists() and not dry_run:
        p.unlink()
        removed = True
    return {"label": label, "plist": str(p), "steps": steps,
            "plist_removed": removed, "dry_run": dry_run}


def restart(label: str = LABEL, dry_run: bool = False) -> list[Step]:
    """Kill and respawn without unloading. Also the way to reopen the log files
    after :func:`rotate_logs` — launchd holds those descriptors open."""
    return [_run([LAUNCHCTL, "kickstart", "-k", f"{domain()}/{label}"], dry_run)]


def _grab(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else None


def status(label: str = LABEL, path: str | Path | None = None,
           dry_run: bool = False) -> dict:
    """What launchd thinks of the job right now.

    The output of ``launchctl print`` is a human-readable dump, not an API, so
    the parse is best-effort: the raw text is kept in the result and the parsed
    fields are ``None`` when a key is not found rather than a guess.
    """
    p = Path(path) if path else plist_path(label)
    out: dict = {"label": label, "plist": str(p), "installed": p.exists(),
                 "loaded": False, "pid": None, "state": None,
                 "last_exit": None, "raw": ""}
    if dry_run:
        return out

    step = _run([LAUNCHCTL, "print", f"{domain()}/{label}"])
    if step.ok:
        out["loaded"] = True
        out["raw"] = step.out
        out["state"] = _grab(step.out, r"^\s*state\s*=\s*(\S+)")
        pid = _grab(step.out, r"^\s*pid\s*=\s*(\d+)")
        out["pid"] = int(pid) if pid else None
        code = _grab(step.out, r"last exit code\s*=\s*(-?\d+)")
        out["last_exit"] = int(code) if code else None
        return out

    # Older macOS, or `print` denied: `list` still reports PID and last status.
    step = _run([LAUNCHCTL, "list", label])
    if step.ok:
        out["loaded"] = True
        out["raw"] = step.out
        pid = _grab(step.out, r'"PID"\s*=\s*(\d+)')
        out["pid"] = int(pid) if pid else None
        code = _grab(step.out, r'"LastExitStatus"\s*=\s*(-?\d+)')
        out["last_exit"] = int(code) if code else None
    else:
        out["raw"] = step.out
    return out


def format_status(st: dict) -> str:
    lines = [
        f"label      {st['label']}",
        f"plist      {st['plist']} {'(present)' if st['installed'] else '(MISSING)'}",
        f"loaded     {st['loaded']}",
        f"pid        {st['pid'] if st['pid'] else '- (not running)'}",
    ]
    if st.get("state"):
        lines.append(f"state      {st['state']}")
    if st.get("last_exit") is not None:
        lines.append(f"last exit  {st['last_exit']}")
    lines.append(f"stdout     {out_log(st['label'])}")
    lines.append(f"stderr     {err_log(st['label'])}")
    return "\n".join(lines)


# --- logs -------------------------------------------------------------------

def tail_logs(lines: int = 40, label: str = LABEL, stream: str = "both",
              follow: bool = False) -> str:
    """Last ``lines`` of the job's output. ``follow=True`` blocks on tail -f.

    stderr is worth reading separately after a failed install: launchd's own
    spawn errors (bad interpreter path, a file it may not open) land there,
    and they are the ones that explain a job that never produced any stdout.
    """
    paths: list[Path] = []
    if stream in ("both", "out"):
        paths.append(out_log(label))
    if stream in ("both", "err"):
        paths.append(err_log(label))

    if follow:
        existing = [str(p) for p in paths if p.exists()]
        if not existing:
            return f"no log files yet: {', '.join(str(p) for p in paths)}"
        try:
            subprocess.run(["/usr/bin/tail", "-n", str(lines), "-F", *existing])
        except KeyboardInterrupt:
            pass
        except OSError as exc:
            return f"could not follow logs: {exc}"
        return ""

    chunks = []
    for p in paths:
        if not p.exists():
            chunks.append(f"--- {p} (does not exist yet) ---")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            chunks.append(f"--- {p} (unreadable: {exc}) ---")
            continue
        tail = "".join(text.splitlines(keepends=True)[-lines:])
        chunks.append(f"--- {p} ---\n{tail.rstrip()}")
    return "\n\n".join(chunks)


def rotate_logs(label: str = LABEL, max_bytes: int = 32 * 1024 * 1024) -> list[str]:
    """Truncate the job's logs in place once they pass ``max_bytes``.

    In place, not renamed: launchd opened these files and holds the
    descriptors, so renaming leaves the job writing into the renamed inode
    while the fresh file stays empty forever. Truncation keeps the descriptor
    valid. Follow with :func:`restart` if you want the offsets reset too —
    otherwise the job may keep writing at its old offset and the file reads
    back with a run of NULs until it catches up.
    """
    done = []
    for p in (out_log(label), err_log(label)):
        try:
            if p.exists() and p.stat().st_size > max_bytes:
                keep = p.read_text(encoding="utf-8", errors="replace")[-max_bytes // 8:]
                p.with_suffix(p.suffix + ".1").write_text(keep, encoding="utf-8")
                os.truncate(p, 0)
                done.append(str(p))
        except OSError as exc:
            done.append(f"{p}: could not rotate ({exc})")
    return done


# --- preflight --------------------------------------------------------------

class Check(NamedTuple):
    name: str
    ok: bool
    detail: str


def _chromium_path() -> Path | None:
    """Where playwright's chromium lives, without starting the driver.

    The cache layout is playwright's private business, so a miss here falls
    through to asking playwright itself — which is authoritative but starts a
    node process and prints noise on teardown.
    """
    root = Path(os.getenv("PLAYWRIGHT_BROWSERS_PATH")
                or Path.home() / "Library" / "Caches" / "ms-playwright")
    try:
        for d in sorted(root.glob("chromium-*"), reverse=True):
            for exe in d.glob("chrome-mac*/*.app/Contents/MacOS/*"):
                if exe.is_file():
                    return exe
            for exe in d.glob("chrome-linux/chrome"):
                if exe.is_file():
                    return exe
    except OSError:
        pass
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            exe = Path(pw.chromium.executable_path)
        return exe if exe.exists() else None
    except Exception:
        return None


def profile_lock_held() -> bool:
    """True when a Chromium is *currently* holding the persistent profile.

    ``SingletonLock`` is Chromium's own lock: a symlink at the top of the user
    data directory whose target is ``hostname-pid``. It matters here because
    the profile cannot be opened twice — with the LaunchAgent running,
    ``cli login`` and ``cli portfolio`` fail, and so would a session check made
    from this process.

    The existence of the file is not the question. A lock left behind by a
    browser that was killed rather than closed survives, and treating that as
    "held" would wedge the session check and the profile export permanently
    with no way back. So the target is resolved and the owner tested, which is
    also how Chromium itself decides a lock is stale.
    """
    import socket

    from .investopedia import profile_dir
    lock = profile_dir() / "SingletonLock"
    try:
        target = os.readlink(lock)
    except OSError:
        return False       # absent, or not a symlink: nothing holds it

    host, _, pid = target.rpartition("-")
    if host and host != socket.gethostname():
        return False       # came from another machine — an imported profile
    if not pid.isdigit():
        return True        # unrecognised shape: the conservative answer
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False       # stale
    except PermissionError:
        return True        # alive, owned by another user
    except OSError:
        return True
    return True


def _check_session(timeout_s: int = 45) -> Check:
    """Can the configured venue actually be traded right now?

    Venue-aware because the two adapters fail in completely different ways and
    a check that names the wrong one sends you to fix the wrong thing. Alpaca
    fails on credentials; Investopedia fails on an expired browser session.
    """
    from .broker import ALPACA, configured_venue
    venue = configured_venue()

    if venue == ALPACA:
        try:
            from .alpaca import AlpacaBroker, MissingCredentials
            try:
                broker = AlpacaBroker()
            except MissingCredentials:
                return Check("alpaca credentials", False,
                             "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — free "
                             "paper keys: https://app.alpaca.markets/signup")
            ok = broker.is_logged_in()
            if not ok:
                return Check("alpaca session", False,
                             "keys rejected or account blocked — check they are "
                             "PAPER keys (the id starts with PK)")
            acct = broker.account()
            return Check("alpaca session", True,
                         f"paper account reachable, "
                         f"${acct.account_value:,.0f} equity, "
                         f"{len(acct.holdings)} position(s)")
        except Exception as exc:
            return Check("alpaca session", False, f"could not check: {exc}")

    # Investopedia: a live browser holding the profile means the desk is up.
    if profile_lock_held():
        return Check("investopedia session", True,
                     "skipped — a browser already holds the profile "
                     "(the desk is probably running)")
    try:
        from .investopedia import InvestopediaBroker
        with InvestopediaBroker(headless=True, timeout=timeout_s * 1000) as b:
            ok = b.is_logged_in()
        return Check("investopedia session", ok,
                     "signed in" if ok else
                     "NOT signed in — run: python -m tradingagents.live.cli login")
    except Exception as exc:
        return Check("investopedia session", False, f"could not check: {exc}")


def preflight_check(cfg: DeployConfig | None = None,
                    check_session: bool = True) -> list[Check]:
    """Everything that has to be true before an unattended install is honest.

    Returns ``(name, ok, detail)`` rows. Every probe is wrapped: a check that
    cannot run reports itself as failed with the reason, and never raises into
    the caller.
    """
    cfg = cfg or DeployConfig()
    checks: list[Check] = []

    exe = Path(cfg.python)
    checks.append(Check(
        "interpreter", exe.exists() and sys.version_info >= (3, 10),
        f"{exe} ({'.'.join(str(v) for v in sys.version_info[:3])})"))

    live_cli = cfg.repo / "tradingagents" / "live" / "cli.py"
    checks.append(Check("repo", live_cli.exists(), f"{cfg.repo}"))

    # The site-packages trap, tested rather than assumed: start the configured
    # interpreter with exactly the environment and directory the plist gives
    # it, and see whether `tradingagents.live` resolves at all. This is the
    # check that catches a job which would die on its first line with
    # ModuleNotFoundError and be restarted forever.
    if not cfg.repo.is_dir():
        # launchd refuses to spawn a job whose WorkingDirectory is missing, so
        # running the probe from some other directory would report a pass for
        # a job that cannot start.
        checks.append(Check("module resolves", False,
                            f"WorkingDirectory {cfg.repo} does not exist"))
    else:
        try:
            probe = subprocess.run(
                [cfg.python, "-c",
                 "import tradingagents.live as m; print(m.__file__)"],
                capture_output=True, text=True, timeout=60,
                cwd=str(cfg.repo), env=captured_env(cfg))
            if probe.returncode == 0:
                found = (probe.stdout or "").strip().splitlines()
                detail = found[-1] if found else "imported"
            else:
                failed = (probe.stderr or "").strip().splitlines()
                detail = failed[-1] if failed else "import failed"
            checks.append(Check("module resolves", probe.returncode == 0, detail))
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(Check("module resolves", False, f"could not check: {exc}"))

    try:
        import playwright
        checks.append(Check("playwright", True,
                            getattr(playwright, "__version__", "installed")))
    except Exception as exc:
        checks.append(Check("playwright", False,
                            f"{exc} — pip install playwright"))

    chrome = _chromium_path()
    checks.append(Check("chromium", chrome is not None,
                        str(chrome) if chrome else
                        "not downloaded — python -m playwright install chromium"))

    # Asked of the environment the *job* will get, not the one running this
    # check. A key that is exported in this shell but not carried into the
    # plist is exactly the silent launchd failure this module exists to stop.
    try:
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        provider = cfg.provider or _provider()
        var = get_api_key_env(provider)
        if var is None:
            # Bedrock and local runtimes have no single key var; saying so is
            # more useful than reporting a pass that was never tested.
            checks.append(Check("llm key", True,
                                f"provider {provider!r} does not use a single key env"))
        else:
            present = bool(captured_env(cfg).get(var))
            checks.append(Check("llm key", present,
                                f"{var} {'in the job env' if present else 'MISSING'} "
                                f"(provider {provider})"))
    except Exception as exc:
        checks.append(Check("llm key", False, f"could not resolve provider: {exc}"))

    try:
        from .secretary import kill_switch_engaged, kill_switch_path
        engaged = kill_switch_engaged()
        checks.append(Check("kill switch", not engaged,
                            f"{kill_switch_path()} "
                            f"{'ENGAGED — nothing will trade' if engaged else 'clear'}"))
    except Exception as exc:
        checks.append(Check("kill switch", False, f"could not check: {exc}"))

    try:
        d = log_dir()
        probe = d / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        checks.append(Check("log dir", True, str(d)))
    except Exception as exc:
        checks.append(Check("log dir", False, f"{log_dir()} not writable: {exc}"))

    missing = [b for b in (LAUNCHCTL, CAFFEINATE) if not Path(b).exists()]
    checks.append(Check("system tools", not missing,
                        f"{LAUNCHCTL}, {CAFFEINATE}" if not missing
                        else f"missing: {', '.join(missing)}"))

    if check_session:
        checks.append(_check_session())

    checks.append(_check_power(cfg))
    return checks


def _check_power(cfg: DeployConfig) -> Check:
    """Report the sleep settings rather than judging them.

    This is informational, so it always passes: caffeinate is what handles a
    non-zero sleep timer, and the point of printing the value is to show the
    user what caffeinate is being asked to hold off.
    """
    step = _run(["/usr/bin/pmset", "-g"], timeout=10)
    sleep = _grab(step.out, r"^\s*sleep\s+(\d+)") if step.ok else None
    if sleep is None:
        return Check("power", True, "could not read pmset; assuming the Mac sleeps")
    if sleep == "0":
        return Check("power", True, "system sleep is disabled")
    guard = "caffeinate -is will hold it off" if cfg.caffeinate else \
            "caffeinate is OFF — the desk will stop when the Mac sleeps"
    return Check("power", True, f"sleep {sleep} min; {guard}")


def format_checks(checks: list[Check]) -> str:
    width = max((len(c.name) for c in checks), default=10)
    rows = [f"  {'ok' if c.ok else 'FAIL':<4}  {c.name:<{width}}  {c.detail}"
            for c in checks]
    bad = sum(1 for c in checks if not c.ok)
    tail = "all clear" if not bad else f"{bad} check(s) failed"
    return "\n".join(rows) + f"\n\n{tail}"


# --- tier 1: foreground / nohup ---------------------------------------------

def foreground_command(cfg: DeployConfig | None = None) -> str:
    """Tier 0: watch it work in front of you. Dies with the terminal."""
    cfg = cfg or DeployConfig()
    return f"cd {cfg.repo} && " + " ".join(program_arguments(cfg))


def nohup_command(cfg: DeployConfig | None = None) -> str:
    """Survives the terminal. Not a reboot, not a logout, not sleep-with-lid-shut."""
    cfg = cfg or DeployConfig()
    args = " ".join(program_arguments(cfg))
    return (f"cd {cfg.repo} && nohup {args} "
            f">> {out_log(cfg.label)} 2>&1 &")


NOHUP_SCRIPT = """#!/bin/zsh
# Tier 1: run the live desk detached from this terminal.
#
# This survives closing the terminal and nothing else. It does not come back
# after a reboot, a logout, or a crash, and `caffeinate -is` only holds off
# idle and (on AC power) system sleep — a closed lid still stops it.
# For anything longer than an afternoon, use the LaunchAgent:
#   python -m tradingagents.live.deploy install
set -eu

REPO="{repo}"
PYTHON="{python}"
LOG="{log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")"
nohup /usr/bin/caffeinate -is "$PYTHON" -m tradingagents.live.cli run "$@" \\
      >> "$LOG" 2>&1 &
echo "live desk pid $! -> $LOG"
echo "stop trading:  python -m tradingagents.live.cli stop"
echo "stop process:  kill $!"
"""


def write_nohup_script(path: str | Path | None = None,
                       cfg: DeployConfig | None = None) -> Path:
    cfg = cfg or DeployConfig()
    p = Path(path) if path else (_home() / "bin" / "livedesk.sh")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(NOHUP_SCRIPT.format(repo=cfg.repo, python=cfg.python,
                                     log=out_log(cfg.label)), encoding="utf-8")
    os.chmod(p, 0o755)
    return p


# --- tier 3: a Linux box ----------------------------------------------------

CLOUD_NOTES = """Tier 3 — running it on a VPS instead of this Mac

You do not need a VM. A LaunchAgent on a Mac that stays powered is the simpler
and cheaper answer, and it is the one this module installs. A VPS buys exactly
one thing: independence from this laptop's lid, battery and network.

What the box actually has to provide

  * A real Chromium. The Investopedia adapter drives a browser because the
    site has no API and blocks non-browser clients; there is no headless-HTTP
    fallback to degrade to.
      sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
      python3 -m playwright install --with-deps chromium
    `--with-deps` installs the system libraries Chromium needs and wants root.
  * Memory. Chromium is the floor here, not Python: budget about 1 GB for the
    browser and take 2 GB of RAM if you want headroom. (That figure is the
    usual guidance for headless Chromium, not something measured on your
    machine.) A 512 MB instance will OOM-kill the browser mid-order, and an
    order killed between "submitted" and "confirmed" is the one state this
    system cannot reason about afterwards.
  * Nothing about the timezone. clock.py converts into America/New_York
    explicitly with ZoneInfo, so the host clock can be UTC and usually should be.

The awkward part: the session does not travel cleanly

Your logged-in state is a Chromium profile directory — ~/.tradingagents/browser_profile
— holding the cookies for an OIDC session at auth.investopedia.com. To run on a
server you have to move it:

    python -m tradingagents.live.deploy export-profile        # -> profile.tar.gz
    scp ~/.tradingagents/browser_profile.tar.gz user@host:
    ssh user@host 'python3 -m tradingagents.live.deploy import-profile profile.tar.gz'

And then it may simply not work. A session cookie presented from a new IP, in a
new datacentre, with a different browser fingerprint, is exactly the pattern an
auth provider is built to distrust. If it is rejected you have to sign in *on
the server*, and signing in needs a visible browser. The options are all
clumsy:

  * ssh -X with an X server locally, and run `cli login` with --show-browser
  * a VNC desktop on the box (xvfb + x11vnc, or a full lightweight DE)
  * log in locally, re-export, and accept doing that again whenever it lapses

There is no clean headless login. Investopedia protects sign-in with a
challenge, and this codebase deliberately never handles your password, so the
manual step cannot be automated away. Budget for repeating it.

Also copy the keys. The repo's .env does not travel with git; the server needs
its own copy, or the equivalent exported into the service environment.

Supervision on Linux is systemd, not launchd: Restart=always,
RestartSec=600, and the same environment problem — a systemd unit inherits
nothing from a login shell either. `deploy write-cloud-bootstrap` emits a
script that sets all of that up.
"""

CLOUD_BOOTSTRAP = """#!/usr/bin/env bash
# Tier 3: provision a Debian/Ubuntu VPS to run the live desk.
#
# Run as a normal user with sudo. This installs chromium's dependencies, the
# package, and a systemd unit that restarts the desk on failure and at boot.
#
# It does NOT solve the login problem. After this runs you still have to get a
# valid Investopedia session onto the box — either import a profile exported
# from your Mac, or sign in here through X-forwarding or VNC. See CLOUD_NOTES.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/TauricResearch/TradingAgents.git}"
APP_DIR="${APP_DIR:-$HOME/TradingAgents}"
VENV="$APP_DIR/.venv"

sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip

[ -d "$APP_DIR" ] || git clone "$REPO_URL" "$APP_DIR"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$APP_DIR"
"$VENV/bin/pip" install playwright
# --with-deps pulls the shared libraries headless chromium links against.
# Without it chromium starts and dies with a linker error the adapter reports
# as a launch failure.
sudo "$VENV/bin/python" -m playwright install --with-deps chromium
"$VENV/bin/python" -m playwright install chromium

mkdir -p "$HOME/.tradingagents/logs"

# systemd inherits nothing from a login shell either — same failure as launchd.
# Keys go in an EnvironmentFile with 0600 permissions rather than in the unit.
if [ ! -f "$HOME/.tradingagents/live.env" ]; then
  cat > "$HOME/.tradingagents/live.env" <<'ENVEOF'
# One LLM key, plus any TRADINGAGENTS_* overrides. No quotes, no export.
GOOGLE_API_KEY=
TRADINGAGENTS_LLM_PROVIDER=google
ENVEOF
  chmod 600 "$HOME/.tradingagents/live.env"
  echo "edit $HOME/.tradingagents/live.env and put your key in it"
fi

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/tradingagents-live.service" <<UNITEOF
[Unit]
Description=TradingAgents live desk
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$HOME/.tradingagents/live.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$APP_DIR
ExecStart=$VENV/bin/python -m tradingagents.live.cli run
Restart=always
# Matches the launchd ThrottleInterval: a desk with no valid session exits
# immediately, and a ten-minute gap makes that legible in the journal instead
# of drowning it.
RestartSec=600

[Install]
WantedBy=default.target
UNITEOF

systemctl --user daemon-reload
systemctl --user enable tradingagents-live.service
# Without lingering, a user unit stops when the SSH session ends — the exact
# problem this whole file exists to avoid.
sudo loginctl enable-linger "$USER"

echo
echo "installed. Next, get a session onto this box, then:"
echo "  systemctl --user start tradingagents-live"
echo "  journalctl --user -u tradingagents-live -f"
"""


def write_cloud_bootstrap(path: str | Path | None = None) -> Path:
    p = Path(path) if path else (_home() / "bin" / "provision_vps.sh")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(CLOUD_BOOTSTRAP, encoding="utf-8")
    os.chmod(p, 0o755)
    return p


# --- profile portability ----------------------------------------------------

# Chromium's own lock and socket files. They name a host and a pid, so carrying
# them to another machine at best does nothing and at worst makes the profile
# look busy to a browser that is the only thing using it.
_VOLATILE = {"SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"}


def export_profile(dest: str | Path | None = None, force: bool = False) -> Path:
    """Tar the logged-in Chromium profile so it can move to another machine.

    Refuses while a browser holds the profile: half of a profile copied out
    from under a running Chromium is a profile that will not open, and the
    failure surfaces later as a mysterious logged-out state on the server.
    """
    from .investopedia import profile_dir
    src = profile_dir()
    if profile_lock_held() and not force:
        raise RuntimeError(
            f"{src} is in use by a running browser. Stop the desk first "
            f"(launchctl bootout {domain()}/{LABEL}), or pass force=True.")

    p = Path(dest) if dest else (_home() / "browser_profile.tar.gz")
    p.parent.mkdir(parents=True, exist_ok=True)

    def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        base = Path(ti.name).name
        # Sockets are dropped by tarfile itself (it cannot represent them);
        # fifos and device nodes it *can* store, and neither means anything on
        # the destination machine.
        if base in _VOLATILE or ti.isfifo() or ti.isdev():
            return None
        # uid/gid are meaningless on the destination box.
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti

    with tarfile.open(p, "w:gz") as tf:
        tf.add(src, arcname="browser_profile", filter=_filter)
    return p


def import_profile(archive: str | Path, dest: str | Path | None = None,
                   force: bool = False) -> Path:
    """Unpack a profile exported by :func:`export_profile`.

    An existing profile is moved aside rather than merged: two half-profiles
    interleaved produce a Chromium that starts and is signed out, which reads
    as "the session expired" and sends you looking in the wrong place.
    """
    from .investopedia import profile_dir
    target = Path(dest) if dest else profile_dir()
    if profile_lock_held() and not force:
        raise RuntimeError(f"{target} is in use by a running browser; stop the desk first.")

    stage = target.with_name(f"{target.name}.incoming-{datetime.now():%Y%m%d_%H%M%S}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive, "r:*") as tf:
            # Python >= 3.11.4 can refuse members that escape the destination
            # (the tarfile path-traversal problem). Older interpreters get the
            # unfiltered extract, which is why the archive should be one you
            # made yourself.
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tf.extractall(stage, **kw)
        # export_profile writes a single `browser_profile/` top level; unwrap it
        # so `dest` decides the directory name rather than the archive.
        entries = list(stage.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else stage
        if target.exists() and any(target.iterdir()):
            target.rename(target.with_name(
                f"{target.name}.bak-{datetime.now():%Y%m%d_%H%M%S}"))
        elif target.exists():
            target.rmdir()
        root.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return target


# --- what to actually run ---------------------------------------------------

def instructions(cfg: DeployConfig | None = None) -> str:
    cfg = cfg or DeployConfig()
    p = plist_path(cfg.label)
    return f"""Tier 1 — detached from the terminal, dies at reboot:
  {nohup_command(cfg)}

Tier 2 — LaunchAgent (survives crash, logout and reboot):
  python -m tradingagents.live.deploy preflight
  python -m tradingagents.live.deploy install
  python -m tradingagents.live.deploy status
  python -m tradingagents.live.deploy logs -f
  python -m tradingagents.live.deploy uninstall

  plist   {p}   (mode 0600 — it holds your API keys in cleartext)
  stdout  {out_log(cfg.label)}
  stderr  {err_log(cfg.label)}

  Raw launchctl, if you prefer it:
    launchctl bootstrap {domain()} {p}
    launchctl bootout   {domain()}/{cfg.label}
    launchctl kickstart -k {domain()}/{cfg.label}

Tier 3 — a Linux VPS:
  python -m tradingagents.live.deploy cloud

Stopping it: `cli stop` engages the kill switch and the desk keeps running and
journalling without placing orders. `bootout` stops the process outright. Use
the kill switch unless you actually want the process gone — and remember the
LaunchAgent holds the browser profile, so `cli login` and `cli portfolio` will
collide with it until it is stopped."""


def main(argv: list[str] | None = None) -> int:
    import argparse

    # allow_abbrev=False, and no two flags sharing a prefix. This is not
    # fastidiousness: an earlier revision had `--dry-run` on the parent and
    # `--dry-run-desk` on the install subcommand, argparse resolved
    # `install --dry-run` to the unique abbreviation of `--dry-run-desk`, and
    # the "rehearsal" bootstrapped a live trading agent. A flag whose whole
    # purpose is "change nothing" must not be reachable by abbreviation.
    def _parser(prog: str, **kw) -> argparse.ArgumentParser:
        return argparse.ArgumentParser(prog=prog, allow_abbrev=False, **kw)

    p = _parser("deploy", description="run the live desk continuously")
    p.add_argument("--label", default=LABEL)
    p.add_argument("--plist", default=None, help="write/read this path instead")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_parser)

    def _add(name: str, help: str, dry: bool = False):
        s = sub.add_parser(name, help=help)
        if dry:
            s.add_argument("--dry-run", action="store_true",
                           help="print what would happen; touch nothing")
        return s

    s = _add("preflight", "check everything before installing")
    s.add_argument("--skip-session", action="store_true",
                   help="do not open a browser to test the Investopedia login")
    s = _add("plist", "print the generated plist")
    s.add_argument("--raw", action="store_true",
                   help="show API keys instead of redacting them")
    s = _add("install", "install and start the LaunchAgent", dry=True)
    s.add_argument("--no-caffeinate", action="store_true",
                   help="do not hold a power assertion (the Mac may sleep)")
    s.add_argument("--no-submit", action="store_true",
                   help="the desk fills tickets but never submits an order")
    s.add_argument("--skip-session", action="store_true",
                   help="do not open a browser to test the Investopedia login")
    s.add_argument("--force", action="store_true", help="install despite preflight")
    _add("uninstall", "stop and remove the LaunchAgent", dry=True)
    _add("status", "what launchd thinks of the job", dry=True)
    _add("restart", "kill and respawn the job", dry=True)
    s = _add("logs", "tail the job's logs")
    s.add_argument("-n", type=int, default=40)
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("--stream", choices=["both", "out", "err"], default="both")
    _add("nohup-script", "write the tier-1 launcher script")
    _add("cloud", "what a VPS deployment actually requires")
    _add("write-cloud-bootstrap", "write the VPS provisioning script")
    s = _add("export-profile", "tar the logged-in browser profile")
    s.add_argument("dest", nargs="?", default=None)
    s.add_argument("--force", action="store_true",
                   help="copy even while a browser holds the profile")
    s = _add("import-profile", "unpack a browser profile")
    s.add_argument("archive")
    s.add_argument("--force", action="store_true")
    _add("help", "the commands, with paths filled in")

    a = p.parse_args(argv)
    dry = getattr(a, "dry_run", False)
    cfg = DeployConfig(label=a.label)
    if getattr(a, "no_caffeinate", False):
        cfg.caffeinate = False
    if getattr(a, "no_submit", False):
        cfg.run_args = ["--dry-run"]

    session = not getattr(a, "skip_session", False)
    if a.cmd == "preflight":
        print(format_checks(preflight_check(cfg, check_session=session)))
        return 0
    if a.cmd == "plist":
        sys.stdout.write(render_plist(cfg, redact=not a.raw).decode("utf-8"))
        return 0
    if a.cmd == "install":
        rep = install(cfg, path=a.plist, dry_run=dry, force=a.force,
                      check_session=session)
        print(format_checks(rep["checks"]), "\n")
        if rep.get("blocked"):
            print(f"not installed — failed: {', '.join(rep['blocked'])}")
            print("re-run with --force to install anyway")
            return 1
        for st in rep["steps"]:
            print(st)
        print(f"\nplist  {rep['plist']}\nstdout {rep['out_log']}\n"
              f"stderr {rep['err_log']}")
        if not dry:
            print("\n" + format_status(rep["status"]))
            # A job that has just been bootstrapped reports a transient state
            # and an empty log; the only honest confirmation is watching the
            # first cycle arrive.
            print(f"\nwatch it come up:  python -m {_MODULE} logs -f")
        return 0 if rep["ok"] else 1
    if a.cmd == "uninstall":
        rep = uninstall(a.label, path=a.plist, dry_run=dry)
        for st in rep["steps"]:
            print(st)
        print(f"plist removed: {rep['plist_removed']}")
        return 0
    if a.cmd == "status":
        print(format_status(status(a.label, path=a.plist, dry_run=dry)))
        return 0
    if a.cmd == "restart":
        for st in restart(a.label, dry_run=dry):
            print(st)
        return 0
    if a.cmd == "logs":
        out = tail_logs(a.n, a.label, a.stream, a.follow)
        if out:
            print(out)
        return 0
    if a.cmd == "nohup-script":
        print(write_nohup_script(cfg=cfg))
        return 0
    if a.cmd == "cloud":
        print(CLOUD_NOTES)
        return 0
    if a.cmd == "write-cloud-bootstrap":
        print(write_cloud_bootstrap())
        return 0
    if a.cmd == "export-profile":
        try:
            print(export_profile(a.dest, force=a.force))
        except (RuntimeError, OSError) as exc:
            print(exc)
            return 1
        return 0
    if a.cmd == "import-profile":
        try:
            print(import_profile(a.archive, force=a.force))
        except (RuntimeError, OSError, tarfile.TarError) as exc:
            print(exc)
            return 1
        return 0
    print(instructions(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
