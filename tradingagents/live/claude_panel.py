"""The panel's seats, each sat by its own local Claude Code process.

``brain.Panel`` asks a model one question per seat and does everything else in
code: the seat weights, the majority, the 0.55 conviction floor, the median
share count, the Risk Officer's veto and scale. From the model it needs exactly
one thing — send a system prompt and a message, get text back — and this module
supplies that by running ``claude -p`` once per question.

Why one process per seat rather than one conversation playing four parts: the
panel's whole argument is that the votes are independent. A model that has just
written the Aggressive Analyst's case cannot un-read it before voting as the
Conservative one. A fresh process has never seen another seat's answer.

What every call is cut off from, deliberately (each checked on 2026-09-14 by
asking a juror what tools and memory it had — "NONE" to both):

* **Tools** — ``--tools ""``. A juror reads the evidence pack; it does not go
  and fetch a different one.
* **Settings, memory, project instructions** — ``--setting-sources ""`` and a
  working directory outside the repository. Run from the checkout, Claude Code
  would load this project's memory, which is a running commentary on these very
  positions; a juror that has read it is not independent of whoever wrote it.
* **MCP servers and slash commands** — ``--strict-mcp-config``,
  ``--disable-slash-commands``.
* **The caller's environment** — the child gets a minimal environment, never
  ``os.environ``. Launched from inside a Claude Code session it would otherwise
  inherit ``CLAUDECODE`` and the session's messaging socket and run as that
  session's child, so a test from a terminal would not be the 15:00 run under
  launchd. And an ``ANTHROPIC_API_KEY`` in the parent — ``.env`` is loaded into
  it — would quietly move the panel from the subscription to metered billing.

``--bare`` is not used, although it would do much of this in one flag: it also
skips the keychain, which is where the subscription login lives, and every call
then fails to authenticate.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 180.0

# Passed through because the process cannot run without them: HOME finds the
# login, TMPDIR is where it writes, LANG keeps a Chinese company name in the
# evidence from being mangled on the way in. Nothing else crosses.
_ENV_KEEP = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL")


class PanelCallError(RuntimeError):
    """A seat got no usable answer. ``brain.Panel`` records it as an abstention,
    which is the point: a process that timed out has not voted Hold."""


@dataclass
class Reply:
    """The shape ``brain.Panel._ask`` reads: an object with ``content``."""

    content: str


def claude_bin() -> str:
    """The CLI, found without a login shell's PATH (launchd has none)."""
    explicit = os.getenv("TRADINGAGENTS_CLAUDE_BIN")
    if explicit:
        return explicit
    local = Path.home() / ".local" / "bin" / "claude"
    if local.exists():
        return str(local)
    return shutil.which("claude") or "claude"


def panel_workdir() -> Path:
    home = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
    d = home / "panel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _split(messages) -> tuple[str, str]:
    """System and user text from LangChain messages or (role, text) pairs.

    ``brain.Panel`` sends exactly [SystemMessage, HumanMessage]. Anything else is
    folded rather than refused: every system part becomes the system prompt,
    every other part the message, in order.
    """
    system, user = [], []
    for m in messages:
        if isinstance(m, tuple) and len(m) == 2:
            role, text = m
        else:
            role = getattr(m, "type", "") or type(m).__name__
            text = getattr(m, "content", "")
        is_system = str(role).lower() in ("system", "systemmessage")
        (system if is_system else user).append(str(text))
    return "\n\n".join(system), "\n\n".join(user)


class ClaudeCodeLLM:
    """A LangChain-shaped ``invoke`` over one isolated ``claude -p`` call."""

    def __init__(self, model: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 binary: str | None = None, workdir: Path | None = None, run=None):
        self.model = model or os.getenv("TRADINGAGENTS_PANEL_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.binary = binary or claude_bin()
        self.workdir = Path(workdir) if workdir else None
        self._run = run or subprocess.run
        self.calls = 0

    def command(self, system: str, user: str) -> list[str]:
        return [self.binary, "-p",
                "--model", self.model,
                "--output-format", "json",
                "--no-session-persistence",
                "--tools", "",
                "--setting-sources", "",
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--system-prompt", system,
                user]

    @staticmethod
    def environment() -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_KEEP if os.environ.get(k)}
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return env

    def invoke(self, messages) -> Reply:
        system, user = _split(messages)
        self.calls += 1
        try:
            proc = self._run(self.command(system, user),
                             cwd=str(self.workdir or panel_workdir()),
                             env=self.environment(), capture_output=True, text=True,
                             timeout=self.timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise PanelCallError(f"no answer within {self.timeout:.0f}s") from exc
        except OSError as exc:
            raise PanelCallError(f"could not start {self.binary}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:300]
            raise PanelCallError(f"exit {proc.returncode}: {detail}")
        try:
            data = json.loads(proc.stdout)
        except (TypeError, ValueError) as exc:
            raise PanelCallError(f"unreadable reply: {(proc.stdout or '')[:200]!r}") from exc
        if not isinstance(data, dict):
            raise PanelCallError(f"unexpected reply shape: {type(data).__name__}")
        if data.get("is_error"):
            raise PanelCallError(f"{data.get('subtype') or 'error'}: "
                                 f"{str(data.get('result') or '')[:300]}")
        text = data.get("result")
        if not isinstance(text, str) or not text.strip():
            raise PanelCallError("empty reply")
        return Reply(content=text)
