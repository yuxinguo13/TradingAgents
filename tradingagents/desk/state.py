"""Durable state for a desk that runs on a fresh machine every morning.

A scheduled cloud session starts from a clean VM, so ``~/.tradingagents/desk``
— the book with its theses, the decision log, the portfolio file — would be
gone by the next run. It is kept instead on a branch of this repository,
``desk-state``, that holds nothing but that directory:

    python -m tradingagents.desk state pull    # branch → ~/.tradingagents/desk
    python -m tradingagents.desk state push    # ~/.tradingagents/desk → branch

``pack trade``, ``report`` and ``advise`` pull before they read; ``order`` and
``log`` push after they write, so a session that dies mid-morning has still
saved every order it placed. The push is a plain-file snapshot committed
with a temporary index — the working tree and the code branch are never
touched — and a non-fast-forward (another session pushed first) is retried
on top of the newer commit.

The stops themselves never depend on this: they rest at the venue, and
``pack`` rebuilds a lost book row from the resting orders.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from . import home

logger = logging.getLogger(__name__)

BRANCH = os.getenv("TRADINGAGENTS_STATE_BRANCH", "desk-state")
REMOTE = "origin"
MAX_FILE_BYTES = 5 * 1024 * 1024


def repo_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def _git(*args: str, cwd: Path, env: dict | None = None, check: bool = True) -> str:
    e = {**os.environ, **(env or {})}
    r = subprocess.run(["git", *args], cwd=cwd, env=e, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def remote_head(root: Path) -> str | None:
    try:
        _git("fetch", "-q", REMOTE, BRANCH, cwd=root)
    except RuntimeError as exc:
        if "couldn't find remote ref" in str(exc) or "fatal: couldn't find" in str(exc):
            return None
        raise
    return _git("rev-parse", f"{REMOTE}/{BRANCH}", cwd=root)


def pull(target: Path | None = None) -> str:
    """Overlay the branch's files onto the desk directory. Never raises."""
    target = target or home()
    root = repo_root()
    if root is None:
        return "no git repository; state stays local"
    try:
        head = remote_head(root)
        if head is None:
            return f"no {BRANCH} branch yet; state stays local"
        target.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "archive", "--format=tar", head], cwd=root,
                                 capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(target)], input=archive.stdout, check=True)
        return f"pulled {BRANCH} @ {head[:10]} into {target}"
    except Exception as exc:
        logger.warning("state pull failed: %s", exc)
        return f"state pull failed: {exc}"


def push(source: Path | None = None, message: str = "desk state") -> str:
    """Snapshot the desk directory onto the branch. Never raises."""
    source = source or home()
    root = repo_root()
    if root is None:
        return "no git repository; state stays local"
    if not source.exists():
        return "nothing to push"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            env = {"GIT_INDEX_FILE": str(index)}
            files = [p for p in source.rglob("*") if p.is_file() and p.stat().st_size <= MAX_FILE_BYTES
                     and "__pycache__" not in p.parts]
            if not files:
                return "nothing to push"
            rel = [str(p.relative_to(source)) for p in files]
            _git("--work-tree", str(source), "add", "-f", "--", *rel, cwd=root, env=env)
            tree = _git("write-tree", cwd=root, env=env)
        for attempt in range(3):
            parent = remote_head(root)
            if parent is not None and _git("rev-parse", f"{parent}^{{tree}}", cwd=root) == tree:
                return f"{BRANCH} already up to date"
            args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
            commit = _git(*args, cwd=root, env={"GIT_AUTHOR_NAME": "desk", "GIT_COMMITTER_NAME": "desk",
                                                 "GIT_AUTHOR_EMAIL": "desk@localhost",
                                                 "GIT_COMMITTER_EMAIL": "desk@localhost"})
            r = subprocess.run(["git", "push", "-q", REMOTE, f"{commit}:refs/heads/{BRANCH}"],
                               cwd=root, capture_output=True, text=True)
            if r.returncode == 0:
                return f"pushed {BRANCH} @ {commit[:10]} ({len(files)} files)"
            if "rejected" not in r.stderr and "fetch first" not in r.stderr:
                raise RuntimeError(r.stderr.strip())
            logger.info("state push raced another session; retrying (%d)", attempt + 1)
        return "state push failed: the branch kept moving"
    except Exception as exc:
        logger.warning("state push failed: %s", exc)
        return f"state push failed: {exc}"


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk state",
                                description=f"sync ~/.tradingagents/desk with the {BRANCH} branch")
    p.add_argument("what", choices=["pull", "push"])
    p.add_argument("-m", "--message", default="desk state")
    args = p.parse_args(argv)
    out = pull() if args.what == "pull" else push(message=args.message)
    print(out)
    return 0 if "failed" not in out else 1
