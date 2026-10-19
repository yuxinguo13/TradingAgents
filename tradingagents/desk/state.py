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
touched. It lays the local files onto the branch's current tree: files are
added or updated, never deleted, and ``*.jsonl`` logs are merged line by line,
so a session holding an older copy cannot erase what a later one saved. A
non-fast-forward (another session pushed first) is rebuilt on the newer commit.

The stops themselves never depend on this: they rest at the venue, and
``pack`` rebuilds a lost book row from the resting orders.
"""

from __future__ import annotations

import json
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
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
            n = _overlay(Path(tmp), target)
        return f"pulled {BRANCH} @ {head[:10]} into {target} ({n} files)"
    except Exception as exc:
        logger.warning("state pull failed: %s", exc)
        return f"state pull failed: {exc}"


def merge_jsonl(remote: str, local: str) -> str:
    """Union of two append-only logs: every remote line, then the local lines
    the remote does not have, in their local order. A line is a record; a
    record is never dropped because one session's copy was older."""
    seen = set()
    out = []
    for text in (remote, local):
        for line in text.splitlines():
            if line.strip() and line not in seen:
                seen.add(line)
                out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def newer_book(remote: str, local: str) -> str:
    """Of two copies of a book, the one saved later. A copy without a
    ``saved_at`` counts as oldest. 2026-10-14: another session's older copy
    overwrote the book and undid a stop raise and a resting buy."""
    def stamp(text: str) -> str:
        try:
            return str(json.loads(text).get("saved_at") or "")
        except ValueError:
            return ""
    return remote if stamp(remote) > stamp(local) else local


def _build_tree(root: Path, source: Path, parent: str | None, env: dict) -> tuple[str, int]:
    """The remote tree with the local files laid on top.

    Starts from what the branch already holds, so a session with an older copy
    adds and updates but never deletes what another session saved after it
    pulled (2026-09-30 and 2026-10-12 lost a day's log that way). Append-only
    logs (``*.jsonl``) are merged line by line rather than overwritten.
    """
    files = [p for p in source.rglob("*") if p.is_file() and p.stat().st_size <= MAX_FILE_BYTES
             and "__pycache__" not in p.parts]
    if parent is not None:
        _git("read-tree", parent, cwd=root, env=env)
    else:
        _git("read-tree", "--empty", cwd=root, env=env)
    plain, merged = [], []
    for p in files:
        rel = str(p.relative_to(source))
        if p.suffix == ".jsonl" and parent is not None:
            remote = _git("show", f"{parent}:{rel}", cwd=root, check=False)
            if remote:
                merged.append((rel, merge_jsonl(remote + "\n", p.read_text(encoding="utf-8"))))
                continue
        if p.name == "book.json" and parent is not None:
            remote = _git("show", f"{parent}:{rel}", cwd=root, check=False)
            if remote:
                merged.append((rel, newer_book(remote, p.read_text(encoding="utf-8"))))
                continue
        plain.append(rel)
    if plain:
        _git("--work-tree", str(source), "add", "-f", "--", *plain, cwd=root, env=env)
    for rel, text in merged:
        r = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=root, input=text,
                           capture_output=True, text=True, check=True)
        _git("update-index", "--add", "--cacheinfo", f"100644,{r.stdout.strip()},{rel}", cwd=root, env=env)
    return _git("write-tree", cwd=root, env=env), len(files)


def _overlay(src: Path, dst: Path) -> int:
    """Copy the branch's files over the local ones; append-only logs are
    merged, so lines logged locally since the last push survive a pull."""
    import shutil
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".jsonl" and out.exists():
            merged = merge_jsonl(p.read_text(encoding="utf-8"), out.read_text(encoding="utf-8"))
            out.write_text(merged, encoding="utf-8")
        elif p.name == "book.json" and out.exists():
            out.write_text(newer_book(p.read_text(encoding="utf-8"), out.read_text(encoding="utf-8")),
                           encoding="utf-8")
        else:
            shutil.copy2(p, out)
        n += 1
    return n


def push(source: Path | None = None, message: str = "desk state") -> str:
    """Lay the desk directory onto the branch (add/update, never delete). Never raises."""
    source = source or home()
    root = repo_root()
    if root is None:
        return "no git repository; state stays local"
    if not source.exists():
        return "nothing to push"
    try:
        for attempt in range(3):
            parent = remote_head(root)
            with tempfile.TemporaryDirectory() as tmp:
                env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
                tree, n = _build_tree(root, source, parent, env)
            if n == 0:
                return "nothing to push"
            if parent is not None and _git("rev-parse", f"{parent}^{{tree}}", cwd=root) == tree:
                return f"{BRANCH} already up to date"
            args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
            commit = _git(*args, cwd=root, env={"GIT_AUTHOR_NAME": "desk", "GIT_COMMITTER_NAME": "desk",
                                                 "GIT_AUTHOR_EMAIL": "desk@localhost",
                                                 "GIT_COMMITTER_EMAIL": "desk@localhost"})
            r = subprocess.run(["git", "push", "-q", REMOTE, f"{commit}:refs/heads/{BRANCH}"],
                               cwd=root, capture_output=True, text=True)
            if r.returncode == 0:
                return f"pushed {BRANCH} @ {commit[:10]} ({n} files)"
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
