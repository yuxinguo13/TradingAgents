"""The panel's seats as local Claude Code calls. No process is started here."""

import json
import subprocess
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.live.brain import PANEL, Panel
from tradingagents.live.broker import Account
from tradingagents.live.claude_panel import DEFAULT_MODEL, ClaudeCodeLLM, PanelCallError
from tradingagents.live.secretary import Secretary


def done(result, *, returncode=0, is_error=False, stderr=""):
    body = json.dumps({"type": "result", "subtype": "success",
                       "is_error": is_error, "result": result})
    return SimpleNamespace(returncode=returncode, stdout=body, stderr=stderr)


class Recorder:
    """Stands in for subprocess.run and remembers every process it was asked for."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        reply = self.replies.pop(0) if self.replies else done("{}")
        if isinstance(reply, BaseException):
            raise reply
        return reply


def seat(tmp_path, *replies):
    rec = Recorder(*replies)
    return ClaudeCodeLLM(binary="/opt/claude", workdir=tmp_path, run=rec), rec


def ask(llm):
    return llm.invoke([SystemMessage(content="SYSTEM-TEXT"),
                       HumanMessage(content="EVIDENCE-TEXT")])


def vote(action="Buy", qty=100, conf=0.7, why="because"):
    return done(json.dumps({"action": action, "symbol": "ACME", "quantity": qty,
                            "order_type": "Market", "limit_price": None,
                            "confidence": conf, "rationale": why}))


@pytest.mark.unit
class TestIsolation:
    def test_a_juror_gets_no_tools_settings_mcp_or_slash_commands(self, tmp_path):
        llm, rec = seat(tmp_path, done("ok"))
        ask(llm)
        (cmd, _), = rec.calls
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--setting-sources") + 1] == ""
        for flag in ("--strict-mcp-config", "--disable-slash-commands",
                     "--no-session-persistence"):
            assert flag in cmd
        assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM-TEXT"
        assert cmd[-1] == "EVIDENCE-TEXT"

    def test_bare_mode_is_never_used(self, tmp_path):
        """--bare skips the keychain, and the subscription login lives there."""
        llm, rec = seat(tmp_path, done("ok"))
        ask(llm)
        assert "--bare" not in rec.calls[0][0]

    def test_the_child_inherits_neither_the_session_nor_an_api_key(
            self, tmp_path, monkeypatch):
        """An API key in the parent would move the panel onto metered billing;
        CLAUDECODE would make a terminal test differ from the launchd run."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
        monkeypatch.setenv("TRADINGAGENTS_BROKER", "alpaca")
        llm, rec = seat(tmp_path, done("ok"))
        ask(llm)
        env = rec.calls[0][1]["env"]
        assert not any("ANTHROPIC" in k or "CLAUDE" in k for k in env)
        assert "TRADINGAGENTS_BROKER" not in env
        assert "HOME" in env and env["PATH"].startswith("/usr/bin")

    def test_it_runs_outside_the_repository(self, tmp_path, monkeypatch):
        """From the checkout, Claude Code loads this project's memory."""
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        rec = Recorder(done("ok"))
        ask(ClaudeCodeLLM(binary="/opt/claude", run=rec))
        assert rec.calls[0][1]["cwd"] == str(tmp_path / "panel")
        assert (tmp_path / "panel").is_dir()

    def test_the_model_defaults_and_can_be_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRADINGAGENTS_PANEL_MODEL", raising=False)
        assert ClaudeCodeLLM(binary="x", workdir=tmp_path).model == DEFAULT_MODEL
        monkeypatch.setenv("TRADINGAGENTS_PANEL_MODEL", "opus")
        llm, rec = seat(tmp_path, done("ok"))
        ask(llm)
        cmd = rec.calls[0][0]
        assert cmd[cmd.index("--model") + 1] == "opus"


@pytest.mark.unit
class TestReplies:
    def test_the_result_text_comes_back_as_content(self, tmp_path):
        llm, _ = seat(tmp_path, done('{"action": "Hold"}'))
        assert ask(llm).content == '{"action": "Hold"}'

    @pytest.mark.parametrize("reply", [
        done("x", returncode=1, stderr="not logged in"),
        done("rate limited", is_error=True),
        SimpleNamespace(returncode=0, stdout="<html>", stderr=""),
        SimpleNamespace(returncode=0, stdout="[1, 2]", stderr=""),
        done("   "),
        subprocess.TimeoutExpired(cmd="claude", timeout=1),
        FileNotFoundError("no such file"),
    ], ids=["exit", "is_error", "not-json", "not-object", "empty", "timeout", "missing"])
    def test_a_failed_call_raises_rather_than_answering(self, tmp_path, reply):
        """A raise is an abstention in brain.Panel. Returning text here would be
        read as a vote nobody cast."""
        llm, _ = seat(tmp_path, reply)
        with pytest.raises(PanelCallError):
            ask(llm)


@pytest.mark.unit
class TestThePanelItSits:
    def acct(self):
        return Account(account_value=100_000.0, cash=100_000.0, buying_power=100_000.0)

    def test_every_seat_is_its_own_process_and_none_sees_another_vote(self, tmp_path):
        replies = [vote(why=f"RATIONALE-{i}") for i in range(len(PANEL))]
        replies.append(done(json.dumps({"veto": False, "concern": "c", "scale": 1.0})))
        llm, rec = seat(tmp_path, *replies)
        result = Panel(llm, Secretary()).deliberate("ACME", "EVIDENCE", self.acct(), 50.0)

        assert result.consensus == "Buy" and result.order is not None
        assert len(rec.calls) == len(PANEL) + 1
        persona_prompts = [cmd[-1] for cmd, _ in rec.calls[:len(PANEL)]]
        assert all("RATIONALE-" not in p for p in persona_prompts)

    def test_a_seat_that_times_out_abstains_instead_of_voting_hold(self, tmp_path):
        """Counted as Hold, one dead process would drag a 3-1 Buy to a split."""
        replies = [subprocess.TimeoutExpired(cmd="claude", timeout=1)]
        replies += [vote() for _ in range(len(PANEL) - 1)]
        replies.append(done(json.dumps({"veto": False, "concern": "c", "scale": 1.0})))
        llm, _ = seat(tmp_path, *replies)
        result = Panel(llm, Secretary()).deliberate("ACME", "EVIDENCE", self.acct(), 50.0)

        assert result.votes[0].error and result.consensus == "Buy"


@pytest.mark.unit
def test_a_hold_keeps_its_sentence(tmp_path):
    """The order parser drops a Hold's rationale, and the page needs it."""
    replies = [vote(action="Hold", qty=0, conf=0.3, why=f"HOLD-{i}") for i in range(len(PANEL))]
    llm, _ = seat(tmp_path, *replies)
    acct = Account(account_value=100_000.0, cash=100_000.0, buying_power=100_000.0)
    result = Panel(llm, Secretary()).deliberate("ACME", "EVIDENCE", acct, 50.0)
    assert result.consensus == "Hold"
    assert [v.rationale for v in result.votes] == [f"HOLD-{i}" for i in range(len(PANEL))]
    assert all(v.confidence == pytest.approx(0.3) for v in result.votes)
