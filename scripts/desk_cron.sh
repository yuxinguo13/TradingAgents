#!/bin/zsh
# One entry point for both of the desk's LaunchAgents.
#
#   desk_cron.sh report   # write the next session's page   (weekdays 15:00 PT)
#   desk_cron.sh submit   # place what the book is missing  (weekdays 06:35 PT)
#
# Three things launchd will not do for you, each of which fails silently:
#
#   * It does not read ~/.zshrc or ~/.zprofile. PATH and the interpreter are
#     spelled out absolutely below; nothing here may rely on a login shell.
#   * It does not cd anywhere. The package resolves .env with
#     find_dotenv(usecwd=True), so a run started outside the repo loses the
#     Alpaca keys, the broker goes unreachable, and the report still renders —
#     priced off an *assumed* account value, with every share count on the
#     page scaled by the wrong number. Hence the hard cd, and the exit if it
#     fails.
#   * It does not serialise runs. A report takes ~10 minutes and a machine
#     that woke late can fire two in a row; mkdir is the atomic primitive
#     that stops them polling the same feeds twice ([ -d ] then mkdir is a
#     race, not a lock).
#
# To stop all trading instantly, from anywhere, without touching launchd:
#     touch ~/.tradingagents/STOP
# Both the execute entry point and the Secretary check it on every order.

set -u

MODE="${1:?usage: desk_cron.sh report|submit}"
REPO="/Users/emilyguo/Stock/TradingAgents"
PY="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
HOME_DIR="${TRADINGAGENTS_HOME:-$HOME/.tradingagents}"
LOGS="$HOME_DIR/logs"
LOCK="${TMPDIR:-/tmp}/tradingagents-desk-$MODE.lock"

mkdir -p "$LOGS"
LOG="$LOGS/desk-$MODE.log"

if ! mkdir "$LOCK" 2>/dev/null; then
  print -r -- "$(date '+%F %T %Z')  skipped: a $MODE run still holds $LOCK" >>"$LOG"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

if ! cd "$REPO"; then
  print -r -- "$(date '+%F %T %Z')  repo not found at $REPO — nothing run" >>"$LOG"
  exit 1
fi

{
  print -r -- ""
  print -r -- "======== $(date '+%F %T %Z')  $MODE ========"
} >>"$LOG"

case "$MODE" in
  report) "$PY" -m tradingagents.live.advisor --no-llm >>"$LOG" 2>&1 ;;
  submit) "$PY" -m tradingagents.live.execute --submit >>"$LOG" 2>&1 ;;
  *)      print -r -- "unknown mode: $MODE (want report|submit)" >>"$LOG"; exit 2 ;;
esac
rc=$?

print -r -- "-------- exit $rc --------" >>"$LOG"
exit $rc
