#!/bin/zsh
# One entry point for both of the desk's LaunchAgents.
#
#   desk_cron.sh report   # next session's page, panel seated (weekdays 15:00 PT)
#   desk_cron.sh submit   # place what the book is missing  (weekdays 06:25 PT, places ~06:35)
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

# Hold the machine awake for as long as this script lives. The pmset wake only
# gets the Mac up; with `sleep 1` it is back asleep a minute later — before the
# open, and long before a ten-minute report finishes. -w on our own PID
# releases the assertion however the script ends. (-s only counts on AC power;
# -i is what holds on battery.)
/usr/bin/caffeinate -i -s -w $$ &

# The submit agent fires at 06:25 PT, the same minute as the pmset wake,
# because a calendar job whose minute passes while the Mac sleeps waits for the
# *next* wake — hours later if nobody opens the lid. So it launches before the
# open and waits here until five minutes after 09:30 ET, when the opening
# prints have settled. If no session opens within twenty minutes (a holiday, a
# wake that came after the close) it exits 3 and places nothing. A machine that
# wakes mid-session places at once; the Secretary's 5% limit-deviation rule is
# what refuses a price that has run away from the book.
WAIT_FOR_OPEN='
import sys, time
from tradingagents.live import clock
SETTLE, MAX_WAIT = 5 * 60, 20 * 60
st = clock.market_state()
if st.is_open:
    wait = max(0.0, SETTLE - (st.minutes_from_open or 0.0) * 60)
else:
    wait = clock.seconds_until_open() + SETTLE
    if wait > MAX_WAIT + SETTLE:
        print(f"no session opens within {MAX_WAIT // 60} min "
              f"(next {clock.next_open():%a %F %H:%M %Z}); nothing placed")
        sys.exit(3)
print(f"waiting {wait / 60:.1f} min for the open to settle", flush=True)
time.sleep(wait)
'

# The panel sits on the evening report, not the morning order: every seat is a
# separate Claude Code process (claude_panel.py), a full report is dozens of
# them, and the morning run has ten minutes and nobody watching. --no-llm stays:
# it forbids an external model API, and the panel uses none.
case "$MODE" in
  report) "$PY" -m tradingagents.live.advisor --no-llm --panel claude >>"$LOG" 2>&1 ;;
  submit) "$PY" -c "$WAIT_FOR_OPEN" >>"$LOG" 2>&1 &&
          "$PY" -m tradingagents.live.execute --submit >>"$LOG" 2>&1 ;;
  *)      print -r -- "unknown mode: $MODE (want report|submit)" >>"$LOG"; exit 2 ;;
esac
rc=$?

print -r -- "-------- exit $rc --------" >>"$LOG"
exit $rc
