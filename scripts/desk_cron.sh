#!/bin/zsh
# One entry point for both of the desk's LaunchAgents.
#
#   desk_cron.sh report   # next session's page, panel seated (weekdays 15:00 PT)
#   desk_cron.sh submit   # place what the book is missing  (weekdays 06:25 PT, places ~06:35)
#   desk_cron.sh close    # exits before the bell; core top-ups the morning missed (weekdays 12:30 PT)
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

MODE="${1:?usage: desk_cron.sh report|submit|close}"
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

start=$SECONDS
{
  print -r -- ""
  print -r -- "======== $(date '+%F %T %Z')  $MODE ========"
} >>"$LOG"

# The caffeinate below cannot hold a closed-lid Mac on battery: -s counts only
# on AC, and -i stops idle sleep, not the clamshell and maintenance sleeps that
# actually fire. The run still finishes, but in two-second slivers — the report
# that takes ten minutes at the desk took 57 on 2026-09-15. Say so in the log,
# so a run that crawled does not read afterwards as a run that was merely slow.
if /usr/bin/pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
  print -r -- "⚠ 电池供电：合盖后 caffeinate 拦不住睡眠，这一趟会断断续续地跑（插电可避免）" >>"$LOG"
fi

# Hold the machine awake for as long as this script lives. The pmset wake only
# gets the Mac up; with `sleep 1` it is back asleep a minute later — before the
# open, and long before a ten-minute report finishes. -w on our own PID
# releases the assertion however the script ends. (-s only counts on AC power;
# -i is what holds on battery.)
/usr/bin/caffeinate -i -s -w $$ &

# The submit agent fires at 06:25 PT, the same minute as the pmset wake,
# because a calendar job whose minute passes while the Mac sleeps waits for the
# *next* wake — hours later if nobody opens the lid. So it launches before the
# open and waits in `waitopen` until five minutes after 09:30 ET, when the
# opening prints have settled. That wait is against the wall clock: a sleeping
# Mac counts `time.sleep` in awake seconds only, which on 2026-09-15 turned ten
# minutes into seven hours and started `execute` twenty minutes after the close.
#
#   0  the session is open and settled — place
#   3  no session opens within twenty minutes (a holiday, a wake after the
#      close) — place nothing
#   4  the session ended while the machine slept — reconcile read-only, so the
#      morning leaves a record instead of a column of "market is closed"
#      refusals from the gate
#
# A machine that wakes mid-session places at once; the Secretary's 5%
# limit-deviation rule is what refuses a price that has run away from the book.

# The panel sits on the evening report, not the morning order: every seat is a
# separate Claude Code process (claude_panel.py), a full report is dozens of
# them, and the morning run has ten minutes and nobody watching. --no-llm stays:
# it forbids an external model API, and the panel uses none.
# What reached the venue today, filed next to that day's page. The evening
# report says what *should* be placed; this file is the only record of what
# *was* — the gap between the two is what the book used to hide.
ORDERS="$HOME_DIR/reports/$(date +%F)-orders.log"
[ "$MODE" != report ] && print -r -- "======== $(date '+%F %T %Z')  $MODE ========" >>"$ORDERS"
set -o pipefail
case "$MODE" in
  report) "$PY" -m tradingagents.live.advisor --no-llm --panel claude >>"$LOG" 2>&1 ;;
  submit) "$PY" -m tradingagents.live.waitopen >>"$LOG" 2>&1
          wait_rc=$?
          case $wait_rc in
            0) "$PY" -m tradingagents.live.execute --submit 2>&1 | tee -a "$ORDERS" >>"$LOG" ;;
            4) "$PY" -m tradingagents.live.execute 2>&1 | tee -a "$ORDERS" >>"$LOG" ;;
            *) (exit $wait_rc) ;;
          esac ;;
  # Half an hour before the bell: the exit review's signals become sell orders
  # in the same session they fired, and a core name the morning run failed to
  # top up — the first order after a wake has been refused two mornings
  # running — gets a second attempt. Same execute, same gate; nothing here is
  # a new rule, only a second reading of the existing ones.
  close)  "$PY" -m tradingagents.live.execute --submit 2>&1 | tee -a "$ORDERS" >>"$LOG" ;;
  *)      print -r -- "unknown mode: $MODE (want report|submit|close)" >>"$LOG"; exit 2 ;;
esac
rc=$?

print -r -- "-------- exit $rc · 用时 $(( (SECONDS - start) / 60 )) 分钟 --------" >>"$LOG"
exit $rc
