#!/bin/zsh
# The desk, on a schedule. One script, three jobs:
#
#   desk_cron.sh trade    # weekdays 06:25 PT — waits for the open, trades the paper account (~09:35 ET)
#   desk_cron.sh report   # weekdays 15:00 PT — the market, sector by sector
#   desk_cron.sh advise   # weekdays 15:10 PT — what to do with the portfolio in desk/advise/my_portfolio.json
#
# Three things launchd will not do for you, each of which fails silently:
#
#   1. Wake the Mac. `pmset repeat wakeorpoweron MTWRF 06:25:00` does; launchd
#      only defers a job whose minute passed while the machine slept.
#   2. Keep it awake. caffeinate below holds it for the life of this script.
#      On battery with the lid closed it holds nothing; the log says so.
#   3. Run from Desktop/Documents/Downloads. TCC refuses a LaunchAgent there.
#      Keep the repo somewhere else (~/Stock/TradingAgents).
#
# Off switch, instant: `touch ~/.tradingagents/STOP` (only `trade` reads it;
# the report and the advice are read-only and keep running).

set -u

MODE="${1:?usage: desk_cron.sh trade|report|advise}"
REPO="${TRADINGAGENTS_REPO:-/Users/emilyguo/Stock/TradingAgents}"
PY="${TRADINGAGENTS_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11}"
HOME_DIR="${TRADINGAGENTS_HOME:-$HOME/.tradingagents}"
LOG="$HOME_DIR/logs/desk-$MODE.log"

export TRADINGAGENTS_HOME="$HOME_DIR"
export PATH="/usr/local/bin:/opt/homebrew/bin:$HOME/.local/bin:$PATH"
[ -f "$HOME/.zshenv" ] && source "$HOME/.zshenv"
[ -f "$REPO/.env" ] && set -a && source "$REPO/.env" && set +a

mkdir -p "$HOME_DIR/logs"
if ! cd "$REPO"; then
  print -r -- "$(date '+%F %T') cannot cd to $REPO" >>"$LOG"
  exit 1
fi

start=$SECONDS
print -r -- "" >>"$LOG"
print -r -- "======== $(date '+%F %T %Z')  $MODE ========" >>"$LOG"
if /usr/bin/pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
  print -r -- "⚠ 电池供电：合盖后 caffeinate 拦不住睡眠，这一趟会断断续续地跑（插电可避免）" >>"$LOG"
fi
/usr/bin/caffeinate -i -s -w $$ &

set -o pipefail
case "$MODE" in
  trade)
    # waitopen exits 0 when the session is open and settled, 3 when nothing
    # opens within twenty minutes (holiday), 4 when the session was missed
    # while the Mac slept. Only 0 trades; 4 still writes the note, read-only.
    "$PY" -m tradingagents.live.waitopen >>"$LOG" 2>&1
    rc=$?
    case $rc in
      0) "$PY" -m tradingagents.desk trade >>"$LOG" 2>&1 ;;
      4) "$PY" -m tradingagents.desk trade --dry-run >>"$LOG" 2>&1 ;;
      *) (exit $rc) ;;
    esac ;;
  report)  "$PY" -m tradingagents.desk report -q >>"$LOG" 2>&1 ;;
  advise)  "$PY" -m tradingagents.desk advise -q >>"$LOG" 2>&1 ;;
  *)       print -r -- "unknown mode: $MODE (want trade|report|advise)" >>"$LOG"; exit 2 ;;
esac
rc=$?

print -r -- "-------- exit $rc · 用时 $(( (SECONDS - start) / 60 )) 分钟 --------" >>"$LOG"
exit $rc
