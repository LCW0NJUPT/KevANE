#!/bin/bash
# Launch only on demand; wait for the server and Core ML worker to exit on stop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="info.kevane.systemone"
SERVICE="gui/$(id -u)/$LABEL"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/kevane"
PLIST="$STATE/$LABEL.plist"
LOG="$STATE/server.log"
HEALTH="http://127.0.0.1:8008/healthz"
loaded() { launchctl print "$SERVICE" >/dev/null 2>&1; }
job_pid() { launchctl print "$SERVICE" 2>/dev/null | awk '/^[[:space:]]*pid = [0-9]+/ { print $3; exit }'; }
health() { curl --noproxy '*' -fsS --max-time 2 "$HEALTH" 2>/dev/null; }
json_pid() { sed -n "s/.*\"$1\":\([0-9][0-9]*\).*/\1/p"; }
alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

wait_ready() {
  local parent response
  for ((i=0; i<180; i++)); do
    parent="$(job_pid)"
    response="$(health || true)"
    if [[ -n "$parent" && "$(printf '%s' "$response" | json_pid pid)" == "$parent" ]]; then
      echo "KevANE ready at $HEALTH (PID $parent)."; return 0
    fi
    if ! loaded || [[ -z "$parent" && $i -gt 8 ]]; then
      echo "KevANE exited during startup; see $LOG" >&2
      stop_service || true
      return 1
    fi
    sleep 0.5
  done
  echo "KevANE startup timed out; see $LOG" >&2
  stop_service || true
  return 1
}

stop_service() {
  if ! loaded; then echo "KevANE is stopped."; return 0; fi
  local parent worker children child remaining=0
  parent="$(job_pid)"
  worker="$(health | json_pid worker_pid || true)"
  children=""
  if [[ -n "$parent" ]]; then
    children="$(pgrep -P "$parent" 2>/dev/null || true)"
    kill -TERM "$parent" 2>/dev/null || true
    # Give Uvicorn's shutdown handler and backbone.close() time to release Core ML.
    for ((i=0; i<120; i++)); do
      if ! alive "$parent" && ! alive "$worker"; then break; fi
      sleep 0.25
    done
  fi
  launchctl bootout "$SERVICE" 2>/dev/null || true
  # A launchd bootout can kill the HTTP process before Python reaps a spawned
  # worker. Only touch PIDs captured from this service before shutdown.
  for child in $children $worker; do
    if alive "$child"; then kill -TERM "$child" 2>/dev/null || true; fi
  done
  for ((i=0; i<40; i++)); do
    remaining=0
    for child in $children $worker; do
      if alive "$child"; then remaining=1; fi
    done
    if (( ! remaining )); then break; fi
    sleep 0.25
  done
  for child in $children $worker; do
    if alive "$child"; then kill -KILL "$child" 2>/dev/null || true; fi
  done
  for ((i=0; i<40; i++)); do
    if ! loaded && ! alive "$parent"; then break; fi
    sleep 0.25
  done
  if loaded || alive "$parent"; then
    echo "KevANE shutdown incomplete; inspect $LOG" >&2
    return 1
  fi
  for child in $children $worker; do
    if alive "$child"; then
      echo "KevANE worker $child is still alive; inspect it before restarting." >&2
      return 1
    fi
  done
  echo "KevANE stopped; HTTP and Core ML worker processes exited."
}

case "${1:-}" in
  start)
    case "${2:-}" in
      '') wait_for_ready=0 ;;
      --wait) wait_for_ready=1 ;;
      *) echo "Usage: kev-ane start [--wait]" >&2; exit 2 ;;
    esac
    if (( $# > 2 )); then echo "Usage: kev-ane start [--wait]" >&2; exit 2; fi
    if loaded; then
      parent="$(job_pid)"
      response="$(health || true)"
      if [[ -n "$parent" && "$(printf '%s' "$response" | json_pid pid)" == "$parent" ]]; then
        echo "KevANE is already running (PID $parent)."; exit 0
      fi
      if alive "$parent"; then
        if (( wait_for_ready )); then wait_ready; else echo "KevANE is starting (PID $parent); use 'kev-ane status' to check readiness."; fi
        exit $?
      fi
      stop_service
    fi
    if [[ -n "$(health || true)" ]]; then
      echo "Port 8008 already serves a process outside this launchd job; refusing to start." >&2
      exit 1
    fi
    if [[ -n "${KEVANE_PYTHON:-}" ]]; then
      PYTHON="$KEVANE_PYTHON"
    elif [[ -f "$PLIST" ]]; then
      PYTHON="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$PLIST" 2>/dev/null || true)"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
      PYTHON="$CONDA_PREFIX/bin/python"
    else
      PYTHON=""
    fi
    if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
      echo "Activate the KevANE runtime Conda environment for the first start." >&2; exit 2
    fi
    if ! "$PYTHON" -c 'import sys,pathlib; assert sys.version_info[:2] == (3,12) and (pathlib.Path(sys.prefix)/"conda-meta").is_dir()' 2>/dev/null; then
      echo "The selected Python is not a Python 3.12 Conda environment: $PYTHON" >&2; exit 2
    fi
    mkdir -p "$STATE"
    "$PYTHON" - "$PLIST" "$LABEL" "$ROOT" "$LOG" <<'PY'
import os, plistlib, sys
from pathlib import Path
plist, label, root, log = sys.argv[1:]
old_env = {}
if Path(plist).exists():
    old_env = plistlib.loads(Path(plist).read_bytes()).get('EnvironmentVariables', {})
env = {'PYTHONUNBUFFERED': '1', 'CONDA_PREFIX': sys.prefix,
       'CONDA_DEFAULT_ENV': Path(sys.prefix).name,
       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
       'VECLIB_MAXIMUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
       'PATH': str(Path(sys.prefix) / 'bin') + ':/usr/bin:/bin'}
api_key = os.environ.get('KEVANE_API_KEY', old_env.get('KEVANE_API_KEY'))
if api_key:
    env['KEVANE_API_KEY'] = api_key
model_dir = os.environ.get('KEVANE_MODEL_DIR')
if not model_dir and Path(plist).exists():
    previous_args = plistlib.loads(Path(plist).read_bytes()).get('ProgramArguments', [])
    if '--model-dir' in previous_args:
        model_dir = previous_args[previous_args.index('--model-dir') + 1]
program_args = [sys.executable, str(Path(root)/'scripts/30_systemone_server.py')]
if model_dir:
    program_args.extend(['--model-dir', model_dir])
entry = {'Label': label,
         'ProgramArguments': program_args,
         'WorkingDirectory': root, 'EnvironmentVariables': env,
         'RunAtLoad': True, 'KeepAlive': False,
         'ProcessType': 'Standard', 'Nice': 5, 'LowPriorityIO': True,
         'StandardOutPath': log, 'StandardErrorPath': log}
Path(plist).write_bytes(plistlib.dumps(entry))
Path(plist).chmod(0o600)
PY
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    if (( wait_for_ready )); then
      wait_ready
    else
      echo "KevANE is starting; use 'kev-ane status' to check readiness."
    fi
    ;;
  stop) stop_service ;;
  restart) stop_service; "$0" start "${2:-}" ;;
  status)
    if ! loaded; then echo "KevANE is stopped."; exit 0; fi
    parent="$(job_pid)"
    response="$(health || true)"
    if [[ -n "$parent" && "$(printf '%s' "$response" | json_pid pid)" == "$parent" ]]; then
      printf '%s\n' "$response"
    elif alive "$parent"; then
      echo "KevANE is starting (PID $parent); see $LOG if this persists."
    else
      echo "KevANE launchd job is loaded but not healthy; see $LOG" >&2
      exit 1
    fi
    ;;
  logs)
    if [[ -f "$LOG" ]]; then tail -n 80 "$LOG"; else echo "No KevANE logs yet."; fi
    ;;
  cache)
    case "${2:-}" in
      status)
        if [[ -d "$ROOT/build/compiled" ]]; then
          du -sh "$ROOT/build/compiled"
        else
          echo "No compiled Core ML cache."
        fi
        ;;
      clear)
        if loaded; then
          echo "Stop KevANE before clearing its compiled model cache." >&2; exit 1
        fi
        rm -rf "$ROOT/build/compiled"
        echo "Compiled Core ML cache removed; the next start will compile the model again."
        ;;
      *) echo "Usage: kev-ane cache {status|clear}" >&2; exit 2 ;;
    esac
    ;;
  help|-h|--help)
    echo "Usage: kev-ane {start [--wait]|stop|restart [--wait]|status|logs|cache {status|clear}|uninstall}"
    echo "The service starts on demand and does not start at login."
    ;;
  *) echo "Usage: kev-ane {start [--wait]|stop|restart [--wait]|status|logs|cache {status|clear}|uninstall}" >&2; exit 2 ;;
esac
