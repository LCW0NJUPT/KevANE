#!/bin/bash
# Compare idle and KevANE request power. Run as a normal user in kevane-runtime.
# Only powermetrics itself gets sudo; neither Python nor launchd runs as root.
set -euo pipefail

if (( EUID == 0 )); then
  echo "Run as your normal user: conda activate kevane-runtime && bash scripts/23_profile_ane.sh" >&2
  exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the kevane-runtime Conda environment first." >&2
  exit 2
fi

DUR="${1:-20}"
if [[ ! "$DUR" =~ ^[1-9][0-9]*$ ]] || (( DUR > 120 )); then
  echo "Duration must be 1–120 seconds." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SERVICE="gui/$(id -u)/info.kevane.systemone"
STARTED_SERVICE=0
LOAD_PID=""
cleanup() {
  if [[ -n "$LOAD_PID" ]]; then
    kill "$LOAD_PID" 2>/dev/null || true
    wait "$LOAD_PID" 2>/dev/null || true
  fi
  if (( STARTED_SERVICE )); then
    bash scripts/kevane_service.sh stop || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

python scripts/24_inspect_ane_plan.py
sudo -v
if ! launchctl print "$SERVICE" >/dev/null 2>&1; then
  bash scripts/kevane_service.sh start
  STARTED_SERVICE=1
fi

OUT_DIR="benchmarks/results/ane-profile"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
IDLE="$OUT_DIR/${STAMP}-idle.txt"
LOAD="$OUT_DIR/${STAMP}-load.txt"
REQUEST_LOG="$OUT_DIR/${STAMP}-requests.txt"

echo "Sampling idle system power for 5 seconds..."
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 10 > "$IDLE"

echo "Sampling ${DUR}s of KevANE requests (about 5 per second)..."
export KEVANE_PROFILE_KEY="${KEVANE_API_KEY:-local}"
export KEVANE_PROFILE_DURATION="$DUR"
python -u - > "$REQUEST_LOG" 2>&1 <<'PY' &
import json
import os
import time
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
body = {"model": "kevane-0.6b", "state": "你好", "questions": {
    "q": {"type": "choice", "instructions": "请选择问候", "criteria": {"问候": None}}}}
payload = json.dumps(body).encode()
deadline = time.monotonic() + int(os.environ["KEVANE_PROFILE_DURATION"]) + 5
count = 0
while time.monotonic() < deadline:
    req = urllib.request.Request("http://127.0.0.1:8008/v1/systemone", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["KEVANE_PROFILE_KEY"]})
    with opener.open(req, timeout=35) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        response.read()
    count += 1
    time.sleep(0.2)
print(f"completed {count} predictions", flush=True)
PY
LOAD_PID=$!
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n "$((DUR * 2))" > "$LOAD"
kill "$LOAD_PID" 2>/dev/null || true
wait "$LOAD_PID" 2>/dev/null || true
LOAD_PID=""

python - "$IDLE" "$LOAD" "$REQUEST_LOG" <<'PY'
import pathlib
import re
import statistics
import sys

for label, name in zip(("idle", "requests"), sys.argv[1:3]):
    raw = pathlib.Path(name).read_text(errors="replace")
    powers = re.findall(r"CPU Power: (\d+) mW\s+GPU Power: (\d+) mW\s+ANE Power: (\d+) mW", raw)
    if not powers:
        raise SystemExit(f"No complete power samples in {name}")
    means = [round(statistics.mean(int(sample[i]) for sample in powers)) for i in range(3)]
    print(f"{label}: {len(powers)} samples; CPU {means[0]} mW, GPU {means[1]} mW, ANE {means[2]} mW")
print(f"request log: {sys.argv[3]}")
print("System-wide power is not per-process attribution; compare deltas and keep other apps idle.")
PY
