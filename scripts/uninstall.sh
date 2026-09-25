#!/bin/bash
# Remove only the user-level installation managed by scripts/install.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/kevane"
LAUNCHER="$HOME/.local/bin/kev-ane"
OLD_LAUNCHER="$HOME/.local/bin/kevane"
REMOVE_MODEL=0

case "${1:-}" in
  '') ;;
  --remove-model) REMOVE_MODEL=1 ;;
  *) echo "Usage: kev-ane uninstall [--remove-model]" >&2; exit 2 ;;
esac
if [[ ! -f "$STATE/install-root" || "$(cat "$STATE/install-root")" != "$ROOT" ]]; then
  echo "This checkout is not the installed KevANE source directory." >&2
  exit 2
fi

MODEL="$(cat "$STATE/install-model" 2>/dev/null || true)"
MODEL_OWNED="$(cat "$STATE/model-owned" 2>/dev/null || true)"
/bin/bash "$ROOT/scripts/kevane_service.sh" stop
rm -rf "$ROOT/build/compiled"
if [[ -f "$STATE/created-env" && "$(cat "$STATE/created-env")" == kevane-runtime ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is needed to remove the environment created by KevANE." >&2
    exit 2
  fi
  conda env remove -n kevane-runtime -y
fi

if [[ -f "$LAUNCHER" ]] && grep -Fq '# Managed by KevANE install.sh' "$LAUNCHER"; then
  rm "$LAUNCHER"
fi
if [[ -f "$OLD_LAUNCHER" ]] && grep -Fq '# Managed by KevANE install.sh' "$OLD_LAUNCHER"; then
  rm "$OLD_LAUNCHER"
fi
rm -f "$STATE/info.kevane.systemone.plist" "$STATE/server.log" \
      "$STATE/install-root" "$STATE/install-python" "$STATE/install-model" \
      "$STATE/model-owned" "$STATE/created-env"
rmdir "$STATE" 2>/dev/null || true

if (( REMOVE_MODEL )) && [[ "$MODEL_OWNED" == 1 && "$MODEL" == "$ROOT/hf-model" ]]; then
  rm -rf "$MODEL/kev-qwen3-0.6b-hidden.mlpackage" "$MODEL/.cache"
  rm -f "$MODEL/kev-qwen3-0.6b-hidden-meta.json" \
        "$MODEL/kev-qwen3-0.6b-pointer-head.pt" \
        "$MODEL/tokenizer.json" "$MODEL/tokenizer_config.json" \
        "$MODEL/chat_template.jinja"
fi
echo "KevANE service and command removed. Source checkout remains at $ROOT"
if [[ "$MODEL_OWNED" == 1 && "$REMOVE_MODEL" == 0 ]]; then
  echo "Downloaded model files were kept; use --remove-model to delete them."
elif [[ "$MODEL_OWNED" == 0 ]]; then
  echo "The supplied model directory was kept: $MODEL"
fi
