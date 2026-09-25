#!/bin/bash
# Install an on-demand user LaunchAgent without changing the active Conda env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/kevane"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/kev-ane"
OLD_LAUNCHER="$BIN_DIR/kevane"
ENV_NAME="kevane-runtime"
MODEL="$ROOT/hf-model"
MODEL_OWNED=0

usage() {
  echo "Usage: bash scripts/install.sh [--model-dir PATH]"
  echo "The default model directory is this checkout's hf-model/."
  echo "Use --model-dir to run an existing model from another directory."
}
case "${1:-}" in
  '') ;;
  --model-dir)
    if [[ $# != 2 ]]; then usage >&2; exit 2; fi
    MODEL="$(cd "$2" 2>/dev/null && pwd -P)" || {
      echo "Model directory does not exist: $2" >&2; exit 2;
    }
    ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

model_ready() {
  local dir="$1"
  [[ -s "$dir/kev-qwen3-0.6b-hidden.mlpackage/Manifest.json" &&
     -s "$dir/kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/model.mlmodel" &&
     -s "$dir/kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/weights/weight.bin" &&
     -s "$dir/kev-qwen3-0.6b-hidden-meta.json" &&
     -s "$dir/tokenizer.json" && -s "$dir/tokenizer_config.json" &&
     -s "$dir/kev-qwen3-0.6b-pointer-head.pt" ]] || return 1
  # A Git LFS pointer or interrupted download is far smaller than these files.
  [[ "$(stat -f%z "$dir/kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/weights/weight.bin")" -gt 100000000 &&
     "$(stat -f%z "$dir/kev-qwen3-0.6b-pointer-head.pt")" -gt 1000000 &&
     "$(stat -f%z "$dir/tokenizer.json")" -gt 1000000 ]]
}

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "KevANE requires an Apple Silicon Mac." >&2; exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Install Conda and make its conda command available in this shell first." >&2; exit 2
fi
if [[ -f "$STATE/install-root" && "$(cat "$STATE/install-root")" != "$ROOT" ]]; then
  echo "KevANE is installed from another directory: $(cat "$STATE/install-root")" >&2
  echo "Uninstall that copy before installing this one." >&2; exit 2
fi
if [[ -e "$LAUNCHER" ]] && ! grep -Fq '# Managed by KevANE install.sh' "$LAUNCHER"; then
  echo "Command path already exists and is not managed by KevANE: $LAUNCHER" >&2; exit 2
fi
if [[ -f "$STATE/install-root" && -f "$STATE/install-model" &&
      "$(cat "$STATE/install-root")" == "$ROOT" &&
      "$(cat "$STATE/install-model")" == "$MODEL" &&
      "$(cat "$STATE/model-owned" 2>/dev/null || true)" == 1 ]]; then
  MODEL_OWNED=1
fi

if ! model_ready "$MODEL"; then
  if [[ "$MODEL" != "$ROOT/hf-model" ]]; then
    echo "The supplied model directory is incomplete: $MODEL" >&2; exit 2
  fi
    # Check the published file list before creating an environment or
    # downloading anything. A placeholder model card is not a usable model.
    CONDA_BASE="$(conda info --base)"
    "$CONDA_BASE/bin/python" - <<'PY'
import json
import os
import sys
import urllib.request

url = 'https://huggingface.co/api/models/flylcw/KevANE-0.6B'
headers = {'User-Agent': 'KevANE-installer'}
if os.environ.get('HF_TOKEN'):
    headers['Authorization'] = 'Bearer ' + os.environ['HF_TOKEN']
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as response:
        files = {item['rfilename'] for item in json.load(response)['siblings']}
except (OSError, KeyError, ValueError) as exc:
    sys.exit(f'Could not inspect the model repository: {exc}\n'
             'Use --model-dir PATH if you already have the converted model.')
required = {
    'kev-qwen3-0.6b-hidden.mlpackage/Manifest.json',
    'kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/model.mlmodel',
    'kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/weights/weight.bin',
    'kev-qwen3-0.6b-hidden-meta.json',
    'kev-qwen3-0.6b-pointer-head.pt',
    'tokenizer.json',
    'tokenizer_config.json',
}
missing = sorted(required - files)
if missing:
    sys.exit('The Hugging Face repository does not contain the converted model yet.\n'
             'Missing: ' + ', '.join(missing) + '\n'
             'Upload the model or install with --model-dir PATH.')
PY
fi

# The Conda environment is selected explicitly; the caller's active env is
# irrelevant. The server imports src/ from its own checkout, so pip -e is
# unnecessary and no package is installed into an existing environment.
CREATED_ENV=0
if ! conda run -n "$ENV_NAME" python -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  if conda run -n "$ENV_NAME" python -c 'pass' >/dev/null 2>&1; then
    echo "Existing $ENV_NAME environment does not use Python 3.12." >&2; exit 2
  fi
  conda env create -f "$ROOT/environment/runtime.yml"
  CREATED_ENV=1
fi
PYTHON="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' | tail -n 1)"
if [[ ! -x "$PYTHON" ]]; then
  echo "Could not find the $ENV_NAME Python executable." >&2; exit 1
fi
if ! "$PYTHON" -c 'import coremltools, fastapi, huggingface_hub, torch, transformers, uvicorn' >/dev/null 2>&1; then
  echo "Runtime dependencies are missing from $ENV_NAME." >&2
  printf 'Run: conda env update -n %s -f %q\n' "$ENV_NAME" "$ROOT/environment/runtime.yml" >&2
  exit 1
fi

if ! model_ready "$MODEL"; then
  HF="$(dirname "$PYTHON")/hf"
  if [[ ! -x "$HF" ]]; then
    echo "The Hugging Face 'hf' command is missing from $ENV_NAME." >&2; exit 1
  fi
  mkdir -p "$MODEL"
  "$HF" download flylcw/KevANE-0.6B \
    kev-qwen3-0.6b-hidden.mlpackage/Manifest.json \
    kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/model.mlmodel \
    kev-qwen3-0.6b-hidden.mlpackage/Data/com.apple.CoreML/weights/weight.bin \
    kev-qwen3-0.6b-hidden-meta.json \
    kev-qwen3-0.6b-pointer-head.pt tokenizer.json tokenizer_config.json \
    chat_template.jinja --local-dir "$MODEL"
  if ! model_ready "$MODEL"; then
    echo "The model download is incomplete in $MODEL." >&2; exit 1
  fi
  MODEL_OWNED=1
fi

mkdir -p "$STATE" "$BIN_DIR"
tmp="$LAUNCHER.tmp.$$"
{
  printf '#!/bin/bash\n# Managed by KevANE install.sh\n'
  printf 'case "${1:-}" in\n'
  printf '  uninstall) shift; exec /bin/bash %q "$@" ;;\n' "$ROOT/scripts/uninstall.sh"
  printf '  *) KEVANE_PYTHON=%q KEVANE_MODEL_DIR=%q exec /bin/bash %q "$@" ;;\n' \
    "$PYTHON" "$MODEL" "$ROOT/scripts/kevane_service.sh"
  printf 'esac\n'
} > "$tmp"
chmod 755 "$tmp"
mv "$tmp" "$LAUNCHER"
if [[ -f "$OLD_LAUNCHER" ]] && grep -Fq '# Managed by KevANE install.sh' "$OLD_LAUNCHER"; then
  rm "$OLD_LAUNCHER"
fi
printf '%s\n' "$ROOT" > "$STATE/install-root"
printf '%s\n' "$PYTHON" > "$STATE/install-python"
printf '%s\n' "$MODEL" > "$STATE/install-model"
printf '%s\n' "$MODEL_OWNED" > "$STATE/model-owned"
if (( CREATED_ENV )); then
  printf '%s\n' "$ENV_NAME" > "$STATE/created-env"
fi

echo "Installed: $LAUNCHER"
echo "Model: $MODEL"
echo "The service is stopped until you run: $LAUNCHER start"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "To use 'kev-ane' without its full path, add this to ~/.zprofile:"
  echo '  export PATH="$HOME/.local/bin:$PATH"'
fi
