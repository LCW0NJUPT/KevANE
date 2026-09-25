<p align="center"><img src="assets/kevane-logo.png" alt="KevANE logo" width="320"></p>

# KevANE

[简体中文](README.zh-CN.md) · [Model files and card](https://huggingface.co/flylcw/KevANE-0.6B) · [License](LICENSE)

KevANE runs the **Kev 0.6B decision model** locally on Apple Silicon. It converts the Qwen3 backbone to Core ML and exposes a TypeSafe-compatible `POST /v1/systemone` API. It can answer choice, score, and yes/no questions for clients such as [Jev Jarvis](https://github.com/jev-chat/jev-chat-jarvis-mac). It does not generate reply text.

## Why Kev? Why ANE?

- **Kev** provides the System One decision format that Jarvis uses: a state, several questions, and probabilities over possible answers. KevANE keeps the encoder, API schema, and pointer head from a [fixed Kev revision](src/kev/UPSTREAM.txt).
- **ANE** is an available Core ML device on Apple Silicon. KevANE loads the backbone with `CPU_AND_NE`, excluding GPU for that model. Its execution plan prefers ANE for many reported operations, but that plan alone does not prove where every request runs. Tokenization and the pointer head use CPU.

## Install

Requirements: an Apple Silicon Mac, [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), and several GB of free storage for the Core ML package and its local compiled cache. Clone the source once, then run the installer:

```bash
git clone https://github.com/LCW0NJUPT/KevANE.git
cd KevANE
bash scripts/install.sh
```

The installer selects `kevane-runtime` explicitly, regardless of the currently active Conda environment. It creates that environment if needed, checks the model repository before downloading, and creates `~/.local/bin/kev-ane`. The server imports code from this checkout; the installer does not run `pip install -e`. It does **not** start the service. If `~/.local/bin` is not on your shell's `PATH`, add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zprofile` and open a new terminal. Until then, use `~/.local/bin/kev-ane` in the commands below.

The model files live in **`hf-model/` inside this checkout**. The installer checks for a complete local model first and skips the download when it is already there. Otherwise, it checks the [model repository](https://huggingface.co/flylcw/KevANE-0.6B) before downloading the required files into `hf-model/`. If you keep a converted model elsewhere, run `bash scripts/install.sh --model-dir /absolute/path/to/model` to use it without copying or downloading it.

The runtime environment is defined by **this repository**. The required Kev compatibility modules are included in `src/kev/` at a pinned revision; users do not need to clone or install Kev separately. The Core ML package, tokenizer, and pointer head come from the [separate model repository](https://huggingface.co/flylcw/KevANE-0.6B). Keep the cloned source directory in place while KevANE is installed; the service uses it for its code and model files.

## Start, stop, and uninstall

These commands work from any directory and do not require activating Conda:

```bash
kev-ane start
kev-ane status
kev-ane logs
kev-ane stop
kev-ane restart
kev-ane cache status
```

KevANE uses a per-user macOS LaunchAgent. `start` loads it on demand and returns immediately; it is not configured to start at login. Run `kev-ane start --wait` when you need the endpoint to be ready before continuing. `status` distinguishes loading from ready. The service listens on `127.0.0.1:8008`; `stop` waits for both the HTTP process and its Core ML worker to exit. Users do not run a conversion or compilation command: the downloaded model runs through Core ML. On the first load on a Mac, Core ML performs one-time device preparation that can briefly use a full CPU core. KevANE retains its result in ignored `build/compiled/`, so subsequent starts do not repeat that work. Run `kev-ane cache clear` only while stopped to remove that extra disk usage; the next start will require device preparation again. The job uses a lower CPU priority and limited CPU library threads to reduce interference with foreground apps.

Check the service:

```bash
curl --noproxy '*' http://127.0.0.1:8008/healthz
```

To remove the installed command and service, run:

```bash
kev-ane uninstall
```

Uninstall stops the service, removes the launcher, LaunchAgent state, and compiled cache, and removes the Conda environment only if the installer created it. It keeps the source checkout and the model by default. Use `kev-ane uninstall --remove-model` to remove model files **only when the installer downloaded them**; it never deletes a pre-existing or externally supplied model. Delete the checkout separately when you no longer need the source. To update an existing checkout, pull the new code, rerun `bash scripts/install.sh`, then run `kev-ane restart`. To run in the foreground instead, activate `kevane-runtime` and use `python scripts/30_systemone_server.py --model-dir hf-model --port 8008` from the checkout.

## Connect Jarvis

In Jarvis 0.6.0, set the TypeSafe/System One endpoint to `http://127.0.0.1:8008` and the model to `kevane-0.6b`. The values are also in [`integrations/jarvis/env.example`](integrations/jarvis/env.example). Run `kev-ane start --wait` before Jarvis's connection test. No patch to Jarvis is required.

KevANE handles **judgment and candidate ranking**. Jarvis uses its own separately configured provider for reply generation and its own code for the window, OCR, and text insertion. Jarvis 0.6.0's overlay monitors WeChat windows; switching to Codex hides that overlay by design. A working decision endpoint does not make Jarvis recognize Codex windows or run reply generation locally. Set Jarvis's TypeSafe API key to `local` even when KevANE has no authentication: Jarvis uses a non-empty key to select this path. If `KEVANE_API_KEY` is set for the server, use that value instead.

## How it runs

```text
System One request → Kev encoder → Core ML backbone → CPU pointer head → answers
                                (FP16, ≤512 tokens per pass)   (choice/score/yes-no)
```

The Core ML model runs in a separate process so an unexpected native Core ML exit does not immediately terminate the HTTP server. The merged PyTorch backbone is a conversion intermediate and is not needed at runtime. The model files are listed in the [Hugging Face model card](https://huggingface.co/flylcw/KevANE-0.6B).

## Verification and limits

- The shipped Core ML graph has a fixed 512-token input because the first conversion targeted short, low-latency decisions. This is a conversion choice, not Kev's full context limit. If the packed request exceeds 512 tokens, up to 16 independent questions are evaluated in separate passes, each with the shared state. A state plus any one question that exceeds 512 tokens still returns HTTP 422; nothing is silently truncated. Long chat histories can therefore lose useful context unless a larger model variant is converted and validated. Additional passes increase latency.
- In the recorded comparison with the original FP32 path, the minimum hidden-state cosine similarity was **0.999058** across seven samples; **34 of 35** decisions kept the same top answer. The one changed answer was a near tie. See the [parity record](benchmarks/results/coreml_fp16_parity.json).
- `CPU_AND_NE` excludes GPU for the Core ML backbone, but does not guarantee that every operation executes on ANE. The pointer head and request preparation still use CPU. Inspect placement with Instruments if hardware attribution matters.
- The service is for local use. Do not expose its unauthenticated default endpoint to a network.

## Rebuild from the source checkpoint

Rebuilding is optional and uses more memory and storage than inference. [`environment/build.yml`](environment/build.yml) contains the pinned PyTorch merge environment; [`environment/conversion.yml`](environment/conversion.yml) contains the separate Core ML conversion environment. Both use the Kev code shipped in this repository. The build downloads the original [Kev checkpoint](https://huggingface.co/jaredpalmer/kev-0.6b) and [Qwen base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) weights.

```bash
conda env create -f environment/build.yml
conda activate kevane-build
python -m pip install -e .
python scripts/10_merge_lora.py
conda deactivate

conda env create -f environment/conversion.yml
conda activate kevane-conversion
python -m pip install -e .
git clone https://github.com/anemll/anemll.git third_party/anemll
git -C third_party/anemll checkout f4ad26d061dc2e426faa19f81c367343bdeb9f0d
python -m pip install --no-deps -e third_party/anemll
python scripts/25_prepare_model_repo.py
python scripts/21_convert_anemll_backbone.py --seq-len 512
```

The merge writes to ignored `build/merged/`; conversion writes to `hf-model/`. ANEMLL is pinned for conversion and is **not** an inference dependency. To check a rebuilt package, activate `kevane-runtime` and run `python scripts/22_verify_coreml_hidden.py`.

## Source, model, and license

The source repository is [LCW0NJUPT/KevANE](https://github.com/LCW0NJUPT/KevANE); the model repository is [flylcw/KevANE-0.6B](https://huggingface.co/flylcw/KevANE-0.6B). KevANE and the converted model are Apache-2.0. Selected Kev modules are included at the revision recorded in [`src/kev/UPSTREAM.txt`](src/kev/UPSTREAM.txt); see [NOTICE](NOTICE) for attribution. KevANE is independent of Kev, Qwen, ANEMLL, and Jarvis.

This directory is the GitHub source repository. Source, scripts, Conda environments, and documentation can be committed directly from here. The large Core ML weights are published separately in the Hugging Face model repository and are excluded from Git by `.gitignore`. The local `hf-model/` directory holds those weights next to the model card and is the installer's default model path. Build intermediates and other local files are also excluded from Git.
