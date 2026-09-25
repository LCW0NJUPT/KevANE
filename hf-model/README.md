---
license: apache-2.0
base_model:
  - jaredpalmer/kev-0.6b
  - Qwen/Qwen3-0.6B-Base
tags:
  - coreml
  - apple-neural-engine
  - decision-model
  - typesafe
---

# KevANE-0.6B

A Core ML release of [Kev 0.6B](https://huggingface.co/jaredpalmer/kev-0.6b) for Apple Silicon. It answers TypeSafe System One choice, score, and yes/no questions through the [KevANE server](https://github.com/LCW0NJUPT/KevANE). **This is a decision model, not a chat or text-generation model.** It is not directly usable with `transformers.pipeline` or `AutoModelForCausalLM`.

## What is in this repository?

| File | Role |
| --- | --- |
| `kev-qwen3-0.6b-hidden.mlpackage/` | FP16 Core ML Qwen3 backbone; outputs final hidden states |
| `kev-qwen3-0.6b-pointer-head.pt` | Pointer head and temperature for the CPU decision step |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Tokenizer assets used by the Kev encoder |
| `kev-qwen3-0.6b-hidden-meta.json` | Conversion settings, shapes, and pinned source revisions |
| `LICENSE`, `NOTICE` | License and attribution |

The merged PyTorch backbone is a build intermediate and is not required for inference. The server and pinned Kev compatibility code are in the [GitHub repository](https://github.com/LCW0NJUPT/KevANE).

## Run locally

Requires an Apple Silicon Mac. Follow the [KevANE installation guide](https://github.com/LCW0NJUPT/KevANE#install):

```bash
git clone https://github.com/LCW0NJUPT/KevANE.git
cd KevANE
bash scripts/install.sh
~/.local/bin/kev-ane start --wait
```

Example request:

```bash
curl --noproxy '*' http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"kevane-0.6b","state":"The user says hello.","questions":{"intent":{"type":"choice","instructions":"What is the intent?","criteria":{"greeting":null,"request":null}}}}'
```

Use `~/.local/bin/kev-ane stop` when finished. No manual conversion or compilation step is needed. Core ML prepares the downloaded model for the local Mac on first load and may briefly use a full CPU core. KevANE keeps the result in the ignored source checkout's `build/compiled/` directory (roughly another 1.1 GB); subsequent starts reuse it. Run `kev-ane cache status` to inspect it or `kev-ane cache clear` while the service is stopped to remove it. The service binds to localhost by default. Jarvis users can set its TypeSafe/System One URL to `http://127.0.0.1:8008` and model name to `kevane-0.6b`, with a non-empty TypeSafe key such as `local`.

## Architecture and provenance

The [Kev 0.6B checkpoint](https://huggingface.co/jaredpalmer/kev-0.6b) is a LoRA adaptation of [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base). KevANE merges that adaptation, converts the 28-layer backbone to a Core ML ML Program using [ANEMLL](https://github.com/anemll/anemll), and retains Kev's pointer head on CPU. The Core ML graph is FP16, with no LUT quantization and a fixed 512-token packed context. Conversion inputs are token IDs, position IDs, an additive branch mask, and Core ML state; the output is final hidden states.

Pinned revisions and exact input shapes are in `kev-qwen3-0.6b-hidden-meta.json`. The KevANE runtime includes selected Kev source modules at the pinned revision and does not require a separate Kev installation. The server defaults to `CPU_AND_NE`, which excludes GPU for the Core ML backbone but does not guarantee ANE placement for every operation. Tokenization and the pointer head use CPU.

## Verification and limitations

The [recorded FP16/FP32 comparison](https://github.com/LCW0NJUPT/KevANE/blob/main/benchmarks/results/coreml_fp16_parity.json) reports minimum hidden-state cosine similarity **0.999058** across seven samples and matching top answers for **34/35** decisions. The changed answer was a near tie. These are conversion checks on a small fixture set, not a general accuracy benchmark.

The Core ML graph accepts 512 tokens per pass. KevANE can evaluate up to 16 independent questions in separate passes when a packed request exceeds 512 tokens, but shared state plus any single question must still fit or the server returns HTTP 422. No input is silently truncated. This short context can limit decisions based on long chat histories, and multiple passes add latency. Close decisions may change under FP16. The model only scores options supplied in a System One request; it does not compose answers or verify the truth of its input. For device-level ANE attribution, profile the running app with Instruments rather than relying only on the `CPU_AND_NE` setting.

## License

Apache-2.0. The converted weights derive from Kev and Qwen3-0.6B-Base, both released under Apache-2.0. See `NOTICE` for attribution. This repository is an independent conversion and is not an official release of Kev, Qwen, ANEMLL, or Jarvis.
