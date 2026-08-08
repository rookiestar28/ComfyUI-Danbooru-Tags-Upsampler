# ComfyUI Danbooru Tags Upsampler

ComfyUI Danbooru Tags Upsampler is a Python custom node that expands a short, comma-separated prompt into a more detailed set of Danbooru tags using a DART language model. It is intended for anime-style image workflows where manually composing a long tag prompt would be slow or repetitive.

This project is a ComfyUI port and adaptation of [sd-danbooru-tags-upsampler](https://github.com/p1atdev/sd-danbooru-tags-upsampler), originally created by [p1atdev](https://github.com/p1atdev) for Stable Diffusion Web UI (AUTOMATIC1111).

## Table of Contents

- [What's New in 2.3.5](#whats-new-in-235)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Reference](#input-reference)
- [Models and Backends](#models-and-backends)
- [Compatibility and Verification](#compatibility-and-verification)
- [Model Downloads and Trust](#model-downloads-and-trust)
- [Host Integration API](#host-integration-api)
- [Troubleshooting](#troubleshooting)
- [Development and Validation](#development-and-validation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## What's New in 2.3.5

- **Truthful backend behavior:** Original, ONNX, and quantized ONNX capabilities are explicit. ONNX applies ban tags and rejects active CFG instead of silently ignoring it.
- **Strict request validation:** Numeric bounds and non-finite floats are rejected before model, tokenizer, or analyzer construction. Service callers receive typed error codes.
- **Runtime and cache correctness:** Model cache identity includes the immutable model revision, backend, artifact, requested device, and tokenizer trust policy. Result metadata reports the device actually used after fallback.
- **Safer analyzer lifecycle:** Required tag resources are cached as immutable data, invalidated when files change, and fail clearly when missing or unreadable.
- **Pinned model supply chain:** Every supported DART model uses an approved immutable revision. Remote tokenizer code is enabled only for the reviewed v1 model/revision pair.
- **Host UX alignment:** All 15 inputs now have tooltips, the display name is clearer, search aliases are richer, and both canonical and legacy workflow IDs remain supported.
- **Release hardening:** The legacy self-installer was removed. CI, publishing permissions, Registry metadata, and the release payload are validated for least privilege and privacy.

## Features

- Expands short prompts with generated Danbooru tags.
- Supports three allowlisted DART models and three runtime backend choices.
- Offers four relative output-length profiles: `very short`, `short`, `long`, and `very long`.
- Exposes sampling controls for seed, temperature, top-k, top-p, beam count, and token limit.
- Supports negative prompt CFG on the Original backend.
- Applies comma-separated ban tags and supported `*` patterns on every backend.
- Supports CPU and CUDA requests, with explicit CPU fallback when CUDA is unavailable or initialization fails.
- Preserves the original prompt and appends the generated suffix in the ComfyUI node output.
- Provides a structured Python service API for integrations that need resolved backend/device metadata and typed failures.

## Installation

[ComfyUI's official custom-node guide](https://docs.comfy.org/installation/install_custom_node) recommends ComfyUI Manager when available and requires dependencies to be installed into the same Python environment that runs ComfyUI.

### ComfyUI Manager

Install the node pack through ComfyUI Manager, then restart ComfyUI. Manager normally installs `requirements.txt` into the selected ComfyUI environment.

### Git Clone

Clone the repository into the active ComfyUI installation's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rookiestar28/ComfyUI-Danbooru-Tags-Upsampler.git
```

Install dependencies with the Python interpreter used by that same ComfyUI installation:

```bash
cd ComfyUI-Danbooru-Tags-Upsampler
python -m pip install -r requirements.txt
```

For Windows Portable, use its embedded interpreter instead of a global Python:

```powershell
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Danbooru-Tags-Upsampler\requirements.txt
```

The node-specific dependencies are:

- `transformers>=4.35.0`
- `optimum[onnxruntime]>=1.16.0`
- `tokenizers>=0.14.0`
- `sentencepiece`

`torch`, `torchvision`, and `torchaudio` are intentionally not installed or pinned by this node. They remain owned by the ComfyUI host so that its CPU/CUDA runtime is not replaced accidentally. The removed legacy `install.py` must not be restored or run.

The required `tags/copyright.txt`, `tags/character.txt`, and `tags/quality.txt` resources are included in the repository. Restart ComfyUI after installation and confirm there are no custom-node import errors.

## Quick Start

1. Add **Danbooru Tags Upsampler** by searching for its name, or browse to **Prompt Styling → casual_gamer28**.
2. Enter comma-separated tags in `prompt`, for example `1girl, solo`.
3. Keep `dart-v1-sft` for the recommended starting model.
4. Choose a backend and device. `ONNX (Quantized)` is the default backend; the device defaults to CUDA when available and CPU otherwise.
5. Adjust the output-length and sampling controls as needed.
6. Connect `upsampled_prompt` to `CLIPTextEncode` or another node that accepts a string.

The canonical workflow node ID is `DanbooruTagsUpsampler`. Existing workflows serialized with `DanbooruTagsUpsamplerNodeRay` continue to load through the legacy compatibility mapping.

## Input Reference

| Input | Default / range | Behavior |
| --- | --- | --- |
| `prompt` | `1girl, solo` | Comma-separated input tags. The ComfyUI node appends generated tags to this prompt. |
| `model_name` | `dart-v1-sft` | Selects one of the three allowlisted DART models below. |
| `tag_length` | `long` | Relative DART length profile: `very short`, `short`, `long`, or `very long`. It is not an exact tag-count guarantee. |
| `seed` | `0`; `0`–`4294967295` | Seeds generation. Exact reproducibility can still vary by backend, device, and dependency version. |
| `temperature` | `1.0`; `0.01`–`5.0` | Sampling randomness. Higher values generally increase variation. |
| `top_k` | `30`; `0`–`1000` | Restricts sampling to the highest-probability tokens. `0` disables the limit. |
| `top_p` | `1.0`; `0.0`–`1.0` | Nucleus-sampling probability mass. |
| `num_beams` | `1`; `1`–`20` | Beam-search width. Higher values cost more time and memory. |
| `model_device` | CUDA if available, otherwise CPU | Requests `cpu` or `cuda`. A failed/unavailable CUDA request falls back to CPU and is reported in service metadata/logs. |
| `model_backend` | `ONNX (Quantized)` | Requests `Original`, `ONNX`, or `ONNX (Quantized)`. Artifact availability may resolve the request to another backend as documented below. |
| `max_new_tokens` | `128`; `8`–`512` | Maximum number of generated tokens. |
| `negative_prompt_tags` | empty | Negative context used for CFG. Active CFG requires non-empty negative tags, `cfg_scale > 1.0`, and the Original backend. |
| `ban_tags` | empty | Comma-separated tags or supported wildcard patterns to block on every backend. |
| `cfg_scale` | `1.5`; `1.0`–`10.0` | CFG strength. ONNX rejects active CFG before heavy runtime construction. |
| `debug_logging` | `false` | Enables additional detailed runtime logging. Current standard runtime logs may already include a truncated generated-output preview; do not process sensitive prompts without controlling log access. |

All numeric values must be finite and within these bounds. Invalid values fail with `invalid_request` instead of reaching the model runtime.

## Models and Backends

### Models

| Model | Original | ONNX | Quantized ONNX | Remote tokenizer code |
| --- | --- | --- | --- | --- |
| [`dart-v1-sft`](https://huggingface.co/p1atdev/dart-v1-sft) | Yes | Yes | Yes | Reviewed and enabled only at the pinned revision |
| [`dart-v2-sft`](https://huggingface.co/p1atdev/dart-v2-sft) | Yes | Falls back to quantized ONNX | Yes | Disabled |
| [`dart-v2-moe-sft`](https://huggingface.co/p1atdev/dart-v2-moe-sft) | Yes | Falls back to Original | Falls back to Original | Disabled |

### Backend capabilities

| Backend | Active CFG | Ban tags | Artifact and fallback behavior |
| --- | --- | --- | --- |
| `Original` | Supported | Supported | Loads the Transformers model artifact. |
| `ONNX` | Rejected | Supported | Uses `model.onnx`; if unavailable, tries `model_quantized.onnx`, then Original. |
| `ONNX (Quantized)` | Rejected | Supported | Uses `model_quantized.onnx`; if unavailable, falls back to Original. |

ONNX inference is implemented with Hugging Face Optimum's [`ORTModelForCausalLM`](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/package_reference/modeling_ort). A fallback is returned as a warning through the service result; integrations should inspect `requested_backend`, `resolved_backend`, `resolved_device`, `onnx_file_name`, and `warnings` rather than assuming the request was used unchanged.

## Compatibility and Verification

Package metadata declares:

- Python `>=3.10`
- ComfyUI `>=0.22.3`
- V1 custom-node loading through `NODE_CLASS_MAPPINGS`, as defined by the [ComfyUI node lifecycle](https://docs.comfy.org/custom-nodes/backend/lifecycle)

The declared ComfyUI floor is a packaging compatibility boundary, not a claim that every historical host tuple received full live inference testing.

The following source and runtime baseline was verified on 2026-08-09:

| Surface | Reviewed tuple | Verification scope |
| --- | --- | --- |
| Current stable host | ComfyUI 0.31.0 with packaged frontend 1.48.7 | Live node discovery for both IDs; frozen input order/defaults; 15/15 tooltips; output/search metadata; pinned v1 Original and quantized-ONNX CPU inference |
| Standalone frontend schema | standalone frontend 1.50.3 | Backend metadata remains JSON serializable; no frontend bundle or V3-only entrypoint is shipped |
| Current Desktop stable channel | Comfy Desktop 1.0.37 → ComfyUI 0.31.0 → packaged frontend 1.48.7 | Source-reviewed, channel-resolved Desktop behavior plus the current-stable host validation above |

Desktop's stable channel is channel-resolved rather than a permanently frozen bundle, so a later Desktop installation may select a newer stable core. Treat the dated tuple above as validation evidence, not a permanent compatibility promise.

No live model download occurs in routine automated tests. Separately authorized manual validation downloaded only the pinned v1 Original and quantized-ONNX artifacts and completed both CPU inference paths successfully. CUDA, non-quantized live ONNX inference, and Python 3.14 are not claimed as runtime-validated configurations.

## Model Downloads and Trust

The selected model is downloaded from Hugging Face on first use and reused through the Hub cache. Hugging Face documents the default cache and the `HF_HOME` / `HF_HUB_CACHE` overrides in its [local cache guide](https://huggingface.co/docs/hub/local-cache).

Supply-chain boundaries:

- The selectable model names form a closed allowlist; callers cannot provide arbitrary repositories or revisions.
- Model, tokenizer, and ONNX loads receive the approved immutable revision for the selected model.
- `trust_remote_code=True` is limited to the reviewed `dart-v1-sft` tokenizer at its pinned revision.
- V2 and V2 MoE use standard tokenizer loading with remote code disabled.
- The node never installs packages at import time or runtime. This follows the [Comfy Registry security standard](https://docs.comfy.org/registry/standards), which prohibits subprocess-based runtime package installation.

The exact pinned revisions are maintained in `danbooru_upsampler/dart/settings.py` and covered by regression tests. Review model and dependency changes before updating those pins.

## Host Integration API

External Python integrations can reuse the runtime without parsing ComfyUI node output strings:

```python
from danbooru_upsampler.service import (
    DanbooruUpsamplerRequest,
    upsample_prompt,
)

result = upsample_prompt(
    DanbooruUpsamplerRequest(
        prompt="1girl, solo",
        model_backend="ONNX (Quantized)",
        model_device="cpu",
    )
)

print(result.final_prompt)
print(result.resolved_backend, result.resolved_device)
```

Available integration surfaces include:

- `upsample_prompt()` for structured requests and results.
- `build_toolbar_request()` for a conservative editor-toolbar profile.
- `resolve_backend_capabilities()` and `resolve_runtime_selection()` for preflight behavior.
- Frozen request/result/runtime dataclasses with resolved model revision, artifact, backend, device, and warnings.
- Typed errors with stable codes: `invalid_request`, `unsupported_feature`, `runtime_initialization_failed`, `analyzer_failed`, and `generation_failed`.

The runtime lock serializes shared model/tokenizer access. Analyzer resources are immutable and fingerprinted so repeated requests can reuse them safely while file changes invalidate the cache.

## Troubleshooting

### The node does not appear

- Confirm the repository is directly under the active ComfyUI `custom_nodes` path.
- Restart ComfyUI and inspect startup logs for an import error.
- Verify the dependencies were installed with the same Python interpreter that launches ComfyUI, not a system Python.
- Search for `Danbooru Tags Upsampler`, `danbooru`, or the legacy `Danbooru_Tags_Upsampler` wording.

### Dependency or PyTorch conflicts

Reinstall this node's requirements through ComfyUI Manager or the host environment. Do not install a separate PyTorch stack just for this node; use the version selected by ComfyUI/Desktop.

### CUDA falls back to CPU

The requested CUDA runtime was unavailable or failed initialization. Check the ComfyUI host's PyTorch/CUDA installation. The service result and logs report the resolved device.

### ONNX reports `unsupported_feature`

Active CFG is not supported on ONNX. Clear `negative_prompt_tags`, set `cfg_scale` to `1.0`, or use the Original backend. Ban tags remain supported on ONNX.

### Model download or cache problems

Confirm network access to the linked Hugging Face repositories and available disk space. If you override `HF_HOME` or `HF_HUB_CACHE`, ensure the ComfyUI process can read and write that location.

### Missing or unreadable tag resources

Restore `tags/copyright.txt`, `tags/character.txt`, and `tags/quality.txt` from the same repository revision. The analyzer intentionally fails instead of silently generating with incomplete classification data.

### Parentheses or square brackets behave unexpectedly

The port retains escape/unescape handling inherited from the original extension. Complex WebUI attention or LoRA syntax should be handled upstream; this node expects comma-separated Danbooru tags.

## Development and Validation

Use Python 3.10 or newer and install development tooling into a project-local environment. The deterministic repository gate is documented in `tests/TEST_SOP.md` and enforced in CI on Python 3.10 and 3.13.

```powershell
pre-commit run detect-secrets --all-files
pre-commit run --all-files --show-diff-on-failure
.venv\Scripts\python.exe -m compileall danbooru_upsampler __init__.py
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

This is a Python-only custom node with no `package.json`, browser bundle, or Playwright harness. Compile/import checks and the unit suite are the documented frontend-E2E replacement lane. Automated tests use fakes for model, ONNX, CUDA, and concurrency boundaries and must not download live models.

Release safeguards include:

- least-privilege, full-SHA-pinned CI and publishing workflows,
- main/manual-only Registry publishing with PR secret isolation,
- three-part semantic versioning in `pyproject.toml`,
- `.comfyignore` plus regression checks that keep tests, workflows, internal records, ignored paths, and secrets out of the Registry runtime archive.

The Comfy Registry uses semantic versions and immutable published versions; see the official [Registry overview](https://docs.comfy.org/registry/overview) and [publishing guide](https://docs.comfy.org/registry/publishing).

## Acknowledgements

All credit for the original concept, model training, and core generation logic goes to **p1atdev**. See the original project's [acknowledgements](https://github.com/p1atdev/sd-danbooru-tags-upsampler#acknowledgements) for the broader list of influential work.

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full text.
