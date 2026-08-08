# ComfyUI Danbooru Tags Upsampler

This is a custom node for ComfyUI that upsamples prompts by generating or completing Danbooru tags using a lightweight LLM. It's designed for users who want to quickly create diverse, natural, and detailed prompts for anime-style image generation without extensive manual input.

This project is a port and adaptation of the [sd-danbooru-tags-upsampler](https://github.com/p1atdev/sd-danbooru-tags-upsampler) extension originally developed by [p1atdev](https://github.com/p1atdev) for Stable Diffusion Web UI (AUTOMATIC1111). Many thanks to the original author for their excellent work!

## Current Status

- Supports the V1 ComfyUI custom-node loader through `NODE_CLASS_MAPPINGS`.
- Declares `requires-python = ">=3.10"` and `requires-comfyui = ">=0.22.3"` in `pyproject.toml`.
- Exposes frontend discovery metadata through `DESCRIPTION`, `SEARCH_ALIASES`, and `OUTPUT_TOOLTIPS`.
- Does not pin `torch`, `torchvision`, or `torchaudio`; those packages are managed by the ComfyUI host environment.
- Pins every supported DART model to an audited immutable revision; remote tokenizer code is enabled only for the reviewed v1 tokenizer.

## Features

- **Multiple Model Selection**: Choose from different DART model versions:
  - `dart-v1-sft` - V1 Stable (Recommended)
  - `dart-v2-sft` - V2 Improved
  - `dart-v2-moe-sft` - V2 MoE Architecture
- **Automated Tag Generation**: Leverages DART language models to expand your initial prompts with relevant Danbooru tags.
- **Customizable Output**: Control various aspects of tag generation, including:
  - Desired total tag length (very short, short, long, very long).
  - Generation parameters like temperature, top_k, top_p, and number of beams.
  - Banning specific tags from appearing in the upsampled results.
  - Seed for reproducible upsampling.
- **Classifier-Free Guidance (CFG) Support**: The Original backend supports negative prompt tags. ONNX backends reject CFG requests instead of silently ignoring them.
- **Multiple Model Backends**: Supports original Hugging Face Transformers, ONNX, and Quantized ONNX backends for the DART model, allowing for a balance between speed and resource usage.
- **Device Selection**: Request CPU or CUDA. If CUDA initialization fails, the runtime falls back to CPU and records/logs the actual device.
- **Smart Model Caching**: Models and tokenizers are cached by immutable revision, backend, artifact, device request, and tokenizer trust policy.
- **Host Integration Ready**: Exposes a structured Python service seam for external callers that need clean results, canonical node detection, and toolbar-friendly defaults.

## Installation

1. **Clone the Repository**:
    Navigate to your ComfyUI `custom_nodes` directory and clone this repository:

    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/rookiestar28/ComfyUI-Danbooru-Tags-Upsampler.git
    ```

2. **Install Dependencies**:
    ComfyUI Manager is the recommended installation path because it installs node dependencies into the selected ComfyUI environment. For a manual clone, activate that same environment and run:

    ```bash
    cd ComfyUI-Danbooru-Tags-Upsampler
    python -m pip install -r requirements.txt
    ```

    The `requirements.txt` installs the node-specific runtime dependencies:
    - `transformers`
    - `optimum[onnxruntime]`
    - `tokenizers`
    - `sentencepiece`

    PyTorch is intentionally not pinned by this node because ComfyUI and ComfyUI Desktop manage their own `torch`, `torchvision`, and `torchaudio` builds for the selected device.

3. **Download Tag Files (if not included or if path needs adjustment)**:
    This node relies on specific tag lists (e.g., `copyright.txt`, `character.txt`, `quality.txt`) for analyzing prompts. These files should be located in a `tags` directory within the `ComfyUI-Danbooru-Tags-Upsampler` custom node folder (i.e., `ComfyUI/custom_nodes/ComfyUI-Danbooru-Tags-Upsampler/tags/`).
    If you have cloned the repository, these files should already be in place.

4. **Start/Restart ComfyUI**:
    After installation, restart ComfyUI. The "Danbooru Tags Upsampler" node should appear under the "Prompt Styling/casual_gamer28" category.

## Compatibility Evidence

The following matrix records the source-backed validation baseline reviewed on 2026-08-09. It is an evidence boundary, not a promise that every future host revision is compatible.

| Host surface | Reviewed tuple | Evidence scope |
| --- | --- | --- |
| Current stable core | ComfyUI 0.31.0 with packaged frontend 1.48.7 | Package-style bootstrap, both node IDs, object-info-equivalent metadata, and network-free unit tests |
| Standalone frontend schema | standalone frontend 1.50.3 | V1 input/help/search metadata is JSON serializable; no frontend bundle or V3-only entrypoint is shipped |
| Current successor Desktop stable channel | Comfy Desktop 1.0.37 -> ComfyUI 0.31.0 -> packaged frontend 1.48.7 | The successor is channel-resolved: Desktop selects the current stable core, whose requirements select the packaged frontend |
| Historical Desktop floor | Desktop 0.9.4 -> ComfyUI 0.22.3 -> frontend 1.43.18 | Metadata floor only; this archived Desktop source is not the current successor |

The canonical node ID is `DanbooruTagsUpsampler`; the serialized legacy ID `DanbooruTagsUpsamplerNodeRay` remains mapped to the same implementation. No live model download, CUDA execution, or reference-repository execution was performed for this matrix. First use can download the selected pinned DART artifacts from Hugging Face.

### ComfyUI Desktop Notes

ComfyUI Desktop uses a managed Python environment and installs core packages with uv. During setup, Desktop asks for a ComfyUI files location, stored as `basePath` in Desktop's `config.json`; install this repository under that location's `custom_nodes` directory.

If the node-specific packages are missing, use Desktop or Manager's dependency reinstall flow instead of manually installing a separate PyTorch stack.

For Desktop or non-CUDA systems, select `cpu` as `model_device`. Select `cuda` only when the Desktop environment has a compatible NVIDIA PyTorch runtime.

## How to Use

1. In ComfyUI, right-click and select "Add Node" -> "Prompt Styling" -> "casual_gamer28" -> "Danbooru Tags Upsampler".
2. Connect a text input (your base prompt, e.g., "1girl, solo") to the `prompt` input of the node.
3. Adjust the parameters on the node as needed:

    - **`prompt`**: Your initial Danbooru tags or a simple description.
    - **`model_name`**: Select the DART model version to use:
        - `dart-v1-sft` - Stable version (Recommended, supports ONNX)
        - `dart-v2-sft` - Improved version (supports ONNX)
        - `dart-v2-moe-sft` - MoE architecture (Original backend only)
    - **`tag_length`**: Desired total length of the final prompt after upsampling.
        - `very short`: < 10 tags
        - `short`: < 20 tags
        - `long`: < 40 tags (recommended starting point)
        - `very long`: > 40 tags
    - **`seed`**: Seed for the tag generation process. The node accepts integer seeds from `0` through `4294967295`. Results can still vary across backends, devices, and dependency versions.
    - **`temperature`**: Controls randomness. Higher values (e.g., 1.5-2.0) mean more diverse/surprising tags; lower values (e.g., 0.7-1.0) mean more predictable/conservative tags.
    - **`top_k`**: Considers the k most likely tokens at each step.
    - **`top_p`**: Nucleus sampling; considers the smallest set of tokens whose cumulative probability exceeds p.
    - **`num_beams`**: Number of beams for beam search. `1` means no beam search. Higher values can lead to better quality but are slower.
    - **`model_device`**: Choose "cpu" or "cuda" for the DART model.
    - **`model_backend`**:
        - `Original`: Standard Hugging Face Transformers model.
        - `ONNX`: Optimized ONNX model (larger file size, potentially faster).
        - `ONNX (Quantized)`: Quantized ONNX model (smallest file size, often fastest, slight quality trade-off).
      If the selected model lacks the requested ONNX artifact, the service falls back to an available artifact/backend and records a warning. A failed CUDA initialization similarly falls back to CPU.
    - **`max_new_tokens`**: Maximum number of new tags to be generated by the LLM.
    - **`negative_prompt_tags` (Optional)**: Negative context for CFG on the Original backend. ONNX backends reject requests that activate CFG.
    - **`ban_tags` (Optional)**: Comma-separated tags (or supported patterns with `*`) excluded from generated tags on every backend. Example: `official alternate costume, english text, * background`
    - **`cfg_scale` (Optional)**: Classifier-Free Guidance scale for the Original backend. Values > 1.0 steer generation towards the main prompt and away from negative context.
    - **`debug_logging` (Optional)**: Check this to enable more detailed logging in the console, useful for troubleshooting.

4. The output `upsampled_prompt` can then be connected to a `CLIPTextEncode` node (or similar) for image generation.

## Showcase / Examples

The goal of this node is to enrich simple prompts. For example:

- **Input Prompt**: `1girl, solo, cowboy shot`
- **Upsampled Prompt (Example)**: `1girl, solo, cowboy shot, ahoge, animal ears, bare shoulders, blue hair, blush, closed mouth, collarbone, collared shirt, dress, eyelashes, fox ears, fox girl, fox tail, hair between eyes, heart, long hair, long sleeves, looking at viewer, neck ribbon, ribbon, shirt, simple background, sleeves past wrists, smile, tail, white background, white dress, white shirt, yellow eyes` (Actual output will vary based on seed and settings).

For more visual examples, please refer to the [original sd-danbooru-tags-upsampler showcase](https://github.com/p1atdev/sd-danbooru-tags-upsampler#showcases), as the core generation mechanism is the same.

## Model Access

This node supports multiple DART models from Hugging Face:

| Model | HuggingFace Link | ONNX Support |
|-------|------------------|--------|
| dart-v1-sft | [p1atdev/dart-v1-sft](https://huggingface.co/p1atdev/dart-v1-sft) | ✅ (Both) |
| dart-v2-sft | [p1atdev/dart-v2-sft](https://huggingface.co/p1atdev/dart-v2-sft) | ✅ (Quantized only) |
| dart-v2-moe-sft | [p1atdev/dart-v2-moe-sft](https://huggingface.co/p1atdev/dart-v2-moe-sft) | ❌ |

Models will be downloaded automatically on first use through Hugging Face Hub caching. The exact cache location depends on your operating system and Hugging Face environment variables such as `HF_HOME` or `HF_HUB_CACHE`.

## Host Integration

This repository now exposes a structured service layer for external callers that want to reuse the upsampler without depending on the full ComfyUI node wrapper.

- Programmatic entry point: `danbooru_upsampler.service.upsample_prompt`
- Toolbar helper: `danbooru_upsampler.service.build_toolbar_request`
- Canonical node registry key: `DanbooruTagsUpsampler`
- Legacy compatibility key remains available: `DanbooruTagsUpsamplerNodeRay`

The service path is intended for host integrations such as editor-toolbar actions:

- success returns a structured result object with `final_prompt`, `generated_suffix`, and resolved runtime metadata,
- invalid request/runtime/analyzer/generation failures raise typed exceptions, including malformed toolbar numeric inputs,
- runtime cache access is guarded for background-thread delegation,
- the default toolbar profile pins a conservative ONNX-quantized configuration rather than exposing the full node parameter surface immediately.

## For Developers / Troubleshooting

- **Tags Directory**: The analyzer component loads classification tags from the `tags/` directory within this custom node's folder. Ensure this directory and its contents (`copyright.txt`, `character.txt`, `quality.txt`) are present.
- **Dependencies**: This node depends on ComfyUI's host-managed PyTorch runtime. Do not install a separate `torch` build for this node unless you are intentionally repairing the host environment.
- **Python and ComfyUI Versions**: Package metadata declares Python `>=3.10` and ComfyUI `>=0.22.3`. ComfyUI Desktop's managed Python environment is expected to satisfy this in current Desktop baselines.
- **Escaping Brackets**: The handling of parentheses `()` and square brackets `[]` in prompts can be tricky. This node includes logic (from the original extension) to escape/unescape these, but their interaction with ComfyUI's CLIPTextEncode behavior should be observed. If you encounter issues with prompts containing brackets, this might be an area to investigate.
- **Error Behavior**: External hosts should prefer the structured service API instead of parsing node output strings. The node wrapper now fails explicitly when the service reports runtime errors.

## Acknowledgements

This work is a port and adaptation for ComfyUI. All credit for the original concept, model training, and core logic goes to **p1atdev**.
Please see the original repository for full acknowledgements to other influential projects:
[sd-danbooru-tags-upsampler Acknowledgements](https://github.com/p1atdev/sd-danbooru-tags-upsampler#acknowledgements)

## License

This repository is licensed under the Apache License 2.0. See the `LICENSE` file for the full license text.
