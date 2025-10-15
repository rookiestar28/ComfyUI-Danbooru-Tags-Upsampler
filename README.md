# ComfyUI Danbooru Tags Upsampler

This is a custom node for ComfyUI that upsamples prompts by generating or completing Danbooru tags using a lightweight LLM. It's designed for users who want to quickly create diverse, natural, and detailed prompts for anime-style image generation without extensive manual input.

This project is a port and adaptation of the [sd-danbooru-tags-upsampler](https://github.com/p1atdev/sd-danbooru-tags-upsampler) extension originally developed by [p1atdev](https://github.com/p1atdev) for Stable Diffusion Web UI (AUTOMATIC1111). Many thanks to the original author for their excellent work!

## Features

* **Automated Tag Generation**: Leverages the `p1atdev/dart-v1-sft` language model to expand your initial prompts with relevant Danbooru tags.
* **Customizable Output**: Control various aspects of tag generation, including:
    * Desired total tag length (very short, short, long, very long).
    * Generation parameters like temperature, top_k, top_p, and number of beams.
    * Banning specific tags from appearing in the upsampled results.
    * Seed for reproducible upsampling.
* **Classifier-Free Guidance (CFG) Support**: Optionally provide negative prompt tags to guide the generation process further.
* **Multiple Model Backends**: Supports original Hugging Face Transformers, ONNX, and Quantized ONNX backends for the DART model, allowing for a balance between speed and resource usage.
* **Device Selection**: Run the upsampling model on either CPU or CUDA-enabled GPU.

## Installation

1.  **Clone the Repository**:
    Navigate to your ComfyUI `custom_nodes` directory and clone this repository:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone [https://github.com/rookiestar28/ComfyUI-Danbooru-Tags-Upsampler.git](https://github.com/rookiestar28/ComfyUI-Danbooru-Tags-Upsampler.git)
    ```

2.  **Install Dependencies**:
    Navigate into the cloned directory and install the required Python packages.
    * **Method 1: Using `install.py` (if provided and configured)**
        If an `install.py` script is present in the root of this custom node's directory, ComfyUI might attempt to run it automatically on startup. Alternatively, you might need to run it manually (ensure your ComfyUI's Python environment is active):
        ```bash
        cd ComfyUI-Danbooru-Tags-Upsampler
        python install.py 
        ```
        (Note: The `install.py` script should ideally install dependencies from `requirements.txt`.)
    * **Method 2: Manual Installation via pip**
        If `install.py` is not available or you prefer manual control, activate your ComfyUI's Python environment and run:
        ```bash
        cd ComfyUI-Danbooru-Tags-Upsampler
        pip install -r requirements.txt
        ```
    The `requirements.txt` should include:
    * `torch`
    * `transformers`
    * `optimum[onnxruntime]` 
    * (and any other specific versions or dependencies identified during development)

3.  **Download Tag Files (if not included or if path needs adjustment)**:
    This node relies on specific tag lists (e.g., `copyright.txt`, `character.txt`, `quality.txt`) for analyzing prompts. These files should be located in a `tags` directory within the `ComfyUI-Danbooru-Tags-Upsampler` custom node folder (i.e., `ComfyUI/custom_nodes/ComfyUI-Danbooru-Tags-Upsampler/tags/`).
    If you have cloned the repository, these files should already be in place.

4.  **Start/Restart ComfyUI**:
    After installation, restart ComfyUI. The "Danbooru Tags Upsampler" node should appear under the "Prompt Styling" category (or whichever category you set in `nodes.py`).

## How to Use

1.  In ComfyUI, right-click and select "Add Node" -> "Prompt Styling" -> "Danbooru Tags Upsampler".
2.  Connect a text input (your base prompt, e.g., "1girl, solo") to the `prompt` input of the node.
3.  Adjust the parameters on the node as needed:

    * **`prompt`**: Your initial Danbooru tags or a simple description.
    * **`tag_length`**: Desired total length of the final prompt after upsampling.
        * `very short`: < 10 tags
        * `short`: < 20 tags
        * `long`: < 40 tags (recommended starting point)
        * `very long`: > 40 tags
    * **`seed`**: Seed for the tag generation process. Use `-1` for a random seed (though the node currently expects a positive integer; random seed logic might need to be implemented if desired similarly to the original). A fixed seed with the same input prompt will produce the same upsampled tags.
    * **`temperature`**: Controls randomness. Higher values (e.g., 1.5-2.0) mean more diverse/surprising tags; lower values (e.g., 0.7-1.0) mean more predictable/conservative tags.
    * **`top_k`**: Considers the k most likely tokens at each step.
    * **`top_p`**: Nucleus sampling; considers the smallest set of tokens whose cumulative probability exceeds p.
    * **`num_beams`**: Number of beams for beam search. `1` means no beam search. Higher values can lead to better quality but are slower.
    * **`model_device`**: Choose "cpu" or "cuda" for the DART model.
    * **`model_backend`**:
        * `Original`: Standard Hugging Face Transformers model.
        * `ONNX`: Optimized ONNX model (larger file size, potentially faster).
        * `ONNX (Quantized)`: Quantized ONNX model (smallest file size, often fastest, slight quality trade-off).
    * **`max_new_tokens`**: Maximum number of new tags to be generated by the LLM.
    * **`negative_prompt_tags` (Optional)**: Provide tags here that you want the upsampler to consider as "negative" context if using CFG. This helps guide what *not* to emphasize or include from the LLM's general knowledge.
    * **`ban_tags` (Optional)**: Comma-separated list of tags (or patterns with `*`) that should be explicitly excluded from the generated upsampled tags. Example: `official alternate costume, english text, * background`
    * **`cfg_scale` (Optional)**: Classifier-Free Guidance scale. Only active if `negative_prompt_tags` are provided. Values > 1.0 steer generation towards the main prompt and away from the negative context.
    * **`debug_logging` (Optional)**: Check this to enable more detailed logging in the console, useful for troubleshooting.

4.  The output `upsampled_prompt` can then be connected to a `CLIPTextEncode` node (or similar) for image generation.

*(Consider adding a simple workflow image here if possible, showing [Primitive String Node] -> [Danbooru Tags Upsampler] -> [CLIPTextEncode])*

## Showcase / Examples

*(This section can be adapted from the original README's "Showcases" but should ideally use images generated via the ComfyUI node if possible. For now, you can state that results are similar to the original extension and link to its showcase, or re-use its examples if the generation parameters are comparable.)*

The goal of this node is to enrich simple prompts. For example:

* **Input Prompt**: `1girl, solo, cowboy shot`
* **Upsampled Prompt (Example)**: `1girl, solo, cowboy shot, ahoge, animal ears, bare shoulders, blue hair, blush, closed mouth, collarbone, collared shirt, dress, eyelashes, fox ears, fox girl, fox tail, hair between eyes, heart, long hair, long sleeves, looking at viewer, neck ribbon, ribbon, shirt, simple background, sleeves past wrists, smile, tail, white background, white dress, white shirt, yellow eyes` (Actual output will vary based on seed and settings).

For more visual examples, please refer to the [original sd-danbooru-tags-upsampler showcase](https://github.com/p1atdev/sd-danbooru-tags-upsampler#showcases), as the core generation mechanism is the same.

## Model Access

This node uses the `p1atdev/dart-v1-sft` model from Hugging Face:
* [p1atdev/dart-v1-sft on HuggingFace](https://huggingface.co/p1atdev/dart-v1-sft)

The model will be downloaded automatically on first use if not found in your Hugging Face cache.

## For Developers / Troubleshooting

* **Tags Directory**: The analyzer component loads classification tags from the `tags/` directory within this custom node's folder. Ensure this directory and its contents (`copyright.txt`, `character.txt`, `quality.txt`) are present.
* **Escaping Brackets**: The handling of parentheses `()` and square brackets `[]` in prompts can be tricky. This node includes logic (from the original extension) to escape/unescape these, but their interaction with ComfyUI's CLIPTextEncode behavior should be observed. If you encounter issues with prompts containing brackets, this might be an area to investigate.

## Acknowledgements

This work is a port and adaptation for ComfyUI. All credit for the original concept, model training, and core logic goes to **p1atdev**.
Please see the original repository for full acknowledgements to other influential projects:
[sd-danbooru-tags-upsampler Acknowledgements](https://github.com/p1atdev/sd-danbooru-tags-upsampler#acknowledgements)

## License

This ComfyUI custom node is provided under the same license as the original project, if applicable, or defaults to [MIT License/Apache 2.0/etc. - **YOU NEED TO CHOOSE OR CONFIRM THIS** based on the original `LICENSE` file and your intentions]. Please check the `LICENSE` file in this repository.
