import logging

import torch

from .dart.settings import MODEL_BACKEND_TYPE, DART_MODELS, DEFAULT_MODEL
from .dart.utils import SEED_MAX
from .service import (
    TAG_LENGTH_OPTIONS,
    DanbooruUpsamplerError,
    DanbooruUpsamplerRequest,
    upsample_prompt,
)

# Setup logger for this node
logger = logging.getLogger("Comfy.DanbooruTagsUpsamplerNode") # ComfyUI conventional logger name

class DanbooruTagsUpsamplerNode:
    DESCRIPTION = (
        "Expands a comma-separated anime prompt with generated Danbooru tags "
        "using a DART language model."
    )
    SEARCH_ALIASES = [
        "danbooru",
        "tags",
        "prompt upsampler",
        "anime tags",
        "dart",
        "Danbooru_Tags_Upsampler",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = list(DART_MODELS.keys())
        backend_choices = list(MODEL_BACKEND_TYPE.values())
        default_backend = MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", backend_choices[0] if backend_choices else "Original")

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "1girl, solo",
                    "tooltip": "Comma-separated Danbooru tags to analyze and expand.",
                }),
                "model_name": (model_choices, {
                    "default": DEFAULT_MODEL,
                    "tooltip": "Allowlisted DART model. Model files use a pinned revision and may download on first use.",
                }),
                "tag_length": (TAG_LENGTH_OPTIONS, {
                    "default": "long",
                    "tooltip": "Target amount of detail requested from the DART model.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": SEED_MAX,
                    "tooltip": "Generation seed. Reproducibility can still vary across backends, devices, and library versions.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Sampling randomness; higher values produce more varied tag choices.",
                }),
                "top_k": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Limits sampling to the highest-probability tokens; 0 disables this limit.",
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus-sampling probability mass; 1.0 keeps the full distribution.",
                }),
                "num_beams": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Beam-search width. Larger values use more time and memory.",
                }),
                "model_device": (["cpu", "cuda"], {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                    "tooltip": "Requested execution device. Failed CUDA initialization falls back to CPU and is logged.",
                }),
                "model_backend": (backend_choices, {
                    "default": default_backend,
                    "tooltip": "Original supports CFG. ONNX variants enforce ban tags but reject CFG; unavailable artifacts fall back with a warning.",
                }),
                "max_new_tokens": ("INT", {
                    "default": 128,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Maximum number of new tokens generated for the tag completion.",
                }),
            },
            "optional": {
                "negative_prompt_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Tags used for CFG with the Original backend. ONNX rejects CFG requests.",
                }),
                "ban_tags": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Comma-separated tags or supported wildcard patterns to block on every backend.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for negative tags; supported only by Original.",
                }),
                "debug_logging": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed node/runtime logs for troubleshooting.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upsampled_prompt",)
    OUTPUT_TOOLTIPS = ("The original prompt plus generated Danbooru tag completions.",)
    FUNCTION = "upsample"
    CATEGORY = "Prompt Styling/casual_gamer28"

    def __init__(self):
        logger.info("DanbooruTagsUpsamplerNode initialized.")

    def upsample(self, prompt: str, model_name: str, tag_length: str, seed: int,
                 temperature: float, top_k: int, top_p: float, num_beams: int,
                 model_device: str, model_backend: str, max_new_tokens: int,
                 negative_prompt_tags: str = "", ban_tags: str = "", cfg_scale: float = 1.5,
                 debug_logging: bool = False):

        logger.info(f"Upsampling started. Model: {model_name}, Seed: {seed}, Device: {model_device}, Backend: {model_backend}")
        request = DanbooruUpsamplerRequest(
            prompt=prompt,
            model_name=model_name,
            tag_length=tag_length,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            model_device=model_device,
            model_backend=model_backend,
            max_new_tokens=max_new_tokens,
            negative_prompt_tags=negative_prompt_tags,
            ban_tags=ban_tags,
            cfg_scale=cfg_scale,
            debug_logging=debug_logging,
        )
        try:
            result = upsample_prompt(request)
        except DanbooruUpsamplerError as exc:
            # IMPORTANT: fail the node explicitly instead of embedding runtime errors into the prompt text; host/tool integrations rely on clean prompt outputs.
            logger.error("Danbooru tag upsampling failed: %s", exc, exc_info=True)
            raise RuntimeError(str(exc)) from exc
        return (result.final_prompt,)

# Standard ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "DanbooruTagsUpsampler": DanbooruTagsUpsamplerNode,
    "DanbooruTagsUpsamplerNodeRay": DanbooruTagsUpsamplerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruTagsUpsampler": "Danbooru Tags Upsampler",
    "DanbooruTagsUpsamplerNodeRay": "Danbooru Tags Upsampler"
}
