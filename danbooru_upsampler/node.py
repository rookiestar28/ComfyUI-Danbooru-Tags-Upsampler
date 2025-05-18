import logging
from pathlib import Path
import torch # For torch.device if needed, and seed setting context
from transformers import set_seed # To set seed for generation

# Import core classes from the dart sub-package
from .dart.generator import DartGenerator
from .dart.analyzer import DartAnalyzer, ImagePromptAnalyzingResult

# Import constants that might be used for INPUT_TYPES
from .dart.settings import MODEL_BACKEND_TYPE # For choices in model_backend
from .dart.utils import SEED_MAX # For seed input max value

# Setup logger for this node
logger = logging.getLogger("Comfy.DanbooruTagsUpsamplerNode") # ComfyUI conventional logger name

# Define constants for combo boxes, similar to original TOTAL_TAG_LENGTH
TAG_LENGTH_OPTIONS = ["very short", "short", "long", "very long"]
TAG_LENGTH_MAP = {
    "very short": "<|very_short|>",
    "short": "<|short|>",
    "long": "<|long|>",
    "very long": "<|very_long|>",
}

class DanbooruTagsUpsamplerNode:
    # Cached instances of generator and analyzer to avoid re-initialization
    # if settings that affect their init (like model path, tags_dir) don't change.
    # However, model loading itself is handled by DartGenerator's class-level cache.
    # For simplicity and flexibility with device/backend changes at node level,
    # we will re-initialize them in upsample() but rely on DartGenerator's internal model caching.
    
    # Store the path to the 'tags' directory relative to this node file
    # Assumes 'tags' directory is at ComfyUI-Danbooru-Tags-Upsampler/tags/
    # and this node.py is at ComfyUI-Danbooru-Tags-Upsampler/danbooru_upsampler/nodes.py
    # So, Path(__file__).parent = danbooru_upsampler
    # Path(__file__).parent.parent = ComfyUI-Danbooru-Tags-Upsampler
    TAGS_DIR = Path(__file__).parent.parent / "tags"

    @classmethod
    def INPUT_TYPES(cls):
        # Ensure MODEL_BACKEND_TYPE values are used for choices
        backend_choices = list(MODEL_BACKEND_TYPE.values())
        # Ensure the default from settings.py (if used) is a valid choice, or pick first
        default_backend = MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", backend_choices[0] if backend_choices else "Original")


        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "1girl, solo"}),
                "tag_length": (TAG_LENGTH_OPTIONS, {"default": "long"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}), # Original default was 20
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "model_device": (["cpu", "cuda"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "model_backend": (backend_choices, {"default": default_backend}),
                "max_new_tokens": ("INT", {"default": 128, "min": 8, "max": 512, "step": 8}),
            },
            "optional": {
                "negative_prompt_tags": ("STRING", {"multiline": True, "default": ""}), # For CFG related tags
                "ban_tags": ("STRING", {"multiline": False, "default": ""}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "debug_logging": ("BOOLEAN", {"default": False}),
                # escape_input_brackets_enabled / escape_output_brackets_enabled are more internal to how analyzer
                # and utils work, might be advanced options or fixed based on testing.
                # For now, let's assume fixed True or make them advanced optional inputs if needed.
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upsampled_prompt",)
    FUNCTION = "upsample"
    CATEGORY = "Prompt Styling/Ray" # Or your preferred category

    def __init__(self):
        # __init__ is called when the workflow is loaded.
        # We don't initialize generator/analyzer here to allow device/backend changes from UI
        # without reloading the workflow. DartGenerator handles its own model caching.
        logger.info("DanbooruTagsUpsamplerNode initialized.")
        if not DanbooruTagsUpsamplerNode.TAGS_DIR.exists():
            logger.warning(f"Tags directory not found at: {DanbooruTagsUpsamplerNode.TAGS_DIR}. Analyzer might not load category tags.")


    def upsample(self, prompt: str, tag_length: str, seed: int,
                 temperature: float, top_k: int, top_p: float, num_beams: int,
                 model_device: str, model_backend: str, max_new_tokens: int,
                 negative_prompt_tags: str = "", ban_tags: str = "", cfg_scale: float = 1.5,
                 debug_logging: bool = False):

        logger.info(f"Upsampling started. Seed: {seed}, Device: {model_device}, Backend: {model_backend}")
        if debug_logging:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO) # Reset to INFO if not debug, for subsequent runs

        # --- 1. Initialize DartGenerator ---
        # Model name and tokenizer name are fixed for p1atdev/dart-v1-sft
        model_name = "p1atdev/dart-v1-sft"
        tokenizer_name = "p1atdev/dart-v1-sft"
        
        try:
            generator = DartGenerator(
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                model_backend=model_backend,
                model_device=model_device,
                debug_logging=debug_logging
            )
            # --- 2. Ensure model and tokenizer are loaded ---
            generator.load_model_if_needed() # This will use DartGenerator's class-level cache
            generator.load_tokenizer_if_needed()
        except Exception as e:
            logger.error(f"Error initializing DartGenerator or loading model/tokenizer: {e}")
            return (prompt + f" [Error: Failed to init generator: {e}]",)

        # --- 3. Initialize DartAnalyzer ---
        # escape_input_brackets_enabled and escape_output_brackets_enabled
        # For now, let's use defaults or make them advanced options if necessary.
        # These depend on how utils.escape_webui_special_symbols is handled.
        # Assuming True for now based on original defaults.
        escape_input_brackets = True 
        escape_output_brackets = True # This is for unescaping in analyzer.preprocess_tags

        try:
            analyzer = DartAnalyzer(
                tags_dir_path=DanbooruTagsUpsamplerNode.TAGS_DIR,
                vocab=generator.get_vocab_list(),
                special_vocab=generator.get_special_vocab_list(),
                debug_logging=debug_logging,
                escape_input_brackets_enabled=escape_input_brackets,
                escape_output_brackets_enabled=escape_output_brackets
            )
        except Exception as e:
            logger.error(f"Error initializing DartAnalyzer: {e}")
            return (prompt + f" [Error: Failed to init analyzer: {e}]",)

        # --- 4. Analyze input prompt ---
        logger.debug(f"Analyzing prompt: '{prompt}'")
        analyzed_result: ImagePromptAnalyzingResult = analyzer.analyze(prompt)
        
        # Analyze negative prompt tags if provided (for CFG context)
        negative_analyzed_result: ImagePromptAnalyzingResult | None = None
        if negative_prompt_tags and negative_prompt_tags.strip() != "" and cfg_scale > 1.0:
            logger.debug(f"Analyzing negative_prompt_tags: '{negative_prompt_tags}'")
            negative_analyzed_result = analyzer.analyze(negative_prompt_tags)

        # --- 5. Compose LLM input prompt ---
        # Get the special length tag based on user's choice
        length_special_token = TAG_LENGTH_MAP.get(tag_length, TAG_LENGTH_MAP["long"])

        llm_prompt = generator.compose_prompt(
            rating=f"{analyzed_result.rating_parent}, {analyzed_result.rating_child}",
            copyright=analyzed_result.copyright,
            character=analyzed_result.character,
            general=analyzed_result.general, # General tags from the *positive* prompt
            length=length_special_token
        )
        logger.debug(f"Composed LLM prompt: {llm_prompt}")

        # --- 6. Prepare CFG negative LLM input (if applicable) ---
        cfg_negative_llm_prompt: str | None = None
        if negative_analyzed_result:
            # For CFG, the unconditional prompt often shares structure (rating, char, copyright)
            # with the conditional one, but uses general tags from the negative_prompt_tags.
            # The original extension combined positive char/copyright with negative general.
            
            # Helper to join, avoiding double commas if one part is empty
            def _join_texts(text1: str, text2: str) -> str:
                t1 = text1.strip()
                t2 = text2.strip()
                if t1 and t2: return f"{t1}, {t2}"
                return t1 or t2

            cfg_negative_llm_prompt = generator.compose_prompt(
                rating=f"{analyzed_result.rating_parent}, {analyzed_result.rating_child}", # Usually use positive's rating
                copyright=_join_texts(analyzed_result.copyright, negative_analyzed_result.copyright),
                character=_join_texts(analyzed_result.character, negative_analyzed_result.character),
                general=negative_analyzed_result.general, # General tags from the *negative* prompt
                length=length_special_token # Same length token
            )
            logger.debug(f"Composed CFG negative LLM prompt: {cfg_negative_llm_prompt}")


        # --- 7. Get banned word IDs ---
        banned_ids = generator.get_bad_words_ids(ban_tags)
        if banned_ids:
            logger.debug(f"Banned word IDs prepared for {len(banned_ids)} patterns.")

        # --- 8. Set seed ---
        # Note: transformers.set_seed sets seed for python, numpy, and torch
        # For full reproducibility in distributed settings or complex scenarios, more might be needed,
        # but for a single model generation, this is generally sufficient.
        actual_seed = int(seed) % (SEED_MAX + 1) # Ensure seed is within valid range
        set_seed(actual_seed)
        logger.debug(f"Seed set to: {actual_seed}")

        # --- 9. Generate upsampled tags ---
        try:
            logger.info("Starting tag generation...")
            upsampled_tags_str = generator.generate(
                prompt=llm_prompt,
                max_new_tokens=max_new_tokens,
                min_new_tokens=0, # As per original generator.py
                do_sample=True, # Typically True for creative generation
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                bad_words_ids=banned_ids,
                negative_prompt=cfg_negative_llm_prompt if cfg_scale > 1.0 else None, # Pass CFG prompt
                cfg_scale=cfg_scale if cfg_negative_llm_prompt else 1.0 # Pass CFG scale
            )
            logger.info(f"Tag generation complete. Upsampled tags: '{upsampled_tags_str[:100]}...'")
        except Exception as e:
            logger.error(f"Error during tag generation: {e}", exc_info=True)
            return (prompt + f" [Error: Generation failed: {e}]",)

        # --- 10. Combine and return ---
        if upsampled_tags_str and upsampled_tags_str.strip():
            final_prompt = f"{prompt.strip()}, {upsampled_tags_str.strip()}"
        else:
            final_prompt = prompt.strip()
        
        logger.debug(f"Final prompt: '{final_prompt[:200]}...'")
        return (final_prompt,)

# Standard ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "DanbooruTagsUpsamplerNodeRay": DanbooruTagsUpsamplerNode # Renamed for uniqueness
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruTagsUpsamplerNodeRay": "Danbooru_Tags_Upsampler"
}