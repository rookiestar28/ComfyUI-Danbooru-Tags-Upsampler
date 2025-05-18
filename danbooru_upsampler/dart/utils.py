import logging
import random
import re
from typing import List # PEP 585 for List

# MODIFIED: Removed WebUI specific type checking block for StableDiffusionProcessing
# from typing import TYPE_CHECKING, Any
# if TYPE_CHECKING:
# from modules.processing import (
# StableDiffusionProcessingTxt2Img,
# StableDiffusionProcessingImg2Img,
# )
# StableDiffusionProcessing = (
# StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img
# )
# else:
# StableDiffusionProcessing = Any

logger = logging.getLogger(__name__)

SEED_MIN = 0
SEED_MAX = 2**32 - 1 # Max value for a 32-bit unsigned integer, common for seeds

# Special symbols pattern, potentially for WebUI prompt syntax compatibility
# REVIEW: Necessity of this escaping/unescaping in ComfyUI context
SPECIAL_SYMBOL_PATTERN = re.compile(r"([()])")

# Escaped and unescaped symbols pair for unescaping processing
# REVIEW: Necessity of this escaping/unescaping in ComfyUI context
ESCAPED_SYMBOL_PATTERNS = {re.compile(r"\\\("): "(", re.compile(r"\\\)"): ")"}

# A pattern for escaping special symbols in regex construction
TAG_ESCAPE_SYMBOL_PATTERN = re.compile(r"[\\^${}\[\]()?+|*.]") # Added * and . to be more robust for re.escape like behavior when not using * wildcard

def get_random_seed() -> int:
    """Generates a random seed integer."""
    return random.randint(SEED_MIN, SEED_MAX)

# MODIFIED: Removed get_upmsapling_seeds as it's tightly coupled with SD WebUI's processing object.
# Seed handling in ComfyUI will be managed by node inputs.
# If batch processing within the node requires multiple seeds,
# they can be derived from a main seed input + index.

def escape_webui_special_symbols(tags: List[str]) -> List[str]:
    """
    Escapes parentheses in a list of tags. e.g., item -> \\item\\
    REVIEW: Evaluate if this specific escaping is needed or desirable for ComfyUI.
    It might interfere with ComfyUI's own prompt weighting or syntax.
    """
    if not all(isinstance(tag, str) for tag in tags):
        logger.warning("escape_webui_special_symbols received non-string elements in list.")
        # Optionally, convert to string or raise error. For now, skip non-strings.
        # return [SPECIAL_SYMBOL_PATTERN.sub(r"\\\1", str(tag)) for tag in tags]
        return [tag for tag in tags if isinstance(tag, str)] # Or handle error

    escaped_tags = [SPECIAL_SYMBOL_PATTERN.sub(r"\\\1", tag) for tag in tags]
    logger.debug(f"Escaped tags (sample): {escaped_tags[:5] if escaped_tags else '[]'}")
    return escaped_tags


def unescape_webui_special_symbols(tags: List[str]) -> List[str]:
    """
    Unescapes previously escaped parentheses in a list of tags. e.g., \\item\\ -> item
    REVIEW: Evaluate if this specific unescaping is needed or desirable for ComfyUI.
    """
    if not all(isinstance(tag, str) for tag in tags):
        logger.warning("unescape_webui_special_symbols received non-string elements in list.")
        return [tag for tag in tags if isinstance(tag, str)] # Or handle error

    unescaped_tags: List[str] = []
    for tag in tags:
        current_tag = tag
        for pattern, replace_to in ESCAPED_SYMBOL_PATTERNS.items():
            current_tag = pattern.sub(replace_to, current_tag)
        unescaped_tags.append(current_tag)
    logger.debug(f"Unescaped tags (sample): {unescaped_tags[:5] if unescaped_tags else '[]'}")
    return unescaped_tags


def _get_tag_pattern(tag: str) -> re.Pattern:
    """
    Returns a regex pattern of a tag.
    If tag contains '*', it's treated as a wildcard for '.*'.
    Other regex special characters are escaped.
    """
    if not isinstance(tag, str):
        logger.warning(f"_get_tag_pattern received non-string: {tag}. Returning a benign pattern.")
        return re.compile(re.escape(str(tag))) # Try to convert and escape

    if "*" in tag:
        # Replace wildcard '*' with '.*' and escape other regex characters
        pattern_str = "".join(
            TAG_ESCAPE_SYMBOL_PATTERN.sub(lambda m: "\\" + m.group(0), part)
            if i % 2 == 0 else ".*"
            for i, part in enumerate(tag.split("*"))
        )
        # Ensure pattern is valid if it starts/ends with * or has consecutive **
        # A simple split-join handles this fairly well. e.g. "*a*b*" -> ".*a.*b.*"
    else:
        # Escape all regex special characters if no wildcard
        pattern_str = re.escape(tag)
    
    try:
        return re.compile(pattern_str)
    except re.error as e:
        logger.error(f"Failed to compile regex for tag '{tag}' (pattern: '{pattern_str}'): {e}")
        # Fallback to a safe pattern (e.g., literal match of the original unescaped tag)
        return re.compile(re.escape(tag))


def get_patterns_from_tag_list(tags: List[str]) -> List[re.Pattern]:
    """Returns regex patterns from a list of tag strings."""
    if not tags:
        return []
    return [_get_tag_pattern(tag) for tag in tags if isinstance(tag, str)]


def get_valid_tag_list(tag_text: str) -> List[str]:
    """
    Returns a list of non-empty, stripped tags from a comma-separated tag text string.
    """
    if not isinstance(tag_text, str):
        logger.debug(f"get_valid_tag_list received non-string: {type(tag_text)}. Returning empty list.")
        return []
    if not tag_text.strip():
        return []
    return [tag.strip() for tag in tag_text.split(",") if tag.strip()]