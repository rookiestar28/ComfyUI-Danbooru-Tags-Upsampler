import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple # PEP 585 for List and Tuple

# MODIFIED: Removed WebUI specific imports
# from modules.extra_networks import parse_prompt
# from modules.prompt_parser import parse_prompt_attention
# from modules.shared import opts

# MODIFIED: Removed parse_options from settings as it's WebUI specific
# from dart.settings import parse_options

# MODIFIED: utils will be imported from the local package
from .utils import (
    escape_webui_special_symbols, # Kept for now, review necessity
    unescape_webui_special_symbols, # Kept for now, review necessity
)

logger = logging.getLogger(__name__)

# Rating constants - these are fine and part of the core logic
DART_RATING_GENERAL = "rating:general"
DART_RATING_SENSITIVE = "rating:sensitive"
DART_RATING_QUESTIONABLE = "rating:questionable"
DART_RATING_EXPLICIT = "rating:explicit"

INPUT_RATING_GENERAL = DART_RATING_GENERAL
INPUT_RATING_SENSITIVE = DART_RATING_SENSITIVE
INPUT_RATING_QUESTIONABLE = DART_RATING_QUESTIONABLE
INPUT_RATING_EXPLICIT = DART_RATING_EXPLICIT

DART_RATING_SFW = "rating:sfw"
DART_RATING_NSFW = "rating:nsfw"

INPUT_RATING_SFW = "sfw"
INPUT_RATING_NSFW = "nsfw"

ALL_INPUT_RATING_TAGS = [
    INPUT_RATING_GENERAL,
    INPUT_RATING_SENSITIVE,
    INPUT_RATING_QUESTIONABLE,
    INPUT_RATING_EXPLICIT,
    INPUT_RATING_SFW,
    INPUT_RATING_NSFW,
]

RATING_TAG_PRIORITY = {
    INPUT_RATING_GENERAL: 0,
    INPUT_RATING_SENSITIVE: 1,
    INPUT_RATING_QUESTIONABLE: 2,
    INPUT_RATING_EXPLICIT: 3,
}

RATING_PARENT_TAG_PRIORITY = {INPUT_RATING_SFW: 0, INPUT_RATING_NSFW: 1}

DART_RATING_DEFAULT_PAIR = (DART_RATING_SFW, DART_RATING_GENERAL)


def get_rating_tag_pair(tag: str) -> Tuple[str, str]:
    if tag == INPUT_RATING_NSFW:
        return DART_RATING_NSFW, DART_RATING_EXPLICIT
    elif tag == INPUT_RATING_SFW:
        return DART_RATING_DEFAULT_PAIR
    elif tag == INPUT_RATING_GENERAL:
        return DART_RATING_DEFAULT_PAIR
    elif tag == INPUT_RATING_SENSITIVE:
        return DART_RATING_SFW, DART_RATING_SENSITIVE
    elif tag == INPUT_RATING_QUESTIONABLE:
        return DART_RATING_NSFW, DART_RATING_QUESTIONABLE
    elif tag == INPUT_RATING_EXPLICIT:
        return DART_RATING_NSFW, DART_RATING_EXPLICIT
    else:
        logger.warning(f"Unknown rating tag encountered: {tag}. Falling back to default.")
        return DART_RATING_DEFAULT_PAIR # Fallback for unknown tags


def get_strongest_rating_tag(tags: List[str]) -> str:
    strongest_tag = INPUT_RATING_GENERAL
    for tag in tags:
        if tag in RATING_TAG_PRIORITY and RATING_TAG_PRIORITY[tag] > RATING_TAG_PRIORITY.get(strongest_tag, -1):
            strongest_tag = tag
    return strongest_tag


def normalize_rating_tags(tags: List[str]) -> Tuple[str, str]:
    if not tags:
        return DART_RATING_DEFAULT_PAIR

    valid_tags = [tag for tag in tags if tag in ALL_INPUT_RATING_TAGS]
    if not valid_tags:
        logger.debug("No valid rating tags found in input, using default.")
        return DART_RATING_DEFAULT_PAIR
        
    tags = valid_tags # Use only valid tags for normalization

    if len(tags) == 1:
        return get_rating_tag_pair(tags[0])

    # len(tags) >= 2
    parent_tags_present = [tag for tag in tags if tag in RATING_PARENT_TAG_PRIORITY]
    child_tags_present = [tag for tag in tags if tag in RATING_TAG_PRIORITY]

    if all(tag in RATING_PARENT_TAG_PRIORITY for tag in tags): # e.g. ["sfw", "nsfw"]
        logger.warning(
            'Both "sfw" and "nsfw" parent rating tags are specified! Rating tag fell back to SFW default for upsampling.'
        )
        return DART_RATING_DEFAULT_PAIR
    
    if parent_tags_present and child_tags_present:
        # Determine dominant parent tag
        parent_tag = INPUT_RATING_SFW 
        for p_tag in parent_tags_present:
            if RATING_PARENT_TAG_PRIORITY[p_tag] > RATING_PARENT_TAG_PRIORITY[parent_tag]:
                parent_tag = p_tag
        
        # Determine strongest child tag among those present
        strongest_child_tag = get_strongest_rating_tag(child_tags_present)
        
        # Check for mismatch, e.g., "nsfw" parent with "rating:general" child
        expected_pair_for_parent = get_rating_tag_pair(parent_tag)
        if strongest_child_tag != expected_pair_for_parent[1] and parent_tag == DART_RATING_NSFW and strongest_child_tag in [INPUT_RATING_GENERAL, INPUT_RATING_SENSITIVE]:
             logger.warning(
                f'Specified child rating tag "{strongest_child_tag}" mismatches with dominant parent tag "{parent_tag}". '
                f'Using "{expected_pair_for_parent[1]}" (derived from parent) instead.'
            )
             return expected_pair_for_parent
        elif strongest_child_tag != expected_pair_for_parent[1] and parent_tag == DART_RATING_SFW and strongest_child_tag in [INPUT_RATING_QUESTIONABLE, INPUT_RATING_EXPLICIT]:
             logger.warning(
                f'Specified child rating tag "{strongest_child_tag}" mismatches with dominant parent tag "{parent_tag}". '
                f'Using "{expected_pair_for_parent[1]}" (derived from parent) instead.'
            )
             return expected_pair_for_parent

        return parent_tag, strongest_child_tag

    if child_tags_present: # Only child tags, no parent tags
        strongest_tag = get_strongest_rating_tag(child_tags_present)
        return get_rating_tag_pair(strongest_tag)

    # Only parent tags (but not all, handled above), or invalid mix
    # This case should ideally be covered, but as a fallback:
    if parent_tags_present:
        dominant_parent_tag = INPUT_RATING_SFW
        for p_tag in parent_tags_present:
            if RATING_PARENT_TAG_PRIORITY[p_tag] > RATING_PARENT_TAG_PRIORITY[dominant_parent_tag]:
                dominant_parent_tag = p_tag
        return get_rating_tag_pair(dominant_parent_tag)

    return DART_RATING_DEFAULT_PAIR # Final fallback


def load_tags_in_file(path: Path) -> List[str]:
    if not path.exists():
        logger.error(f"Tag file not found: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            tags = [tag.strip() for tag in file.readlines() if tag.strip()]
        return tags
    except Exception as e:
        logger.error(f"Error reading tag file {path}: {e}")
        return []


@dataclass
class ImagePromptAnalyzingResult:
    """A class of the result of analyzing tags"""
    rating_parent: str
    rating_child: str # Changed from str | None, ensure it always has a value
    copyright: str
    character: str
    general: str
    quality: str
    unknown: str


class DartAnalyzer:
    """A class for analyzing provided prompt and composing prompt for upsampling"""

    def __init__(
        self,
        # MODIFIED: extension_dir should be the path to the 'tags' directory or its parent
        tags_dir_path: Path, # Path to the directory containing copyright.txt etc.
        vocab: List[str],
        special_vocab: List[str],
        debug_logging: bool = False,
        # MODIFIED: escape_input_brackets and escape_output_brackets are now explicit params
        escape_input_brackets_enabled: bool = True, # Corresponds to original 'escape_input_brackets'
        escape_output_brackets_enabled: bool = True # Corresponds to original 'escape_output_brackets'
    ):
        if debug_logging:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # MODIFIED: Store these settings directly
        self.escape_input_brackets_enabled = escape_input_brackets_enabled
        self.escape_output_brackets_enabled = escape_output_brackets_enabled
        
        self.tags_dir = tags_dir_path
        logger.debug(f"DartAnalyzer using tags_dir: {self.tags_dir}")

        self.rating_tags = ALL_INPUT_RATING_TAGS # This list is constant

        self.copyright_tags = load_tags_in_file(self.tags_dir / "copyright.txt")
        self.character_tags = load_tags_in_file(self.tags_dir / "character.txt")
        self.quality_tags = load_tags_in_file(self.tags_dir / "quality.txt")
        
        logger.debug(f"Loaded {len(self.copyright_tags)} copyright tags, {len(self.character_tags)} character tags, {len(self.quality_tags)} quality tags.")

        self.vocab = list(vocab) # Take a copy
        self.special_vocab = list(special_vocab) # Take a copy

        if self.escape_input_brackets_enabled:
            logger.debug("Applying escaped brackets to loaded copyright, character tags, and vocab for matching.")
            # This logic assumes escape_webui_special_symbols works on a list of tags and returns a new list with escaped versions
            # Ensure escape_webui_special_symbols is robust.
            if self.copyright_tags: # Only escape if list is not empty
                self.copyright_tags.extend(escape_webui_special_symbols(self.copyright_tags))
            if self.character_tags:
                self.character_tags.extend(escape_webui_special_symbols(self.character_tags))
            # Modifying vocab like this can make it very large. Consider if this is truly necessary
            # or if matching should handle optional escaping.
            # For now, replicating original logic:
            if self.vocab:
                 self.vocab.extend(escape_webui_special_symbols(self.vocab))
            # Make them unique after extending
            self.copyright_tags = sorted(list(set(self.copyright_tags)))
            self.character_tags = sorted(list(set(self.character_tags)))
            self.vocab = sorted(list(set(self.vocab)))


    def split_tags(self, image_prompt: str) -> List[str]:
        """Splits a comma-separated string of tags into a list of stripped tags."""
        if not image_prompt:
            return []
        return [tag.strip() for tag in image_prompt.split(",") if tag.strip()]

    def extract_tags(self, input_tags: List[str], extract_tag_list: List[str]) -> Tuple[List[str], List[str]]:
        """Extracts tags that are present in extract_tag_list from input_tags."""
        matched: List[str] = []
        not_matched: List[str] = []
        
        # For efficient lookup if extract_tag_list is large
        extract_set = set(extract_tag_list)

        for input_tag in input_tags:
            if input_tag in extract_set:
                matched.append(input_tag)
            else:
                not_matched.append(input_tag)
        return matched, not_matched

    def preprocess_tags(self, tags: List[str]) -> str:
        """Preprocesses a list of tags into a comma-separated string, optionally unescaping."""
        if not tags:
            return ""
            
        processed_tags = list(tags) # Work on a copy

        # MODIFIED: Use the instance variable for the setting
        if self.escape_output_brackets_enabled: # This was originally 'escape_output_brackets'
            # This implies that tags are stored internally *escaped* if input escaping was on,
            # and this function *unescapes* them for the final output string if desired.
            # Or, it means tags going *into* the DART model should be unescaped.
            # The original comment was: "\(\) -> ()". This is unescaping.
            processed_tags = unescape_webui_special_symbols(processed_tags)
            
        return ", ".join(tag for tag in processed_tags if tag) # Ensure no empty strings from unescaping

    def analyze(self, image_prompt: str) -> ImagePromptAnalyzingResult:
        # MODIFIED: Simplified input tag parsing.
        # Assumes image_prompt is a comma-separated string of tags,
        # without WebUI's complex syntax like LoRAs or attention weights here.
        # These should be handled by upstream ComfyUI nodes.
        if not isinstance(image_prompt, str):
            logger.warning(f"DartAnalyzer.analyze received non-string input: {type(image_prompt)}. Attempting to cast to string.")
            image_prompt = str(image_prompt)

        input_tags_raw = self.split_tags(image_prompt)
        
        # Make unique. Original did this after WebUI parsing.
        input_tags = sorted(list(set(input_tags_raw)))
        logger.debug(f"Analyzing unique input tags: {input_tags}")

        # Classification logic remains similar
        rating_tags, remaining_tags = self.extract_tags(input_tags, self.rating_tags)
        copyright_tags, remaining_tags = self.extract_tags(remaining_tags, self.copyright_tags)
        character_tags, remaining_tags = self.extract_tags(remaining_tags, self.character_tags)
        quality_tags, remaining_tags = self.extract_tags(remaining_tags, self.quality_tags)
        
        # Extract special vocabulary (like <|length|>, <|input_end|>)
        # These generally shouldn't be part of the 'general' tags for upsampling.
        _special_tags_extracted, remaining_tags = self.extract_tags(remaining_tags, self.special_vocab)
        logger.debug(f"Extracted special vocab tags: {_special_tags_extracted}")

        # Remaining tags are categorized based on main vocab or marked as unknown
        # The main vocab might be very large after escape_input_brackets_enabled.
        general_tags, unknown_tags = self.extract_tags(remaining_tags, self.vocab)
        logger.debug(f"Rating tags: {rating_tags}")
        logger.debug(f"Copyright tags: {copyright_tags}")
        logger.debug(f"Character tags: {character_tags}")
        logger.debug(f"Quality tags: {quality_tags}")
        logger.debug(f"General tags (from vocab): {general_tags}")
        logger.debug(f"Unknown tags: {unknown_tags}")


        rating_parent, rating_child = normalize_rating_tags(rating_tags)
        logger.debug(f"Normalized rating: Parent='{rating_parent}', Child='{rating_child}'")

        return ImagePromptAnalyzingResult(
            rating_parent=rating_parent,
            rating_child=rating_child, # Ensure normalize_rating_tags always returns a non-None child or handle it
            copyright=self.preprocess_tags(copyright_tags),
            character=self.preprocess_tags(character_tags),
            general=self.preprocess_tags(general_tags), # Pass general_tags here
            quality=self.preprocess_tags(quality_tags),
            unknown=self.preprocess_tags(unknown_tags),
        )