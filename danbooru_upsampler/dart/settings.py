import logging
from typing import Literal, Any # Literal 和 Any 可能仍然有用，如果 DEFAULT_VALUES 被保留並在別處引用

logger = logging.getLogger(__name__)

# Constants that might still be useful for generator.py or other parts
MODEL_BACKEND_TYPE: dict[str, str] = {
    "ORIGINAL": "Original",
    "ONNX": "ONNX",
    "ONNX_QUANTIZED": "ONNX (Quantized)",
}

# OPTION_NAME and DEFAULT_VALUES might be useful if you want to centralize
# default configurations that your ComfyUI node might use or expose.
# However, many of these will be directly set as defaults in the ComfyUI node's INPUT_TYPES.

# If you decide these defaults are better managed directly in the ComfyUI node class,
# you can remove OPTION_NAME and DEFAULT_VALUES entirely from this file.
# For now, I'll keep them as they might provide a reference.

OPTION_NAME = Literal[
    "model_name",
    "tokenizer_name",
    "model_backend_type",
    "model_device",
    "debug_logging",
    "escape_input_brackets", # This might become a boolean input for a utility function if kept
    "escape_output_brackets",# This might become a boolean input for a utility function if kept
]

# These default values can serve as a reference when defining
# the ComfyUI node's input fields and their default settings.
# The ComfyUI node will be the primary source for these parameter values during execution.
DEFAULT_VALUES: dict[OPTION_NAME, Any] = {
    "model_name": "p1atdev/dart-v1-sft",
    "tokenizer_name": "p1atdev/dart-v1-sft",
    "model_backend_type": MODEL_BACKEND_TYPE["ONNX_QUANTIZED"], # Default backend
    "model_device": "cpu", # Default device
    "escape_input_brackets": True, # Might not be relevant as a global setting anymore
    "escape_output_brackets": True, # Might not be relevant as a global setting anymore; could be specific to escape_webui_special_symbols
    "debug_logging": False, # This will be passed to DartGenerator init
}

# All functions related to WebUI (parse_options, on_ui_settings) are removed.

if __name__ == '__main__':
    # You can add some simple tests or print statements here if you want
    # to verify the constants when running this file directly.
    logger.info(f"MODEL_BACKEND_TYPE: {MODEL_BACKEND_TYPE}")
    logger.info(f"DEFAULT_VALUES for 'model_name': {DEFAULT_VALUES['model_name']}")