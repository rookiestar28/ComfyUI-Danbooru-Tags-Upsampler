import logging

from .danbooru_upsampler.node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

logger = logging.getLogger(__name__)

# Re-export them so ComfyUI can find them at the top level of this custom node directory
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

logger.debug("Loaded ComfyUI-Danbooru-Tags-Upsampler outer package bootstrap.")
