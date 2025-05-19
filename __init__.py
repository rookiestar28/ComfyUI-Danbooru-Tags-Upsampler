# File: ComfyUI-Danbooru-Tags-Upsampler/__init__.py

# Import the mappings from your actual node package (danbooru_upsampler)
# and the correct module name ('node' instead of 'nodes')
from .danbooru_upsampler.node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS # MODIFIED HERE

# Re-export them so ComfyUI can find them at the top level of this custom node directory
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Optional print statement to see if this outer __init__.py is loaded
print("ComfyUI-Danbooru-Tags-Upsampler package (outer __init__.py) is being loaded...")