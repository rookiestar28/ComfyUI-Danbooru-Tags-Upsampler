# File: ComfyUI-Danbooru-Tags-Upsampler/danbooru_upsampler/__init__.py

# Import the node class mappings and display name mappings from your nodes.py file
from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .service import (
    DEFAULT_TOOLBAR_PROFILE,
    DanbooruUpsamplerAnalyzerError,
    DanbooruUpsamplerError,
    DanbooruUpsamplerGenerationError,
    DanbooruUpsamplerInvalidRequestError,
    DanbooruUpsamplerRequest,
    DanbooruUpsamplerResolvedRuntime,
    DanbooruUpsamplerResult,
    DanbooruUpsamplerRuntimeInitializationError,
    DanbooruUpsamplerToolbarProfile,
    build_toolbar_request,
    resolve_runtime_selection,
    upsample_prompt,
)

# This is the standard way to export the mappings for ComfyUI
__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'DEFAULT_TOOLBAR_PROFILE',
    'DanbooruUpsamplerAnalyzerError',
    'DanbooruUpsamplerError',
    'DanbooruUpsamplerGenerationError',
    'DanbooruUpsamplerInvalidRequestError',
    'DanbooruUpsamplerRequest',
    'DanbooruUpsamplerResolvedRuntime',
    'DanbooruUpsamplerResult',
    'DanbooruUpsamplerRuntimeInitializationError',
    'DanbooruUpsamplerToolbarProfile',
    'build_toolbar_request',
    'resolve_runtime_selection',
    'upsample_prompt',
]

# You can add a print statement to confirm the node is being loaded by ComfyUI
# This will appear in the console when ComfyUI starts up.
print("Initializing Danbooru Tags Upsampler (Ray) custom node...")
