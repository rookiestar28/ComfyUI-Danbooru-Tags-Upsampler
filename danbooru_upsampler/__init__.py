import logging

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

logger = logging.getLogger(__name__)

logger.debug("Initialized danbooru_upsampler package exports.")
