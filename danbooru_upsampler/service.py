from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import set_seed

from .dart.analyzer import DartAnalyzer, ImagePromptAnalyzingResult
from .dart.generator import DartGenerator
from .dart.settings import DART_MODELS, DEFAULT_MODEL, MODEL_BACKEND_TYPE
from .dart.utils import SEED_MAX

logger = logging.getLogger(__name__)

DEFAULT_TAGS_DIR = Path(__file__).resolve().parent.parent / "tags"
TOOLBAR_PROFILE_ID = "rookieui_editor_toolbar_v1"
TAG_LENGTH_OPTIONS = ("very short", "short", "long", "very long")
TAG_LENGTH_MAP = {
    "very short": "<|very_short|>",
    "short": "<|short|>",
    "long": "<|long|>",
    "very long": "<|very_long|>",
}
DEFAULT_TOOLBAR_MODEL_BACKEND = MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)")


class DanbooruUpsamplerError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class DanbooruUpsamplerInvalidRequestError(DanbooruUpsamplerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_request")


class DanbooruUpsamplerUnsupportedFeatureError(DanbooruUpsamplerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_feature")


class DanbooruUpsamplerRuntimeInitializationError(DanbooruUpsamplerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="runtime_initialization_failed")


class DanbooruUpsamplerAnalyzerError(DanbooruUpsamplerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="analyzer_failed")


class DanbooruUpsamplerGenerationError(DanbooruUpsamplerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="generation_failed")


@dataclass(frozen=True)
class DanbooruBackendCapabilities:
    backend: str
    supports_cfg: bool
    supports_ban_tags: bool
    supported_devices: tuple[str, ...]
    supports_artifact_fallback: bool


@dataclass(frozen=True)
class DanbooruUpsamplerToolbarProfile:
    profile_id: str = TOOLBAR_PROFILE_ID
    model_name: str = DEFAULT_MODEL
    tag_length: str = "long"
    seed: int = 0
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 1.0
    num_beams: int = 1
    model_device: str = "auto"
    model_backend: str = DEFAULT_TOOLBAR_MODEL_BACKEND
    max_new_tokens: int = 128
    cfg_scale: float = 1.5
    debug_logging: bool = False
    append_prompt: bool = True


DEFAULT_TOOLBAR_PROFILE = DanbooruUpsamplerToolbarProfile()


@dataclass(frozen=True)
class DanbooruUpsamplerRequest:
    prompt: str
    model_name: str = DEFAULT_MODEL
    tag_length: str = "long"
    seed: int = 0
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 1.0
    num_beams: int = 1
    model_device: str = "auto"
    model_backend: str = DEFAULT_TOOLBAR_MODEL_BACKEND
    max_new_tokens: int = 128
    negative_prompt_tags: str = ""
    ban_tags: str = ""
    cfg_scale: float = 1.5
    debug_logging: bool = False
    append_prompt: bool = True
    escape_input_brackets_enabled: bool = True
    escape_output_brackets_enabled: bool = True


@dataclass(frozen=True)
class DanbooruUpsamplerResolvedRuntime:
    model_name: str
    model_repo: str
    model_revision: str
    trust_remote_code: bool
    requested_backend: str
    resolved_backend: str
    resolved_device: str
    onnx_file_name: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DanbooruUpsamplerResult:
    final_prompt: str
    generated_suffix: str
    model_name: str
    model_repo: str
    requested_backend: str
    resolved_backend: str
    resolved_device: str
    tag_length: str
    onnx_file_name: str | None = None
    model_revision: str | None = None
    warnings: tuple[str, ...] = ()


def _normalize_device(model_device: str) -> str:
    normalized = str(model_device or "").strip().lower()
    if normalized in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized in ("cpu", "cuda"):
        if normalized == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested CUDA device is unavailable; falling back to CPU.")
            return "cpu"
        return normalized
    raise DanbooruUpsamplerInvalidRequestError(
        "model_device must be one of: auto, cpu, cuda."
    )


def _validate_backend(model_backend: str) -> str:
    normalized = str(model_backend or "").strip()
    if normalized in MODEL_BACKEND_TYPE.values():
        return normalized
    raise DanbooruUpsamplerInvalidRequestError(
        "model_backend must match one of the shipped backend options."
    )


def resolve_backend_capabilities(model_backend: str) -> DanbooruBackendCapabilities:
    backend = _validate_backend(model_backend)
    is_original = backend == MODEL_BACKEND_TYPE.get("ORIGINAL", "Original")
    return DanbooruBackendCapabilities(
        backend=backend,
        supports_cfg=is_original,
        supports_ban_tags=True,
        supported_devices=("cpu", "cuda"),
        supports_artifact_fallback=True,
    )


def _validate_tag_length(tag_length: str) -> str:
    normalized = str(tag_length or "").strip().lower()
    if normalized in TAG_LENGTH_MAP:
        return normalized
    raise DanbooruUpsamplerInvalidRequestError(
        "tag_length must be one of: very short, short, long, very long."
    )


def _coerce_int(value: object, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DanbooruUpsamplerInvalidRequestError(
            f"{field_name} must be an integer-compatible value."
        ) from exc


def _coerce_float(value: object, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DanbooruUpsamplerInvalidRequestError(
            f"{field_name} must be a float-compatible value."
        ) from exc


def _validate_int_range(
    value: object,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    normalized = _coerce_int(value, field_name=field_name)
    if normalized < minimum or normalized > maximum:
        raise DanbooruUpsamplerInvalidRequestError(
            f"{field_name} must be between {minimum} and {maximum}, inclusive."
        )
    return normalized


def _validate_float_range(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    normalized = _coerce_float(value, field_name=field_name)
    if not math.isfinite(normalized):
        raise DanbooruUpsamplerInvalidRequestError(
            f"{field_name} must be a finite number."
        )
    if normalized < minimum or normalized > maximum:
        raise DanbooruUpsamplerInvalidRequestError(
            f"{field_name} must be between {minimum} and {maximum}, inclusive."
        )
    return normalized


def resolve_runtime_selection(
    *,
    model_name: str,
    model_backend: str,
    model_device: str,
) -> DanbooruUpsamplerResolvedRuntime:
    normalized_model_name = str(model_name or "").strip()
    model_info = DART_MODELS.get(normalized_model_name)
    if not model_info:
        raise DanbooruUpsamplerInvalidRequestError(
            f"Unknown DART model '{normalized_model_name}'."
        )

    requested_backend = _validate_backend(model_backend)
    resolved_backend = requested_backend
    resolved_device = _normalize_device(model_device)
    onnx_file_name: str | None = None
    warnings: list[str] = []
    onnx_files = model_info.get("onnx_files", [])

    if requested_backend == MODEL_BACKEND_TYPE.get("ONNX", "ONNX"):
        if "model.onnx" in onnx_files:
            onnx_file_name = "model.onnx"
        elif "model_quantized.onnx" in onnx_files:
            resolved_backend = MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)")
            onnx_file_name = "model_quantized.onnx"
            warnings.append(
                f"Model '{normalized_model_name}' only provides a quantized ONNX artifact; using ONNX (Quantized)."
            )
        else:
            resolved_backend = MODEL_BACKEND_TYPE.get("ORIGINAL", "Original")
            warnings.append(
                f"Model '{normalized_model_name}' does not provide ONNX artifacts; falling back to Original."
            )
    elif requested_backend == MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)"):
        if "model_quantized.onnx" in onnx_files:
            onnx_file_name = "model_quantized.onnx"
        else:
            resolved_backend = MODEL_BACKEND_TYPE.get("ORIGINAL", "Original")
            warnings.append(
                f"Model '{normalized_model_name}' does not provide a quantized ONNX artifact; falling back to Original."
            )

    return DanbooruUpsamplerResolvedRuntime(
        model_name=normalized_model_name,
        model_repo=str(model_info.get("repo", "")).strip(),
        model_revision=str(model_info["revision"]),
        trust_remote_code=bool(model_info["trust_remote_code"]),
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        resolved_device=resolved_device,
        onnx_file_name=onnx_file_name,
        warnings=tuple(warnings),
    )


def build_toolbar_request(
    prompt: object,
    *,
    negative_prompt_tags: object = "",
    ban_tags: object = "",
    seed: object | None = None,
    profile: DanbooruUpsamplerToolbarProfile = DEFAULT_TOOLBAR_PROFILE,
) -> DanbooruUpsamplerRequest:
    normalized_seed = profile.seed if seed is None else _coerce_int(seed, field_name="seed")
    return DanbooruUpsamplerRequest(
        prompt=str(prompt or ""),
        model_name=profile.model_name,
        tag_length=profile.tag_length,
        seed=normalized_seed,
        temperature=profile.temperature,
        top_k=profile.top_k,
        top_p=profile.top_p,
        num_beams=profile.num_beams,
        model_device=profile.model_device,
        model_backend=profile.model_backend,
        max_new_tokens=profile.max_new_tokens,
        negative_prompt_tags=str(negative_prompt_tags or ""),
        ban_tags=str(ban_tags or ""),
        cfg_scale=profile.cfg_scale,
        debug_logging=profile.debug_logging,
        append_prompt=profile.append_prompt,
    )


def _compose_cfg_negative_prompt(
    generator: DartGenerator,
    *,
    positive_result: ImagePromptAnalyzingResult,
    negative_result: ImagePromptAnalyzingResult | None,
    length_special_token: str,
) -> str | None:
    if negative_result is None:
        return None

    def _join_texts(text1: str, text2: str) -> str:
        left = text1.strip()
        right = text2.strip()
        if left and right:
            return f"{left}, {right}"
        return left or right

    return generator.compose_prompt(
        rating=f"{positive_result.rating_parent}, {positive_result.rating_child}",
        copyright=_join_texts(positive_result.copyright, negative_result.copyright),
        character=_join_texts(positive_result.character, negative_result.character),
        general=negative_result.general,
        length=length_special_token,
    )


def upsample_prompt(
    request: DanbooruUpsamplerRequest,
    *,
    tags_dir: Path = DEFAULT_TAGS_DIR,
) -> DanbooruUpsamplerResult:
    normalized_prompt = str(request.prompt or "").strip()
    tag_length = _validate_tag_length(request.tag_length)
    # IMPORTANT: validate host-facing numeric inputs before runtime construction; external callers rely on typed invalid_request failures instead of raw cast errors.
    normalized_cfg_scale = _validate_float_range(
        request.cfg_scale,
        field_name="cfg_scale",
        minimum=1.0,
        maximum=10.0,
    )
    normalized_seed = _validate_int_range(
        request.seed,
        field_name="seed",
        minimum=0,
        maximum=SEED_MAX,
    )
    normalized_max_new_tokens = _validate_int_range(
        request.max_new_tokens,
        field_name="max_new_tokens",
        minimum=8,
        maximum=512,
    )
    normalized_temperature = _validate_float_range(
        request.temperature,
        field_name="temperature",
        minimum=0.01,
        maximum=5.0,
    )
    normalized_top_p = _validate_float_range(
        request.top_p,
        field_name="top_p",
        minimum=0.0,
        maximum=1.0,
    )
    normalized_top_k = _validate_int_range(
        request.top_k,
        field_name="top_k",
        minimum=0,
        maximum=1000,
    )
    normalized_num_beams = _validate_int_range(
        request.num_beams,
        field_name="num_beams",
        minimum=1,
        maximum=20,
    )
    runtime = resolve_runtime_selection(
        model_name=request.model_name,
        model_backend=request.model_backend,
        model_device=request.model_device,
    )
    capabilities = resolve_backend_capabilities(runtime.resolved_backend)
    negative_prompt_tags = str(request.negative_prompt_tags or "").strip()
    if negative_prompt_tags and normalized_cfg_scale > 1.0 and not capabilities.supports_cfg:
        # IMPORTANT: unsupported host features must fail before heavy runtime construction; silent no-ops produce misleading workflows.
        raise DanbooruUpsamplerUnsupportedFeatureError(
            f"CFG is not supported by backend '{runtime.resolved_backend}'. "
            "Use the Original backend or clear negative_prompt_tags."
        )
    if str(request.ban_tags or "").strip() and not capabilities.supports_ban_tags:
        raise DanbooruUpsamplerUnsupportedFeatureError(
            f"Ban tags are not supported by backend '{runtime.resolved_backend}'."
        )

    generator = DartGenerator(
        model_name=runtime.model_repo,
        tokenizer_name=runtime.model_repo,
        model_backend=runtime.resolved_backend,
        model_device=runtime.resolved_device,
        debug_logging=bool(request.debug_logging),
        model_revision=runtime.model_revision,
        tokenizer_trust_remote_code=runtime.trust_remote_code,
        onnx_file_name=runtime.onnx_file_name,
    )
    actual_device = runtime.resolved_device
    result_warnings = list(runtime.warnings)

    with generator.runtime_guard():
        try:
            generator.load_model_if_needed()
            generator.load_tokenizer_if_needed()
            reported_device = str(generator.get_actual_device()).strip().lower().split(":", 1)[0]
            if reported_device in ("cpu", "cuda"):
                actual_device = reported_device
            if actual_device != runtime.resolved_device:
                result_warnings.append(
                    f"Requested runtime device '{runtime.resolved_device}' fell back to '{actual_device}'."
                )
        except Exception as exc:
            raise DanbooruUpsamplerRuntimeInitializationError(
                f"Failed to initialize DART runtime: {exc}"
            ) from exc

        try:
            analyzer = DartAnalyzer(
                tags_dir_path=Path(tags_dir),
                vocab=generator.get_vocab_list(),
                special_vocab=generator.get_special_vocab_list(),
                debug_logging=bool(request.debug_logging),
                escape_input_brackets_enabled=bool(request.escape_input_brackets_enabled),
                escape_output_brackets_enabled=bool(request.escape_output_brackets_enabled),
            )
        except Exception as exc:
            raise DanbooruUpsamplerAnalyzerError(
                f"Failed to initialize the Danbooru prompt analyzer: {exc}"
            ) from exc

        try:
            positive_result = analyzer.analyze(normalized_prompt)
            negative_result: ImagePromptAnalyzingResult | None = None
            if negative_prompt_tags and normalized_cfg_scale > 1.0:
                negative_result = analyzer.analyze(negative_prompt_tags)
        except Exception as exc:
            raise DanbooruUpsamplerAnalyzerError(
                f"Failed to analyze Danbooru prompt tags: {exc}"
            ) from exc

        length_special_token = TAG_LENGTH_MAP[tag_length]
        llm_prompt = generator.compose_prompt(
            rating=f"{positive_result.rating_parent}, {positive_result.rating_child}",
            copyright=positive_result.copyright,
            character=positive_result.character,
            general=positive_result.general,
            length=length_special_token,
        )
        cfg_negative_prompt = _compose_cfg_negative_prompt(
            generator,
            positive_result=positive_result,
            negative_result=negative_result,
            length_special_token=length_special_token,
        )
        bad_words_ids = generator.get_bad_words_ids(str(request.ban_tags or "").strip())
        set_seed(normalized_seed % (SEED_MAX + 1))

        try:
            generated_suffix = generator.generate(
                prompt=llm_prompt,
                max_new_tokens=normalized_max_new_tokens,
                min_new_tokens=0,
                do_sample=True,
                temperature=normalized_temperature,
                top_p=normalized_top_p,
                top_k=normalized_top_k,
                num_beams=normalized_num_beams,
                bad_words_ids=bad_words_ids,
                negative_prompt=cfg_negative_prompt if normalized_cfg_scale > 1.0 else None,
                cfg_scale=normalized_cfg_scale if cfg_negative_prompt else 1.0,
            )
        except Exception as exc:
            raise DanbooruUpsamplerGenerationError(
                f"Failed to generate Danbooru tag upsampling output: {exc}"
            ) from exc

    normalized_suffix = str(generated_suffix or "").strip()
    if normalized_suffix.startswith("Error generating tags:"):
        raise DanbooruUpsamplerGenerationError(normalized_suffix)

    if request.append_prompt:
        if normalized_prompt and normalized_suffix:
            final_prompt = f"{normalized_prompt}, {normalized_suffix}"
        else:
            final_prompt = normalized_prompt or normalized_suffix
    else:
        final_prompt = normalized_suffix

    return DanbooruUpsamplerResult(
        final_prompt=final_prompt,
        generated_suffix=normalized_suffix,
        model_name=runtime.model_name,
        model_repo=runtime.model_repo,
        requested_backend=runtime.requested_backend,
        resolved_backend=runtime.resolved_backend,
        resolved_device=actual_device,
        tag_length=tag_length,
        onnx_file_name=runtime.onnx_file_name,
        model_revision=runtime.model_revision,
        warnings=tuple(result_warnings),
    )
