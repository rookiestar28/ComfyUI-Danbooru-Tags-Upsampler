import logging
import time
import re
import threading
from contextlib import contextmanager

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    LogitsProcessorList,
)
from optimum.onnxruntime import ORTModelForCausalLM

# MODIFIED: Removed WebUI specific import: from modules.shared import opts

# MODIFIED: parse_options is likely tied to opts and WebUI settings structure.
# MODEL_BACKEND_TYPE can be kept if settings.py still defines it and it's useful.
# We assume MODEL_BACKEND_TYPE will be available from .settings or defined here/passed.
from .settings import MODEL_BACKEND_TYPE # Assuming MODEL_BACKEND_TYPE is still in settings.py and independent of opts
from .utils import (
    escape_webui_special_symbols, # We'll keep this for now, but review its necessity later
    get_valid_tag_list,
    get_patterns_from_tag_list,
)
from .logits_processor import UnbatchedClassifierFreeGuidanceLogitsProcessor

logger = logging.getLogger(__name__)
# MODIFIED: Default logging level, can be changed by ComfyUI node if needed
# logger.setLevel(logging.INFO) # User can configure logging level if debug_logging is an input

class DartGenerator:
    """A class for generating danbooru tags"""

    # Class-level cache for models and tokenizers (支援多模型切換)
    # Key: (model_name, approved_revision, backend, requested_device, ONNX artifact) -> model
    _model_cache: dict[
        tuple[str, str | None, str, str, str | None],
        PreTrainedModel | ORTModelForCausalLM,
    ] = {}
    # Requested runtime identity -> effective cache identity after provider/device fallback.
    _model_resolution_cache: dict[
        tuple[str, str | None, str, str, str | None],
        tuple[str, str | None, str, str, str | None],
    ] = {}
    # Key: (tokenizer_name, approved_revision, scoped_remote_code_trust) -> tokenizer
    _tokenizer_cache: dict[
        tuple[str, str | None, bool],
        PreTrainedTokenizer | PreTrainedTokenizerFast,
    ] = {}
    # 當前使用的模型和分詞器
    dart_model: PreTrainedModel | ORTModelForCausalLM | None = None
    dart_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
    _current_model_key: tuple[str, str | None, str, str, str | None] | None = None
    _current_tokenizer_key: tuple[str, str | None, bool] | None = None
    # CRITICAL: guard the shared class-level model/tokenizer cache; host integrations may call this runtime from worker threads.
    _runtime_lock = threading.RLock()

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        model_backend: str,
        model_device: str = "cpu",
        debug_logging: bool = False,
        model_revision: str | None = None,
        tokenizer_trust_remote_code: bool = False,
        onnx_file_name: str | None = None,  # 新增：ONNX 檔案名稱
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        if model_backend not in list(MODEL_BACKEND_TYPE.values()):
            # Fallback or raise error if MODEL_BACKEND_TYPE is not loaded or backend is unknown
            # For now, let's assume MODEL_BACKEND_TYPE is correctly loaded or handle potential KeyError
            logger.warning(f"MODEL_BACKEND_TYPE might not be fully initialized or '{model_backend}' is unknown. Proceeding with caution.")
            if not hasattr(MODEL_BACKEND_TYPE, 'get') or MODEL_BACKEND_TYPE.get(model_backend.upper().replace("-","_")) is None : # Simple check
                 # If strict validation is needed and MODEL_BACKEND_TYPE is critical,
                 # ensure it's properly passed or defined.
                 # For now, we can make it more robust against missing keys if necessary,
                 # or directly use string comparisons if MODEL_BACKEND_TYPE becomes problematic without WebUI context.
                 # Example: if model_backend not in ["original", "onnx", "onnx_quantized"]:
                 pass # Allowing it for now, but this should be robust

        self.model_backend = model_backend
        self.model_device = model_device
        self.model_revision = model_revision
        self.tokenizer_trust_remote_code = tokenizer_trust_remote_code
        self.onnx_file_name = onnx_file_name  # 儲存 ONNX 檔案名稱

        if debug_logging:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger.debug(f"DartGenerator initialized with: model_name='{model_name}', tokenizer_name='{tokenizer_name}', model_backend='{model_backend}', model_device='{model_device}', debug_logging={debug_logging}")

    @contextmanager
    def runtime_guard(self):
        with DartGenerator._runtime_lock:
            yield

    def _model_cache_key(
        self,
        *,
        device: str | None = None,
    ) -> tuple[str, str | None, str, str, str | None]:
        return (
            self.model_name,
            self.model_revision,
            self.model_backend,
            self.model_device if device is None else device,
            self.onnx_file_name,
        )

    def _tokenizer_cache_key(self) -> tuple[str, str | None, bool]:
        return (
            self.tokenizer_name,
            self.model_revision,
            self.tokenizer_trust_remote_code,
        )

    def get_actual_device(self) -> str:
        model = DartGenerator.dart_model
        if model is None:
            return self.model_device

        # IMPORTANT: ONNX Runtime providers are the execution truth; wrapper device fields can be stale.
        session = getattr(model, "model", None)
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            providers = tuple(str(provider) for provider in get_providers())
            if "CUDAExecutionProvider" in providers:
                return "cuda"
            if "CPUExecutionProvider" in providers:
                return "cpu"

        device = getattr(model, "device", None)
        if device is not None:
            normalized = str(device).strip().lower().split(":", 1)[0]
            if normalized in ("cpu", "cuda"):
                return normalized

        return self.model_device

    def _load_dart_model(self):
        logger.debug(f"Loading DART model: {self.model_name} with backend: {self.model_backend}")
        is_onnx = False

        if self.model_backend == MODEL_BACKEND_TYPE.get("ORIGINAL", "original"):
            DartGenerator.dart_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
            )
        elif self.model_backend in [MODEL_BACKEND_TYPE.get("ONNX", "ONNX"), MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)")]:
            # 使用傳入的 onnx_file_name 或預設值
            file_name = self.onnx_file_name
            if not file_name:
                file_name = "model_quantized.onnx" if "Quantized" in self.model_backend else "model.onnx"

            DartGenerator.dart_model = ORTModelForCausalLM.from_pretrained(
                self.model_name,
                file_name=file_name,
                revision=self.model_revision,
            )
            is_onnx = True
        else:
            logger.error(f"Unknown or unsupported model backend: {self.model_backend}")
            raise ValueError(f"Unknown or unsupported model backend: {self.model_backend}")

        assert DartGenerator.dart_model is not None, "Failed to load DART model"

        # 修復 optimum/transformers 版本不相容問題
        # ORTModelForCausalLM 類別缺少 _is_stateful 屬性，需手動添加
        if is_onnx and not hasattr(ORTModelForCausalLM, '_is_stateful'):
            ORTModelForCausalLM._is_stateful = False
            logger.debug("Patched ORTModelForCausalLM with _is_stateful attribute for compatibility.")

        try:
            DartGenerator.dart_model.to(self.model_device)
            logger.info(f"DART model '{self.model_name}' loaded to {self.model_device} using backend {self.model_backend}")
        except Exception as e:
            logger.error(f"Failed to move DART model to device '{self.model_device}': {e}")
            if self.model_device != "cpu":
                logger.info("Attempting to load DART model to CPU as a fallback.")
                try:
                    DartGenerator.dart_model.to("cpu")
                    logger.info(f"DART model '{self.model_name}' loaded to CPU (fallback) using backend {self.model_backend}")
                except Exception as fallback_e:
                    logger.error(f"Failed to load DART model to CPU (fallback): {fallback_e}")
                    raise fallback_e
            else:
                raise e


    def _load_dart_tokenizer(self):
        logger.debug(f"Loading DART tokenizer: {self.tokenizer_name}")
        # Ensure the class variable is updated
        DartGenerator.dart_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            revision=self.model_revision,
            trust_remote_code=self.tokenizer_trust_remote_code,
        )
        assert DartGenerator.dart_tokenizer is not None, "Failed to load DART tokenizer"
        logger.info(f"DART tokenizer '{self.tokenizer_name}' loaded.")

    def _check_model_available(self) -> bool:
        """檢查當前請求的模型是否已載入"""
        requested_key = self._model_cache_key()
        effective_key = DartGenerator._model_resolution_cache.get(requested_key, requested_key)
        return effective_key == DartGenerator._current_model_key and DartGenerator.dart_model is not None

    def _check_tokenizer_available(self) -> bool:
        """檢查當前請求的分詞器是否已載入"""
        return self._tokenizer_cache_key() == DartGenerator._current_tokenizer_key and DartGenerator.dart_tokenizer is not None

    def load_model_if_needed(self):
        """載入模型，支援模型切換"""
        requested_key = self._model_cache_key()
        with self.runtime_guard():
            effective_key = DartGenerator._model_resolution_cache.get(
                requested_key,
                requested_key,
            )
            if (
                effective_key != DartGenerator._current_model_key
                or DartGenerator.dart_model is None
            ):
                if effective_key in DartGenerator._model_cache:
                    DartGenerator.dart_model = DartGenerator._model_cache[effective_key]
                    DartGenerator._current_model_key = effective_key
                    logger.info(f"Switched to cached model: {self.model_name} ({self.model_backend})")
                else:
                    self._load_dart_model()
                    actual_device = self.get_actual_device()
                    effective_key = self._model_cache_key(device=actual_device)
                    DartGenerator._model_cache[effective_key] = DartGenerator.dart_model  # type: ignore
                    DartGenerator._model_resolution_cache[requested_key] = effective_key
                    DartGenerator._current_model_key = effective_key

    def load_tokenizer_if_needed(self):
        """載入分詞器，支援切換"""
        cache_key = self._tokenizer_cache_key()
        with self.runtime_guard():
            if (
                cache_key != DartGenerator._current_tokenizer_key
                or DartGenerator.dart_tokenizer is None
            ):
                if cache_key in DartGenerator._tokenizer_cache:
                    DartGenerator.dart_tokenizer = DartGenerator._tokenizer_cache[cache_key]
                    DartGenerator._current_tokenizer_key = cache_key
                    logger.info(f"Switched to cached tokenizer: {self.tokenizer_name}")
                else:
                    self._load_dart_tokenizer()
                    DartGenerator._tokenizer_cache[cache_key] = DartGenerator.dart_tokenizer  # type: ignore
                    DartGenerator._current_tokenizer_key = cache_key

    def get_vocab_list(self) -> list[str]:
        with self.runtime_guard():
            self.load_tokenizer_if_needed()
            assert DartGenerator.dart_tokenizer is not None
            return list(DartGenerator.dart_tokenizer.vocab.keys()) # type: ignore

    def get_special_vocab_list(self) -> list[str]:
        with self.runtime_guard():
            self.load_tokenizer_if_needed()
            assert DartGenerator.dart_tokenizer is not None
            return list(DartGenerator.dart_tokenizer.get_added_vocab().keys()) # type: ignore


    def compose_prompt(
        self, rating: str, copyright: str, character: str, general: str, length: str
    ) -> str:
        # This method remains largely the same as it's specific to the DART model's expected input format.
        # No WebUI specific code here.
        # The commented out "apply_chat_template" is fine as is.
        return f"<|bos|><rating>{rating}</rating><copyright>{copyright}</copyright><character>{character}</character><general>{length}{general}<|input_end|>"

    def get_bad_words_ids(self, tag_text: str) -> list[list[int]] | None:
        if not tag_text or tag_text.strip() == "":
            return None

        self.load_tokenizer_if_needed()
        assert DartGenerator.dart_tokenizer is not None

        # sanitize_special_tokens might not be necessary or available on all tokenizers.
        # It's usually for preventing added special tokens from being split during tokenization by other models.
        # For now, let's assume it's okay or check if it causes issues.
        # if hasattr(DartGenerator.dart_tokenizer, 'sanitize_special_tokens'):
        #     DartGenerator.dart_tokenizer.sanitize_special_tokens()


        ban_tags = get_valid_tag_list(tag_text)
        if not ban_tags:
            return None

        ban_tag_patterns = get_patterns_from_tag_list(ban_tags)
        if not ban_tag_patterns:
            return None

        ban_words_ids: list[int] = []
        # Make sure vocab is accessible and is a dict string to int
        vocab = getattr(DartGenerator.dart_tokenizer, 'vocab', {})
        if not isinstance(vocab, dict): # some tokenizers might have vocab differently.
            logger.warning("Tokenizer vocab format not as expected (dict str->int). Ban tags might not work.")
            return None

        for pattern in ban_tag_patterns:
            for tag, token_id in vocab.items():
                if isinstance(tag, str) and isinstance(token_id, int) and pattern.match(tag):
                    ban_words_ids.append(token_id)

        if not ban_words_ids:
            return None

        # dedup
        ban_words_ids = sorted(list(set(ban_words_ids))) # Added sorted for consistent behavior

        return [[id_val] for id_val in ban_words_ids]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0, # Original had 0
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0, # Original had 1
        top_k: int = 20,
        num_beams: int = 1,
        bad_words_ids: list[list[int]] | None = None,
        negative_prompt: str | None = None,
        cfg_scale: float = 1.5, # CFG scale for the custom logits processor
        # no_repeat_ngram_size: int = 1, # Original hardcoded this
    ) -> str:
        """Upsamples prompt using the DART model"""

        with self.runtime_guard():
            start_time = time.time()

            self.load_tokenizer_if_needed()
            self.load_model_if_needed()

            assert DartGenerator.dart_tokenizer is not None, "Tokenizer not loaded"
            assert DartGenerator.dart_model is not None, "Model not loaded"
            model_device = DartGenerator.dart_model.device

            try:
                input_ids_data = DartGenerator.dart_tokenizer.encode_plus(
                    prompt, return_tensors="pt"
                )
                input_ids = input_ids_data.input_ids.to(model_device)
            except Exception as e:
                logger.error(f"Error encoding prompt: {prompt}. Error: {e}")
                raise RuntimeError("Failed to encode the DART prompt input.") from e

            negative_prompt_ids_tensor = None
            if negative_prompt and negative_prompt.strip() != "":
                try:
                    negative_ids_data = DartGenerator.dart_tokenizer.encode_plus(
                        negative_prompt,
                        return_tensors="pt",
                    )
                    negative_prompt_ids_tensor = negative_ids_data.input_ids.to(model_device)
                except Exception as e:
                    logger.error(f"Error encoding negative_prompt: {negative_prompt}. Error: {e}")
                    negative_prompt_ids_tensor = None

            logits_processor = None
            is_onnx_backend = self.model_backend in [
                MODEL_BACKEND_TYPE.get("ONNX", "ONNX"),
                MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)")
            ]

            if negative_prompt_ids_tensor is not None and cfg_scale > 1.0:
                if is_onnx_backend:
                    # IMPORTANT: never silently ignore a requested generation control; callers must receive a deterministic failure.
                    raise ValueError("CFG is not supported with ONNX backends.")
                else:
                    logits_processor = LogitsProcessorList(
                        [
                            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                                guidance_scale=cfg_scale,
                                model=DartGenerator.dart_model,
                                unconditional_ids=negative_prompt_ids_tensor,
                            )
                        ]
                    )

            max_length = input_ids.shape[1] + max_new_tokens

            try:
                if is_onnx_backend:
                    output_ids = DartGenerator.dart_model.generate(
                        input_ids,
                        max_length=max_length,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        top_k=top_k if do_sample else 0,
                        num_beams=num_beams,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=DartGenerator.dart_tokenizer.eos_token_id,
                    )
                else:
                    output_ids = DartGenerator.dart_model.generate(
                        input_ids,
                        max_length=max_length,
                        min_new_tokens=min_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_beams=num_beams,
                        bad_words_ids=bad_words_ids,
                        no_repeat_ngram_size=1,
                        logits_processor=logits_processor,
                    )
            except Exception as e:
                logger.error(f"Error during model.generate: {e}")
                logger.error(f"Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
                logger.error(f"Parameters: max_length={max_length}, do_sample={do_sample}, temperature={temperature}, top_p={top_p}, top_k={top_k}, num_beams={num_beams}")
                raise RuntimeError("Failed to run DART model.generate().") from e

            if output_ids is None:
                logger.error("model.generate returned None")
                raise RuntimeError("DART model returned no generated output.")

            logger.debug(f"Generate output type: {type(output_ids)}, shape: {output_ids.shape if hasattr(output_ids, 'shape') else 'N/A'}")
            generated_token_ids = output_ids[0][input_ids.shape[1]:]

            try:
                decoded = DartGenerator.dart_tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            except Exception as e:
                logger.error(f"Error decoding generated token IDs: {generated_token_ids}. Error: {e}")
                raise RuntimeError("Failed to decode DART generated token IDs.") from e

            logger.debug(f"Raw generated tags: {decoded}")
            tags_list = [tag.strip() for tag in decoded.split(',') if tag.strip()]
            if not tags_list:
                escaped = ""
            else:
                escaped_tags_list = escape_webui_special_symbols(tags_list)
                escaped = ", ".join(escaped_tags_list)

            end_time = time.time()
            logger.info(f"Upsampling tags completed in {end_time - start_time:.2f} seconds. Output: '{escaped[:100]}...'")

            return escaped
