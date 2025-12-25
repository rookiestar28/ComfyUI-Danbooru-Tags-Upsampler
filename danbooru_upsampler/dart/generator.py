import logging
import time
import re

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
    # Key: (model_name, backend) -> model
    _model_cache: dict[tuple[str, str], PreTrainedModel | ORTModelForCausalLM] = {}
    # Key: model_name -> tokenizer
    _tokenizer_cache: dict[str, PreTrainedTokenizer | PreTrainedTokenizerFast] = {}
    # 當前使用的模型和分詞器
    dart_model: PreTrainedModel | ORTModelForCausalLM | None = None
    dart_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
    _current_model_key: tuple[str, str] | None = None
    _current_tokenizer_key: str | None = None

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        model_backend: str,
        model_device: str = "cpu",
        debug_logging: bool = False,
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
        self.onnx_file_name = onnx_file_name  # 儲存 ONNX 檔案名稱

        if debug_logging:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        logger.debug(f"DartGenerator initialized with: model_name='{model_name}', tokenizer_name='{tokenizer_name}', model_backend='{model_backend}', model_device='{model_device}', debug_logging={debug_logging}")

    def _load_dart_model(self):
        logger.debug(f"Loading DART model: {self.model_name} with backend: {self.model_backend}")
        is_onnx = False
        
        if self.model_backend == MODEL_BACKEND_TYPE.get("ORIGINAL", "original"):
            DartGenerator.dart_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        elif self.model_backend in [MODEL_BACKEND_TYPE.get("ONNX", "ONNX"), MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)")]:
            # 使用傳入的 onnx_file_name 或預設值
            file_name = self.onnx_file_name
            if not file_name:
                file_name = "model_quantized.onnx" if "Quantized" in self.model_backend else "model.onnx"
            
            DartGenerator.dart_model = ORTModelForCausalLM.from_pretrained(
                self.model_name,
                file_name=file_name,
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
            self.tokenizer_name, trust_remote_code=True # trust_remote_code might be needed depending on the model
        )
        assert DartGenerator.dart_tokenizer is not None, "Failed to load DART tokenizer"
        logger.info(f"DART tokenizer '{self.tokenizer_name}' loaded.")

    def _check_model_available(self) -> bool:
        """檢查當前請求的模型是否已載入"""
        cache_key = (self.model_name, self.model_backend)
        return cache_key == DartGenerator._current_model_key and DartGenerator.dart_model is not None

    def _check_tokenizer_available(self) -> bool:
        """檢查當前請求的分詞器是否已載入"""
        return self.tokenizer_name == DartGenerator._current_tokenizer_key and DartGenerator.dart_tokenizer is not None

    def load_model_if_needed(self):
        """載入模型，支援模型切換"""
        cache_key = (self.model_name, self.model_backend)
        
        # 檢查是否需要切換模型
        if cache_key != DartGenerator._current_model_key:
            # 嘗試從快取載入
            if cache_key in DartGenerator._model_cache:
                DartGenerator.dart_model = DartGenerator._model_cache[cache_key]
                DartGenerator._current_model_key = cache_key
                logger.info(f"Switched to cached model: {self.model_name} ({self.model_backend})")
            else:
                # 載入新模型
                self._load_dart_model()
                # 加入快取
                DartGenerator._model_cache[cache_key] = DartGenerator.dart_model  # type: ignore
                DartGenerator._current_model_key = cache_key

    def load_tokenizer_if_needed(self):
        """載入分詞器，支援切換"""
        if self.tokenizer_name != DartGenerator._current_tokenizer_key:
            # 嘗試從快取載入
            if self.tokenizer_name in DartGenerator._tokenizer_cache:
                DartGenerator.dart_tokenizer = DartGenerator._tokenizer_cache[self.tokenizer_name]
                DartGenerator._current_tokenizer_key = self.tokenizer_name
                logger.info(f"Switched to cached tokenizer: {self.tokenizer_name}")
            else:
                # 載入新分詞器
                self._load_dart_tokenizer()
                # 加入快取
                DartGenerator._tokenizer_cache[self.tokenizer_name] = DartGenerator.dart_tokenizer  # type: ignore
                DartGenerator._current_tokenizer_key = self.tokenizer_name

    def get_vocab_list(self) -> list[str]:
        self.load_tokenizer_if_needed()
        assert DartGenerator.dart_tokenizer is not None
        return list(DartGenerator.dart_tokenizer.vocab.keys()) # type: ignore

    def get_special_vocab_list(self) -> list[str]:
        self.load_tokenizer_if_needed()
        assert DartGenerator.dart_tokenizer is not None
        # .get_added_vocab() returns a dict of token string to int ID for added tokens
        # To get list of added token strings:
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

        start_time = time.time()

        self.load_tokenizer_if_needed()
        self.load_model_if_needed()

        assert DartGenerator.dart_tokenizer is not None, "Tokenizer not loaded"
        assert DartGenerator.dart_model is not None, "Model not loaded"
        
        # Ensure input_ids are on the same device as the model
        model_device = DartGenerator.dart_model.device

        try:
            input_ids_data = DartGenerator.dart_tokenizer.encode_plus(
                prompt, return_tensors="pt"
            )
            input_ids = input_ids_data.input_ids.to(model_device)
            # attention_mask = input_ids_data.attention_mask.to(model_device) # If model uses attention mask explicitly in generate
        except Exception as e:
            logger.error(f"Error encoding prompt: {prompt}. Error: {e}")
            return "" # Or raise error

        negative_prompt_ids_tensor = None
        # negative_prompt_attention_mask = None # If model uses attention mask explicitly
        if negative_prompt and negative_prompt.strip() != "":
            try:
                negative_ids_data = DartGenerator.dart_tokenizer.encode_plus(
                    negative_prompt,
                    return_tensors="pt",
                )
                negative_prompt_ids_tensor = negative_ids_data.input_ids.to(model_device)
                # negative_prompt_attention_mask = negative_ids_data.attention_mask.to(model_device)
            except Exception as e:
                logger.error(f"Error encoding negative_prompt: {negative_prompt}. Error: {e}")
                # Decide: proceed without negative, or fail? For now, proceed without.
                negative_prompt_ids_tensor = None


        # Prepare logits processor if CFG is enabled and negative prompt exists
        # 注意：ONNX 模型不支援 CFG logits processor (ORTModelForCausalLM 缺少 _is_stateful 屬性)
        logits_processor = None
        is_onnx_backend = self.model_backend in [
            MODEL_BACKEND_TYPE.get("ONNX", "ONNX"),
            MODEL_BACKEND_TYPE.get("ONNX_QUANTIZED", "ONNX (Quantized)")
        ]
        
        if negative_prompt_ids_tensor is not None and cfg_scale > 1.0:
            if is_onnx_backend:
                logger.warning("CFG (Classifier-Free Guidance) is not supported with ONNX backend. Skipping CFG.")
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
        
        # Ensure max_new_tokens is reasonable
        max_length = input_ids.shape[1] + max_new_tokens

        try:
            # ONNX 模型可能不支援所有生成參數，使用簡化版本
            if is_onnx_backend:
                # ONNX 模型使用簡化參數
                output_ids = DartGenerator.dart_model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k if do_sample else 0,
                    num_beams=num_beams,
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
            return f"Error generating tags: {e}"

        # 檢查輸出是否有效
        if output_ids is None:
            logger.error("model.generate returned None")
            return "Error generating tags: model returned None"
        
        logger.debug(f"Generate output type: {type(output_ids)}, shape: {output_ids.shape if hasattr(output_ids, 'shape') else 'N/A'}")

        # Decode only the newly generated tokens
        # output_ids[0] contains the full sequence (input_ids + generated_ids)
        generated_token_ids = output_ids[0][input_ids.shape[1]:]
        
        try:
            decoded = DartGenerator.dart_tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True, # Usually good for readability
            )
        except Exception as e:
            logger.error(f"Error decoding generated token IDs: {generated_token_ids}. Error: {e}")
            return "" # Or raise

        logger.debug(f"Raw generated tags: {decoded}")

        # REVIEW: escape_webui_special_symbols - its necessity in ComfyUI context.
        # For now, we keep it to maintain original behavior as much as possible.
        # If it causes issues or is not needed, it can be simplified to:
        # cleaned_tags = [tag.strip() for tag in decoded.split(",") if tag.strip()]
        # escaped = ", ".join(cleaned_tags)
        
        # Splitting by comma, applying escape, then rejoining.
        # This assumes tags are comma-separated in the decoded string.
        tags_list = [tag.strip() for tag in decoded.split(',') if tag.strip()]
        if not tags_list:
            escaped = ""
        else:
            # The original escape_webui_special_symbols expects a list of strings.
            escaped_tags_list = escape_webui_special_symbols(tags_list)
            escaped = ", ".join(escaped_tags_list)


        end_time = time.time()
        logger.info(f"Upsampling tags completed in {end_time - start_time:.2f} seconds. Output: '{escaped[:100]}...'")

        return escaped