# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging # MODIFIED: Changed to standard logging to avoid potential conflicts if logger was pre-configured
from typing import Optional

import torch

from transformers.generation import LogitsProcessor # This import should work as long as transformers is installed

# MODIFIED: Changed logger instantiation to avoid potential global state issues with 'logging.Logger(__name__)'
# It's generally safer to use logging.getLogger for application code.
logger = logging.getLogger(__name__)


# Copied from transformers.generation.logits_processor.UnbatchedClassifierFreeGuidanceLogitsProcessor
class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`PreTrainedModel`): # In transformers, this is typically the model object
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.
    """

    def __init__(
        self,
        guidance_scale: float,
        model, # This should be an instance of a Hugging Face PreTrainedModel
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True, # Transformers default is True
    ):
        if not isinstance(guidance_scale, float) or guidance_scale < 0: # Basic validation
            raise ValueError(f"`guidance_scale` has to be a non-negative float, but is {guidance_scale}")
        
        self.guidance_scale = guidance_scale
        self.model = model # Stores the model instance (e.g., self.dart_model from DartGenerator)
        
        # Ensure unconditional_ids and unconditional_attention_mask are on the same device as the model
        # This should ideally happen before passing them here, or this class needs to handle it.
        # For now, assume they are on the correct device or model's forward pass handles device mismatches for inputs.
        device = model.device if hasattr(model, 'device') else (unconditional_ids.device if unconditional_ids is not None else 'cpu')

        self.unconditional_context = {
            "input_ids": unconditional_ids.to(device) if unconditional_ids is not None else None,
            "attention_mask": unconditional_attention_mask.to(device) if unconditional_attention_mask is not None else None,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True, # Flag to handle the first pass of unconditional generation
        }

    def get_unconditional_logits(self, current_conditional_input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Computes the logits for the unconditional prompt.
        Handles caching of past_key_values for the unconditional branch.
        """
        model_device = self.model.device
        batch_size = current_conditional_input_ids.shape[0] # Get batch_size from conditional input

        uc_input_ids = self.unconditional_context["input_ids"]
        uc_attention_mask = self.unconditional_context["attention_mask"]
        
        if self.unconditional_context["first_pass"]:
            if uc_input_ids is None:
                # Default to a single BOS token or similar if no unconditional_ids are provided.
                # Or, this case should be an error if unconditional generation is expected.
                # The original transformers code might default to last token of prompt, but here we don't have the conditional prompt easily.
                # This processor is typically used when unconditional_ids ARE provided.
                # If this happens, it might mean cfg_scale > 1 but no negative_prompt was given to DartGenerator.
                logger.warning("Unconditional_ids not provided for CFG, but guidance_scale > 1. Using a single BOS token as fallback for unconditional input.")
                # Attempt to get BOS token ID from model's tokenizer, if possible and model has tokenizer linked
                bos_token_id = getattr(self.model.config, 'bos_token_id', 0) # Fallback to 0 if not found
                if self.model.tokenizer and hasattr(self.model.tokenizer, 'bos_token_id') and self.model.tokenizer.bos_token_id is not None:
                    bos_token_id = self.model.tokenizer.bos_token_id

                uc_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=model_device).repeat(batch_size, 1)

            if uc_attention_mask is None and uc_input_ids is not None:
                uc_attention_mask = torch.ones_like(uc_input_ids, dtype=torch.long, device=model_device)
            
            # Ensure uc_input_ids and uc_attention_mask match the batch size of the conditional input
            # This is crucial if the original unconditional_ids were for batch_size=1 but conditional is >1
            if uc_input_ids is not None and uc_input_ids.shape[0] != batch_size:
                if uc_input_ids.shape[0] == 1: # Common case: negative prompt for single batch item
                    uc_input_ids = uc_input_ids.expand(batch_size, -1)
                    if uc_attention_mask is not None and uc_attention_mask.shape[0] == 1:
                        uc_attention_mask = uc_attention_mask.expand(batch_size, -1)
                else: # Mismatch that's not easily expandable
                    raise ValueError(
                        f"Batch size mismatch for CFG: conditional input has batch_size {batch_size}, "
                        f"but unconditional_ids have batch_size {uc_input_ids.shape[0]}."
                    )
            
            self.unconditional_context["input_ids"] = uc_input_ids
            self.unconditional_context["attention_mask"] = uc_attention_mask
            self.unconditional_context["first_pass"] = False
            current_uc_input_ids = uc_input_ids
            current_uc_attention_mask = uc_attention_mask

        else:
            # Subsequent steps: append the new conditional token to the unconditional sequence
            new_token_id = current_conditional_input_ids[:, -1:].clone() # Get the last token from conditional input (current step)
            
            if not self.unconditional_context["use_cache"]: # If not using cache, append to full sequence
                current_uc_input_ids = torch.cat([self.unconditional_context["input_ids"], new_token_id], dim=1)
            else: # If using cache, only pass the new token
                current_uc_input_ids = new_token_id

            if self.unconditional_context["attention_mask"] is not None:
                 current_uc_attention_mask = torch.cat(
                    [
                        self.unconditional_context["attention_mask"],
                        torch.ones_like(new_token_id, dtype=torch.long, device=model_device),
                    ],
                    dim=1,
                )
            else: # Should not happen if first_pass set it
                current_uc_attention_mask = torch.ones_like(current_uc_input_ids, dtype=torch.long, device=model_device)

            # Update context for next step (only if not using cache, otherwise input_ids refers to the new token only)
            if not self.unconditional_context["use_cache"]:
                self.unconditional_context["input_ids"] = current_uc_input_ids
            self.unconditional_context["attention_mask"] = current_uc_attention_mask
        
        if current_uc_input_ids is None: # Should be handled by first_pass logic
             raise ValueError("Unconditional input_ids are None after first_pass logic. This should not happen.")

        # Perform model forward pass for unconditional logits
        outputs = self.model(
            input_ids=current_uc_input_ids,
            attention_mask=current_uc_attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = outputs.get("past_key_values", None)

        return outputs.logits

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        # `input_ids` here are the *conditional* input_ids for the current generation step
        # `scores` are the *conditional* logits for the next token (output of the main model call)

        if self.guidance_scale == 1.0: # No guidance
            return scores
        
        # Apply log_softmax to conditional scores
        # The original paper and many implementations do CFG in log-probability space
        log_probs_conditional = torch.nn.functional.log_softmax(scores, dim=-1)

        # Get unconditional logits for the current step.
        # The `get_unconditional_logits` method handles the model call with its own context.
        # `input_ids` (conditional) is passed to help determine the new token for the unconditional branch if using KV cache.
        unconditional_full_logits = self.get_unconditional_logits(input_ids)
        
        # We need the logits for the *next* token from the unconditional pass.
        # This corresponds to the same position as `scores` (which are for the next token of conditional pass).
        unconditional_next_token_logits = unconditional_full_logits[:, -1, :] # Get logits for the last token position
        log_probs_unconditional = torch.nn.functional.log_softmax(unconditional_next_token_logits, dim=-1)

        # CFG formula: combined_log_probs = unconditional_log_probs + guidance_scale * (conditional_log_probs - unconditional_log_probs)
        # This can be rewritten as: guidance_scale * conditional_log_probs + (1 - guidance_scale) * unconditional_log_probs
        
        # The formula used in the original copied code was:
        # out = guidance_scale * (scores - unconditional_logits) + unconditional_logits
        # where scores and unconditional_logits were already log_softmaxed.
        # So, log_probs_combined = log_probs_unconditional + self.guidance_scale * (log_probs_conditional - log_probs_unconditional)

        combined_log_probs = log_probs_unconditional + self.guidance_scale * (log_probs_conditional - log_probs_unconditional)
        
        # It's important that `scores` and `unconditional_next_token_logits` have compatible shapes.
        # scores: (batch_size, vocab_size)
        # unconditional_next_token_logits: (batch_size, vocab_size)
        if scores.shape != unconditional_next_token_logits.shape:
             raise ValueError(
                f"Shape mismatch for CFG: conditional scores shape {scores.shape}, "
                f"unconditional next token logits shape {unconditional_next_token_logits.shape}."
            )

        return combined_log_probs