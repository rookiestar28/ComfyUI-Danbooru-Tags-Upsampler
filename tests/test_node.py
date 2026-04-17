from __future__ import annotations

import unittest
from unittest import mock

from danbooru_upsampler.node import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    DanbooruTagsUpsamplerNode,
)
from danbooru_upsampler.service import DanbooruUpsamplerGenerationError, DanbooruUpsamplerResult


class DanbooruUpsamplerNodeTests(unittest.TestCase):
    def test_node_returns_service_result_prompt(self) -> None:
        node = DanbooruTagsUpsamplerNode()

        with mock.patch(
            "danbooru_upsampler.node.upsample_prompt",
            return_value=DanbooruUpsamplerResult(
                final_prompt="1girl, solo, smile",
                generated_suffix="smile",
                model_name="dart-v1-sft",
                model_repo="p1atdev/dart-v1-sft",
                requested_backend="Original",
                resolved_backend="Original",
                resolved_device="cpu",
                tag_length="long",
            ),
        ):
            result = node.upsample(
                prompt="1girl, solo",
                model_name="dart-v1-sft",
                tag_length="long",
                seed=0,
                temperature=1.0,
                top_k=30,
                top_p=1.0,
                num_beams=1,
                model_device="cpu",
                model_backend="Original",
                max_new_tokens=128,
            )

        self.assertEqual(result, ("1girl, solo, smile",))

    def test_node_raises_runtime_error_on_service_failure(self) -> None:
        node = DanbooruTagsUpsamplerNode()

        with mock.patch(
            "danbooru_upsampler.node.upsample_prompt",
            side_effect=DanbooruUpsamplerGenerationError("generation failed"),
        ):
            with self.assertRaises(RuntimeError):
                node.upsample(
                    prompt="1girl, solo",
                    model_name="dart-v1-sft",
                    tag_length="long",
                    seed=0,
                    temperature=1.0,
                    top_k=30,
                    top_p=1.0,
                    num_beams=1,
                    model_device="cpu",
                    model_backend="Original",
                    max_new_tokens=128,
                )

    def test_node_mappings_include_canonical_and_legacy_aliases(self) -> None:
        self.assertIs(NODE_CLASS_MAPPINGS["DanbooruTagsUpsampler"], DanbooruTagsUpsamplerNode)
        self.assertIs(NODE_CLASS_MAPPINGS["DanbooruTagsUpsamplerNodeRay"], DanbooruTagsUpsamplerNode)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["DanbooruTagsUpsampler"], "Danbooru_Tags_Upsampler")
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["DanbooruTagsUpsamplerNodeRay"], "Danbooru_Tags_Upsampler")
