from __future__ import annotations

import json
import unittest
from unittest import mock

from danbooru_upsampler.node import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    DanbooruTagsUpsamplerNode,
)
from danbooru_upsampler.service import DanbooruUpsamplerGenerationError, DanbooruUpsamplerResult


class DanbooruUpsamplerNodeTests(unittest.TestCase):
    def test_object_info_metadata_has_tooltips_and_preserves_widget_contract(self) -> None:
        input_types = DanbooruTagsUpsamplerNode.INPUT_TYPES()
        expected_required = (
            "prompt",
            "model_name",
            "tag_length",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "num_beams",
            "model_device",
            "model_backend",
            "max_new_tokens",
        )
        expected_optional = (
            "negative_prompt_tags",
            "ban_tags",
            "cfg_scale",
            "debug_logging",
        )

        self.assertEqual(tuple(input_types["required"]), expected_required)
        self.assertEqual(tuple(input_types["optional"]), expected_optional)
        self.assertEqual(
            [input_types["required"][name][1]["default"] for name in expected_required],
            [
                "1girl, solo",
                "dart-v1-sft",
                "long",
                0,
                1.0,
                30,
                1.0,
                1,
                input_types["required"]["model_device"][1]["default"],
                "ONNX (Quantized)",
                128,
            ],
        )

        all_inputs = {**input_types["required"], **input_types["optional"]}
        self.assertEqual(len(all_inputs), 15)
        for input_name, input_definition in all_inputs.items():
            with self.subTest(input_name=input_name):
                self.assertIn("tooltip", input_definition[1])
                self.assertTrue(input_definition[1]["tooltip"].strip())

        self.assertIn("ONNX", all_inputs["model_backend"][1]["tooltip"])
        self.assertIn("CFG", all_inputs["negative_prompt_tags"][1]["tooltip"])
        self.assertIn("Original", all_inputs["cfg_scale"][1]["tooltip"])
        self.assertIn("every backend", all_inputs["ban_tags"][1]["tooltip"])
        self.assertIn("CPU", all_inputs["model_device"][1]["tooltip"])

        object_info_equivalent = {
            "input": input_types,
            "output": DanbooruTagsUpsamplerNode.RETURN_TYPES,
            "output_name": DanbooruTagsUpsamplerNode.RETURN_NAMES,
            "output_tooltips": DanbooruTagsUpsamplerNode.OUTPUT_TOOLTIPS,
            "name": "DanbooruTagsUpsampler",
            "display_name": NODE_DISPLAY_NAME_MAPPINGS["DanbooruTagsUpsampler"],
            "description": DanbooruTagsUpsamplerNode.DESCRIPTION,
            "category": DanbooruTagsUpsamplerNode.CATEGORY,
        }
        json.dumps(object_info_equivalent)
        self.assertFalse(hasattr(DanbooruTagsUpsamplerNode, "WEB_DIRECTORY"))

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
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["DanbooruTagsUpsampler"], "Danbooru Tags Upsampler")
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["DanbooruTagsUpsamplerNodeRay"], "Danbooru Tags Upsampler")
        self.assertIn("Danbooru_Tags_Upsampler", DanbooruTagsUpsamplerNode.SEARCH_ALIASES)

    def test_node_exposes_frontend_discovery_metadata(self) -> None:
        self.assertIn("Danbooru", DanbooruTagsUpsamplerNode.DESCRIPTION)
        self.assertIn("danbooru", DanbooruTagsUpsamplerNode.SEARCH_ALIASES)
        self.assertIn("prompt upsampler", DanbooruTagsUpsamplerNode.SEARCH_ALIASES)
        self.assertEqual(
            DanbooruTagsUpsamplerNode.OUTPUT_TOOLTIPS,
            ("The original prompt plus generated Danbooru tag completions.",),
        )
