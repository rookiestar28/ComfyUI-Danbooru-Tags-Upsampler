from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import unittest
from dataclasses import replace
from unittest import mock

from danbooru_upsampler.service import (
    DEFAULT_TOOLBAR_PROFILE,
    DanbooruUpsamplerAnalyzerError,
    DanbooruUpsamplerError,
    DanbooruUpsamplerGenerationError,
    DanbooruUpsamplerInvalidRequestError,
    DanbooruUpsamplerRequest,
    DanbooruUpsamplerRuntimeInitializationError,
    build_toolbar_request,
    resolve_runtime_selection,
    upsample_prompt,
)
from danbooru_upsampler.dart.analyzer import ImagePromptAnalyzingResult
from danbooru_upsampler.dart.settings import MODEL_BACKEND_TYPE
import danbooru_upsampler.service as service_module


def _analysis_result(*, general: str = "1girl, solo") -> ImagePromptAnalyzingResult:
    return ImagePromptAnalyzingResult(
        rating_parent="rating:sfw",
        rating_child="rating:general",
        copyright="",
        character="",
        general=general,
        quality="",
        unknown="",
    )


class DanbooruUpsamplerServiceTests(unittest.TestCase):
    def test_backend_capabilities_report_cfg_and_ban_tag_support(self) -> None:
        resolver = getattr(service_module, "resolve_backend_capabilities", lambda _backend: None)

        original = resolver(MODEL_BACKEND_TYPE["ORIGINAL"])
        onnx = resolver(MODEL_BACKEND_TYPE["ONNX"])
        quantized = resolver(MODEL_BACKEND_TYPE["ONNX_QUANTIZED"])

        self.assertIsNotNone(original)
        self.assertTrue(original.supports_cfg)
        self.assertTrue(original.supports_ban_tags)
        self.assertFalse(onnx.supports_cfg)
        self.assertTrue(onnx.supports_ban_tags)
        self.assertFalse(quantized.supports_cfg)
        self.assertTrue(quantized.supports_ban_tags)
        json.dumps(dataclasses.asdict(original))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            original.supports_cfg = False

    def test_build_toolbar_request_uses_toolbar_profile_defaults(self) -> None:
        request = build_toolbar_request("1girl, solo")

        self.assertEqual(request.prompt, "1girl, solo")
        self.assertEqual(request.model_name, DEFAULT_TOOLBAR_PROFILE.model_name)
        self.assertEqual(request.model_backend, DEFAULT_TOOLBAR_PROFILE.model_backend)
        self.assertEqual(request.model_device, DEFAULT_TOOLBAR_PROFILE.model_device)
        self.assertEqual(request.tag_length, DEFAULT_TOOLBAR_PROFILE.tag_length)

    def test_build_toolbar_request_wraps_invalid_seed_as_typed_request_error(self) -> None:
        with self.assertRaises(DanbooruUpsamplerInvalidRequestError):
            build_toolbar_request("1girl, solo", seed="not-an-int")

    def test_resolve_runtime_selection_falls_back_to_quantized_onnx_when_needed(self) -> None:
        runtime = resolve_runtime_selection(
            model_name="dart-v2-sft",
            model_backend=MODEL_BACKEND_TYPE["ONNX"],
            model_device="cpu",
        )

        self.assertEqual(runtime.resolved_backend, MODEL_BACKEND_TYPE["ONNX_QUANTIZED"])
        self.assertEqual(runtime.onnx_file_name, "model_quantized.onnx")
        self.assertTrue(runtime.warnings)

    def test_resolve_runtime_selection_returns_only_allowlisted_trust_policy(self) -> None:
        v1 = resolve_runtime_selection(
            model_name="dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_device="cpu",
        )
        v2 = resolve_runtime_selection(
            model_name="dart-v2-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_device="cpu",
        )

        self.assertEqual(v1.model_revision, "dd5a3f34f3baa15b5266b5f5e2371a97c8ac7702")  # pragma: allowlist secret
        self.assertTrue(v1.trust_remote_code)
        self.assertEqual(v2.model_revision, "df62d486a9308fde0b4ddbf23742a18f7bc0b8e6")  # pragma: allowlist secret
        self.assertFalse(v2.trust_remote_code)
        self.assertNotIn(
            "model_revision",
            {field.name for field in dataclasses.fields(DanbooruUpsamplerRequest)},
        )

    def test_upsample_prompt_returns_structured_result(self) -> None:
        request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ONNX_QUANTIZED"],
            ban_tags="english text",
        )

        with (
            mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls,
            mock.patch("danbooru_upsampler.service.DartAnalyzer") as mocked_analyzer_cls,
            mock.patch("danbooru_upsampler.service.set_seed") as mocked_set_seed,
        ):
            generator = mocked_generator_cls.return_value
            generator.runtime_guard.return_value = contextlib.nullcontext()
            generator.get_vocab_list.return_value = ["1girl", "solo", "smile"]
            generator.get_special_vocab_list.return_value = []
            generator.compose_prompt.return_value = "<prompt>"
            generator.get_bad_words_ids.return_value = [[123]]
            generator.generate.return_value = "smile, blue eyes"
            analyzer = mocked_analyzer_cls.return_value
            analyzer.analyze.return_value = _analysis_result()

            result = upsample_prompt(request)

        self.assertEqual(result.final_prompt, "1girl, solo, smile, blue eyes")
        self.assertEqual(result.generated_suffix, "smile, blue eyes")
        self.assertEqual(result.resolved_backend, MODEL_BACKEND_TYPE["ONNX_QUANTIZED"])
        self.assertEqual(
            result.model_revision,
            "dd5a3f34f3baa15b5266b5f5e2371a97c8ac7702",  # pragma: allowlist secret
        )
        mocked_set_seed.assert_called_once()
        generator.generate.assert_called_once()
        analyzer.analyze.assert_called_once_with("1girl, solo")
        generator_kwargs = mocked_generator_cls.call_args.kwargs
        self.assertEqual(
            generator_kwargs["model_revision"],
            "dd5a3f34f3baa15b5266b5f5e2371a97c8ac7702",  # pragma: allowlist secret
        )
        self.assertTrue(generator_kwargs["tokenizer_trust_remote_code"])

    def test_upsample_prompt_reports_generator_actual_device_after_fallback(self) -> None:
        request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cuda",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
        )

        with (
            mock.patch("danbooru_upsampler.service.torch.cuda.is_available", return_value=True),
            mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls,
            mock.patch("danbooru_upsampler.service.DartAnalyzer") as mocked_analyzer_cls,
            mock.patch("danbooru_upsampler.service.set_seed"),
        ):
            generator = mocked_generator_cls.return_value
            generator.runtime_guard.return_value = contextlib.nullcontext()
            generator.get_actual_device.return_value = "cpu"
            generator.get_vocab_list.return_value = ["1girl", "solo", "smile"]
            generator.get_special_vocab_list.return_value = []
            generator.compose_prompt.return_value = "<prompt>"
            generator.get_bad_words_ids.return_value = None
            generator.generate.return_value = "smile"
            mocked_analyzer_cls.return_value.analyze.return_value = _analysis_result()

            result = upsample_prompt(request)

        self.assertEqual(result.resolved_device, "cpu")

    def test_upsample_prompt_rejects_unknown_model(self) -> None:
        with self.assertRaises(DanbooruUpsamplerInvalidRequestError):
            upsample_prompt(
                DanbooruUpsamplerRequest(
                    prompt="1girl",
                    model_name="unknown-model",
                )
            )

    def test_upsample_prompt_wraps_invalid_numeric_request_values(self) -> None:
        base_request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ONNX_QUANTIZED"],
        )

        invalid_cases = {
            "cfg_scale": "bad-float",
            "seed": "bad-int",
            "max_new_tokens": "bad-int",
            "temperature": "bad-float",
            "top_p": "bad-float",
            "top_k": "bad-int",
            "num_beams": "bad-int",
        }

        for field_name, invalid_value in invalid_cases.items():
            with self.subTest(field_name=field_name):
                with mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls:
                    with self.assertRaises(DanbooruUpsamplerInvalidRequestError):
                        upsample_prompt(replace(base_request, **{field_name: invalid_value}))
                    mocked_generator_cls.assert_not_called()

    def test_upsample_prompt_rejects_out_of_range_values_before_runtime_construction(self) -> None:
        base_request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
        )
        invalid_cases = {
            "seed": (-1, 2**32),
            "temperature": (0.0, 5.01),
            "top_k": (-1, 1001),
            "top_p": (-0.01, 1.01),
            "num_beams": (0, 21),
            "max_new_tokens": (7, 513),
            "cfg_scale": (0.99, 10.01),
        }

        for field_name, invalid_values in invalid_cases.items():
            for invalid_value in invalid_values:
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    with mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls:
                        with self.assertRaises(DanbooruUpsamplerInvalidRequestError):
                            upsample_prompt(replace(base_request, **{field_name: invalid_value}))
                        mocked_generator_cls.assert_not_called()

    def test_upsample_prompt_rejects_non_finite_float_values_before_runtime_construction(self) -> None:
        base_request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
        )

        for field_name in ("temperature", "top_p", "cfg_scale"):
            for invalid_value in (math.nan, math.inf, -math.inf):
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    with mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls:
                        with self.assertRaises(DanbooruUpsamplerInvalidRequestError):
                            upsample_prompt(replace(base_request, **{field_name: invalid_value}))
                        mocked_generator_cls.assert_not_called()

    def test_upsample_prompt_rejects_cfg_request_for_onnx_before_runtime_construction(self) -> None:
        request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            negative_prompt_tags="blurry",
            cfg_scale=1.5,
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ONNX_QUANTIZED"],
        )

        with mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls:
            with self.assertRaises(DanbooruUpsamplerError) as caught:
                upsample_prompt(request)

        self.assertEqual(caught.exception.code, "unsupported_feature")
        mocked_generator_cls.assert_not_called()

    def test_upsample_prompt_wraps_runtime_failures(self) -> None:
        request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
        )

        with mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls:
            generator = mocked_generator_cls.return_value
            generator.runtime_guard.return_value = contextlib.nullcontext()
            generator.load_model_if_needed.side_effect = RuntimeError("boom")

            with self.assertRaises(DanbooruUpsamplerRuntimeInitializationError):
                upsample_prompt(request)

    def test_upsample_prompt_wraps_analyzer_failures(self) -> None:
        request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
        )

        with (
            mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls,
            mock.patch("danbooru_upsampler.service.DartAnalyzer") as mocked_analyzer_cls,
        ):
            generator = mocked_generator_cls.return_value
            generator.runtime_guard.return_value = contextlib.nullcontext()
            generator.get_vocab_list.return_value = ["1girl"]
            generator.get_special_vocab_list.return_value = []
            mocked_analyzer_cls.return_value.analyze.side_effect = RuntimeError("parse failure")

            with self.assertRaises(DanbooruUpsamplerAnalyzerError):
                upsample_prompt(request)

    def test_upsample_prompt_wraps_generation_failures(self) -> None:
        request = DanbooruUpsamplerRequest(
            prompt="1girl, solo",
            model_name="dart-v1-sft",
            model_device="cpu",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
        )

        with (
            mock.patch("danbooru_upsampler.service.DartGenerator") as mocked_generator_cls,
            mock.patch("danbooru_upsampler.service.DartAnalyzer") as mocked_analyzer_cls,
            mock.patch("danbooru_upsampler.service.set_seed"),
        ):
            generator = mocked_generator_cls.return_value
            generator.runtime_guard.return_value = contextlib.nullcontext()
            generator.get_vocab_list.return_value = ["1girl"]
            generator.get_special_vocab_list.return_value = []
            generator.compose_prompt.return_value = "<prompt>"
            generator.get_bad_words_ids.return_value = None
            generator.generate.side_effect = RuntimeError("generate failed")
            mocked_analyzer_cls.return_value.analyze.return_value = _analysis_result()

            with self.assertRaises(DanbooruUpsamplerGenerationError):
                upsample_prompt(request)
