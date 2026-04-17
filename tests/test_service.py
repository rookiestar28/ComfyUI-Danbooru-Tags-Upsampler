from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from danbooru_upsampler.service import (
    DEFAULT_TOOLBAR_PROFILE,
    DanbooruUpsamplerAnalyzerError,
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
    def test_build_toolbar_request_uses_toolbar_profile_defaults(self) -> None:
        request = build_toolbar_request("1girl, solo")

        self.assertEqual(request.prompt, "1girl, solo")
        self.assertEqual(request.model_name, DEFAULT_TOOLBAR_PROFILE.model_name)
        self.assertEqual(request.model_backend, DEFAULT_TOOLBAR_PROFILE.model_backend)
        self.assertEqual(request.model_device, DEFAULT_TOOLBAR_PROFILE.model_device)
        self.assertEqual(request.tag_length, DEFAULT_TOOLBAR_PROFILE.tag_length)

    def test_resolve_runtime_selection_falls_back_to_quantized_onnx_when_needed(self) -> None:
        runtime = resolve_runtime_selection(
            model_name="dart-v2-sft",
            model_backend=MODEL_BACKEND_TYPE["ONNX"],
            model_device="cpu",
        )

        self.assertEqual(runtime.resolved_backend, MODEL_BACKEND_TYPE["ONNX_QUANTIZED"])
        self.assertEqual(runtime.onnx_file_name, "model_quantized.onnx")
        self.assertTrue(runtime.warnings)

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
        mocked_set_seed.assert_called_once()
        generator.generate.assert_called_once()
        analyzer.analyze.assert_called_once_with("1girl, solo")

    def test_upsample_prompt_rejects_unknown_model(self) -> None:
        with self.assertRaises(DanbooruUpsamplerInvalidRequestError):
            upsample_prompt(
                DanbooruUpsamplerRequest(
                    prompt="1girl",
                    model_name="unknown-model",
                )
            )

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
