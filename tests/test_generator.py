from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from danbooru_upsampler.dart.generator import DartGenerator
from danbooru_upsampler.dart.settings import MODEL_BACKEND_TYPE


class DartGeneratorRuntimeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_runtime_state = (
            DartGenerator._model_cache,
            getattr(DartGenerator, "_model_resolution_cache", {}),
            DartGenerator._tokenizer_cache,
            DartGenerator.dart_model,
            DartGenerator.dart_tokenizer,
            DartGenerator._current_model_key,
            DartGenerator._current_tokenizer_key,
        )
        DartGenerator._model_cache = {}
        if hasattr(DartGenerator, "_model_resolution_cache"):
            DartGenerator._model_resolution_cache = {}
        DartGenerator._tokenizer_cache = {}
        DartGenerator.dart_model = None
        DartGenerator.dart_tokenizer = None
        DartGenerator._current_model_key = None
        DartGenerator._current_tokenizer_key = None

    def tearDown(self) -> None:
        (
            DartGenerator._model_cache,
            model_resolution_cache,
            DartGenerator._tokenizer_cache,
            DartGenerator.dart_model,
            DartGenerator.dart_tokenizer,
            DartGenerator._current_model_key,
            DartGenerator._current_tokenizer_key,
        ) = self._saved_runtime_state
        if hasattr(DartGenerator, "_model_resolution_cache"):
            DartGenerator._model_resolution_cache = model_resolution_cache

    def test_runtime_guard_serializes_concurrent_access(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_device="cpu",
        )
        gate = threading.Event()
        first_entered = threading.Event()
        enter_order: list[str] = []

        def worker_one() -> None:
            with generator.runtime_guard():
                enter_order.append("first")
                first_entered.set()
                gate.wait(1.0)

        def worker_two() -> None:
            first_entered.wait(1.0)
            with generator.runtime_guard():
                enter_order.append("second")

        thread_one = threading.Thread(target=worker_one)
        thread_two = threading.Thread(target=worker_two)
        thread_one.start()
        thread_two.start()

        self.assertTrue(first_entered.wait(1.0))
        time.sleep(0.1)
        self.assertEqual(enter_order, ["first"])

        gate.set()
        thread_one.join(1.0)
        thread_two.join(1.0)

        self.assertEqual(enter_order, ["first", "second"])

    def test_onnx_generation_routes_bad_words_ids_to_model(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ONNX_QUANTIZED"],
            model_device="cpu",
        )
        tokenizer = mock.Mock()
        tokenizer.encode_plus.return_value = SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
        tokenizer.eos_token_id = 1
        tokenizer.decode.return_value = "smile"
        model = mock.Mock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        bad_words_ids = [[123], [456]]

        with (
            mock.patch.object(generator, "load_tokenizer_if_needed"),
            mock.patch.object(generator, "load_model_if_needed"),
            mock.patch.object(DartGenerator, "dart_tokenizer", tokenizer),
            mock.patch.object(DartGenerator, "dart_model", model),
        ):
            generator.generate("<prompt>", bad_words_ids=bad_words_ids)

        self.assertIn("bad_words_ids", model.generate.call_args.kwargs)
        self.assertEqual(model.generate.call_args.kwargs["bad_words_ids"], bad_words_ids)

    def test_onnx_generation_rejects_cfg_instead_of_silently_skipping_it(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ONNX_QUANTIZED"],
            model_device="cpu",
        )
        tokenizer = mock.Mock()
        tokenizer.encode_plus.return_value = SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
        tokenizer.eos_token_id = 1
        tokenizer.decode.return_value = "smile"
        model = mock.Mock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3]])

        with (
            mock.patch.object(generator, "load_tokenizer_if_needed"),
            mock.patch.object(generator, "load_model_if_needed"),
            mock.patch.object(DartGenerator, "dart_tokenizer", tokenizer),
            mock.patch.object(DartGenerator, "dart_model", model),
        ):
            with self.assertRaisesRegex(ValueError, "CFG"):
                generator.generate("<prompt>", negative_prompt="<negative>", cfg_scale=1.5)

        model.generate.assert_not_called()

    def test_model_cache_identity_distinguishes_device_and_onnx_artifact(self) -> None:
        generators = (
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ONNX"],
                model_device="cpu",
                onnx_file_name="model.onnx",
            ),
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ONNX"],
                model_device="cuda",
                onnx_file_name="model.onnx",
            ),
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ONNX"],
                model_device="cpu",
                onnx_file_name="model_quantized.onnx",
            ),
        )
        loaded_models: list[object] = []

        def fake_load() -> None:
            model = object()
            loaded_models.append(model)
            DartGenerator.dart_model = model  # type: ignore[assignment]

        for generator in generators:
            with mock.patch.object(generator, "_load_dart_model", side_effect=fake_load):
                generator.load_model_if_needed()

        self.assertEqual(len(loaded_models), 3)
        self.assertEqual(len(DartGenerator._model_cache), 3)

    def test_model_cache_identity_distinguishes_approved_revision(self) -> None:
        generators = (
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
                model_revision="revision-a",
            ),
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
                model_revision="revision-b",
            ),
        )
        loaded_models: list[object] = []

        def fake_load() -> None:
            model = object()
            loaded_models.append(model)
            DartGenerator.dart_model = model  # type: ignore[assignment]

        for generator in generators:
            with mock.patch.object(generator, "_load_dart_model", side_effect=fake_load):
                generator.load_model_if_needed()

        self.assertEqual(len(loaded_models), 2)
        self.assertEqual(len(DartGenerator._model_cache), 2)

    def test_all_hub_loads_receive_revision_and_scoped_tokenizer_trust(self) -> None:
        revision = "dd5a3f34f3baa15b5266b5f5e2371a97c8ac7702"  # pragma: allowlist secret
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_revision=revision,
            tokenizer_trust_remote_code=True,
        )
        model = mock.Mock()
        model.device = torch.device("cpu")

        with mock.patch(
            "danbooru_upsampler.dart.generator.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as load_model:
            generator._load_dart_model()
        with mock.patch(
            "danbooru_upsampler.dart.generator.AutoTokenizer.from_pretrained",
            return_value=mock.Mock(),
        ) as load_tokenizer:
            generator._load_dart_tokenizer()

        load_model.assert_called_once_with(generator.model_name, revision=revision)
        load_tokenizer.assert_called_once_with(
            generator.tokenizer_name,
            revision=revision,
            trust_remote_code=True,
        )

        onnx_generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ONNX_QUANTIZED"],
            model_revision=revision,
            tokenizer_trust_remote_code=True,
            onnx_file_name="model_quantized.onnx",
        )
        onnx_model = mock.Mock()
        onnx_model.device = torch.device("cpu")
        with mock.patch(
            "danbooru_upsampler.dart.generator.ORTModelForCausalLM.from_pretrained",
            return_value=onnx_model,
        ) as load_onnx:
            onnx_generator._load_dart_model()

        load_onnx.assert_called_once_with(
            onnx_generator.model_name,
            file_name="model_quantized.onnx",
            revision=revision,
        )

    def test_tokenizer_cache_identity_includes_revision_and_trust_policy(self) -> None:
        generators = (
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
                model_revision="revision-a",
                tokenizer_trust_remote_code=True,
            ),
            DartGenerator(
                model_name="p1atdev/dart-v1-sft",
                tokenizer_name="p1atdev/dart-v1-sft",
                model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
                model_revision="revision-b",
                tokenizer_trust_remote_code=False,
            ),
        )

        with mock.patch(
            "danbooru_upsampler.dart.generator.AutoTokenizer.from_pretrained",
            side_effect=(mock.Mock(), mock.Mock()),
        ) as load_tokenizer:
            for generator in generators:
                generator.load_tokenizer_if_needed()

        self.assertEqual(load_tokenizer.call_count, 2)
        self.assertEqual(len(DartGenerator._tokenizer_cache), 2)

    def test_model_load_records_cpu_as_actual_device_after_cuda_fallback(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_device="cuda",
        )

        class FallbackModel:
            device = torch.device("cpu")

            def to(self, device: str) -> "FallbackModel":
                if str(device).startswith("cuda"):
                    raise RuntimeError("CUDA allocation failed")
                self.device = torch.device("cpu")
                return self

        with mock.patch(
            "danbooru_upsampler.dart.generator.AutoModelForCausalLM.from_pretrained",
            return_value=FallbackModel(),
        ):
            generator._load_dart_model()

        get_actual_device = getattr(generator, "get_actual_device", lambda: generator.model_device)
        self.assertEqual(get_actual_device(), "cpu")

    def test_cuda_fallback_cache_uses_actual_device_and_reuses_resolution(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_device="cuda",
            model_revision="revision-a",
        )

        class FallbackModel:
            device = torch.device("cpu")

            def to(self, device: str) -> "FallbackModel":
                if str(device).startswith("cuda"):
                    raise RuntimeError("CUDA allocation failed")
                self.device = torch.device("cpu")
                return self

        with mock.patch(
            "danbooru_upsampler.dart.generator.AutoModelForCausalLM.from_pretrained",
            return_value=FallbackModel(),
        ) as load_model:
            generator.load_model_if_needed()
            generator.load_model_if_needed()

        cached_devices = {cache_key[3] for cache_key in DartGenerator._model_cache}
        self.assertEqual(cached_devices, {"cpu"})
        self.assertEqual(load_model.call_count, 1)

    def test_stale_current_keys_do_not_suppress_missing_runtime_objects(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ORIGINAL"],
            model_device="cpu",
            model_revision="revision-a",
        )
        DartGenerator._current_model_key = generator._model_cache_key()
        DartGenerator._current_tokenizer_key = generator._tokenizer_cache_key()
        DartGenerator.dart_model = None
        DartGenerator.dart_tokenizer = None

        def load_model() -> None:
            DartGenerator.dart_model = SimpleNamespace(device=torch.device("cpu"))  # type: ignore[assignment]

        def load_tokenizer() -> None:
            DartGenerator.dart_tokenizer = mock.Mock()

        with (
            mock.patch.object(generator, "_load_dart_model", side_effect=load_model) as model_loader,
            mock.patch.object(
                generator,
                "_load_dart_tokenizer",
                side_effect=load_tokenizer,
            ) as tokenizer_loader,
        ):
            generator.load_model_if_needed()
            generator.load_tokenizer_if_needed()

        model_loader.assert_called_once_with()
        tokenizer_loader.assert_called_once_with()

    def test_onnx_actual_device_prefers_active_execution_provider(self) -> None:
        generator = DartGenerator(
            model_name="p1atdev/dart-v1-sft",
            tokenizer_name="p1atdev/dart-v1-sft",
            model_backend=MODEL_BACKEND_TYPE["ONNX"],
            model_device="cuda",
        )

        class FakeSession:
            @staticmethod
            def get_providers() -> list[str]:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        class FakeOrtModel:
            device = torch.device("cpu")
            model = FakeSession()

        DartGenerator.dart_model = FakeOrtModel()  # type: ignore[assignment]

        self.assertEqual(generator.get_actual_device(), "cuda")
