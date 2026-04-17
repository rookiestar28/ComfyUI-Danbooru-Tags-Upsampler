from __future__ import annotations

import threading
import time
import unittest

from danbooru_upsampler.dart.generator import DartGenerator
from danbooru_upsampler.dart.settings import MODEL_BACKEND_TYPE


class DartGeneratorRuntimeLockTests(unittest.TestCase):
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
