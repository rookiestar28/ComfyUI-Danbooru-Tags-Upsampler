from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import danbooru_upsampler.dart.analyzer as analyzer_module
from danbooru_upsampler.dart.analyzer import DartAnalyzer


def _write_tag_resources(tags_dir: Path) -> None:
    (tags_dir / "copyright.txt").write_text("original\nvocaloid\n", encoding="utf-8")
    (tags_dir / "character.txt").write_text("hatsune miku\n", encoding="utf-8")
    (tags_dir / "quality.txt").write_text("masterpiece\n", encoding="utf-8")


class DartAnalyzerResourceTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_cache = getattr(analyzer_module, "clear_analyzer_resource_cache", lambda: None)
        clear_cache()

    def tearDown(self) -> None:
        clear_cache = getattr(analyzer_module, "clear_analyzer_resource_cache", lambda: None)
        clear_cache()

    def test_missing_required_tag_resource_raises_instead_of_degrading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(FileNotFoundError):
                DartAnalyzer(
                    tags_dir_path=Path(temporary_dir),
                    vocab=["1girl"],
                    special_vocab=[],
                )

    def test_reuses_immutable_resources_for_identical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tags_dir = Path(temporary_dir)
            _write_tag_resources(tags_dir)
            with mock.patch.object(
                analyzer_module,
                "load_tags_in_file",
                wraps=analyzer_module.load_tags_in_file,
            ) as load_tags:
                first = DartAnalyzer(
                    tags_dir_path=tags_dir,
                    vocab=["1girl", "solo"],
                    special_vocab=["<|input_end|>"],
                )
                second = DartAnalyzer(
                    tags_dir_path=tags_dir,
                    vocab=["1girl", "solo"],
                    special_vocab=["<|input_end|>"],
                )

            self.assertEqual(load_tags.call_count, 3)
            self.assertIsInstance(first.copyright_tags, frozenset)
            self.assertIsInstance(first.vocab, frozenset)
            self.assertEqual(first.copyright_tags, second.copyright_tags)
            self.assertEqual(first.vocab, second.vocab)

            prompt = "rating:general, original, hatsune miku, masterpiece, 1girl, unknown"
            self.assertEqual(first.analyze(prompt), second.analyze(prompt))

    def test_corrupt_required_tag_resource_raises_typed_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tags_dir = Path(temporary_dir)
            _write_tag_resources(tags_dir)
            (tags_dir / "quality.txt").write_bytes(b"\xff\xfe")

            with self.assertRaises(UnicodeError):
                DartAnalyzer(tags_dir_path=tags_dir, vocab=["1girl"], special_vocab=[])

    def test_cached_resources_are_safe_under_concurrent_analyzers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tags_dir = Path(temporary_dir)
            _write_tag_resources(tags_dir)
            results: list[object] = []
            errors: list[BaseException] = []

            def analyze() -> None:
                try:
                    instance = DartAnalyzer(
                        tags_dir_path=tags_dir,
                        vocab=["1girl", "solo"],
                        special_vocab=[],
                    )
                    results.append(instance.analyze("original, hatsune miku, 1girl"))
                except BaseException as exc:  # pragma: no cover - assertion captures worker errors
                    errors.append(exc)

            workers = [threading.Thread(target=analyze) for _ in range(4)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(2.0)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 4)
            self.assertTrue(all(result == results[0] for result in results[1:]))

    def test_resource_cache_invalidates_when_a_tag_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tags_dir = Path(temporary_dir)
            _write_tag_resources(tags_dir)
            with mock.patch.object(
                analyzer_module,
                "load_tags_in_file",
                wraps=analyzer_module.load_tags_in_file,
            ) as load_tags:
                first = DartAnalyzer(tags_dir_path=tags_dir, vocab=["1girl"], special_vocab=[])
                (tags_dir / "quality.txt").write_text(
                    "masterpiece\nbest quality\n",
                    encoding="utf-8",
                )
                second = DartAnalyzer(tags_dir_path=tags_dir, vocab=["1girl"], special_vocab=[])

            self.assertEqual(load_tags.call_count, 6)
            self.assertNotEqual(first.quality_tags, second.quality_tags)
