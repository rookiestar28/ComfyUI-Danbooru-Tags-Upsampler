from __future__ import annotations

import tempfile
import itertools
import threading
import unittest
from pathlib import Path
from unittest import mock

import danbooru_upsampler.dart.analyzer as analyzer_module
from danbooru_upsampler.dart.analyzer import DartAnalyzer


class RatingNormalizationTests(unittest.TestCase):
    def test_singletons_defaults_and_repeated_parents(self) -> None:
        cases = (
            ([], ("rating:sfw", "rating:general")),
            (["unknown"], ("rating:sfw", "rating:general")),
            (["sfw"], ("rating:sfw", "rating:general")),
            (["nsfw"], ("rating:nsfw", "rating:explicit")),
            (["rating:general"], ("rating:sfw", "rating:general")),
            (["rating:sensitive"], ("rating:sfw", "rating:sensitive")),
            (["rating:questionable"], ("rating:nsfw", "rating:questionable")),
            (["rating:explicit"], ("rating:nsfw", "rating:explicit")),
            (["nsfw", "nsfw"], ("rating:nsfw", "rating:explicit")),
            (["sfw", "sfw"], ("rating:sfw", "rating:general")),
            (["sfw", "nsfw"], ("rating:sfw", "rating:general")),
        )
        for tags, expected in cases:
            with self.subTest(tags=tags):
                self.assertEqual(analyzer_module.normalize_rating_tags(tags), expected)

    def test_every_parent_child_pair_is_canonical_and_consistent(self) -> None:
        children = ("rating:general", "rating:sensitive", "rating:questionable", "rating:explicit")
        for parent in ("sfw", "nsfw"):
            for index, child in enumerate(children):
                expected_child = child if (parent == "sfw") == (index < 2) else (
                    "rating:general" if parent == "sfw" else "rating:explicit")
                expected = ("rating:" + parent, expected_child)
                for tags in ([parent, child], [child, parent], [parent, child, parent, child]):
                    with self.subTest(tags=tags):
                        self.assertEqual(analyzer_module.normalize_rating_tags(tags), expected)

    def test_strongest_child_and_mixed_parents_are_order_independent(self) -> None:
        cases = (
            (["rating:general", "rating:questionable", "rating:sensitive"], ("rating:nsfw", "rating:questionable")),
            (["sfw", "nsfw", "rating:sensitive"], ("rating:nsfw", "rating:explicit")),
            (["sfw", "rating:sensitive", "rating:explicit"], ("rating:sfw", "rating:general")),
            (["nsfw", "rating:general", "rating:questionable"], ("rating:nsfw", "rating:questionable")),
        )
        for tags, expected in cases:
            for permutation in itertools.permutations(tags):
                with self.subTest(tags=permutation):
                    self.assertEqual(analyzer_module.normalize_rating_tags(list(permutation)), expected)


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
