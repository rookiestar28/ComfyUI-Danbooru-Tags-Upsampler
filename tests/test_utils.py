from __future__ import annotations

import unittest

from danbooru_upsampler.dart.utils import _get_tag_pattern, get_valid_tag_list


class BanPatternTests(unittest.TestCase):
    def test_wildcards_preserve_every_literal_fragment(self) -> None:
        cases = (
            ("blue*eyes", "blue_eyes", True),
            ("blue*eyes", "blueeyes", True),
            ("blue*eyes", "blue_hair", False),
            ("*eyes", "green_eyes", True),
            ("*eyes", "red_hair", False),
            ("blue*", "blue_hair", True),
            ("a*b*c", "axbxc", True),
            ("a*b*c", "axyc", False),
            ("*a**b*", "xabx", True),
            ("*a**b*", "xax", False),
            ("**", "any_tag", True),
        )
        for pattern, token, expected in cases:
            with self.subTest(pattern=pattern, token=token):
                self.assertEqual(bool(_get_tag_pattern(pattern).fullmatch(token)), expected)

    def test_regex_metacharacters_and_backslashes_are_literal(self) -> None:
        for tag in ("a.b[1](x)?", r"character\(series\)", "a+b|c^$", "tag{2}"):
            with self.subTest(tag=tag):
                pattern = _get_tag_pattern(tag)
                self.assertIsNotNone(pattern.fullmatch(tag))
                self.assertIsNone(pattern.fullmatch(tag + "suffix"))
        pattern = _get_tag_pattern("a.b*eyes")
        self.assertIsNotNone(pattern.fullmatch("a.b_eyes"))
        self.assertIsNone(pattern.fullmatch("axb_eyes"))

    def test_empty_comma_fields_are_ignored(self) -> None:
        self.assertEqual(get_valid_tag_list(" , , \t"), [])
        self.assertEqual(get_valid_tag_list(" cat, , *eyes , cat "), ["cat", "*eyes", "cat"])
