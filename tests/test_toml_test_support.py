from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest import mock

from toml_test_support import load_toml


class TomlTestSupportTests(unittest.TestCase):
    def test_native_parser_takes_precedence(self) -> None:
        parser = types.SimpleNamespace(loads=mock.Mock(return_value={"native": True}))
        with mock.patch("toml_test_support.importlib.import_module", return_value=parser) as importer, mock.patch.object(Path, "read_text", return_value="native=true"):
            self.assertEqual(load_toml(Path("synthetic.toml")), {"native": True})
        importer.assert_called_once_with("tomllib")
        parser.loads.assert_called_once_with("native=true")

    def test_fallback_parser_is_used_when_native_is_absent(self) -> None:
        missing = ModuleNotFoundError("native unavailable", name="tomllib")
        parser = types.SimpleNamespace(loads=mock.Mock(return_value={"fallback": True}))
        with mock.patch("toml_test_support.importlib.import_module", side_effect=[missing, parser]) as importer, mock.patch.object(Path, "read_text", return_value="fallback=true"):
            self.assertEqual(load_toml(Path("synthetic.toml")), {"fallback": True})
        self.assertEqual(importer.call_args_list, [mock.call("tomllib"), mock.call("tomli")])

    def test_absent_parsers_fail_with_actionable_test_only_setup(self) -> None:
        failures = [ModuleNotFoundError("missing", name=name) for name in ("tomllib", "tomli")]
        with mock.patch("toml_test_support.importlib.import_module", side_effect=failures):
            with self.assertRaisesRegex(RuntimeError, 'python -m pip install "tomli==2.4.1"') as caught:
                load_toml(Path("synthetic.toml"))
        self.assertIs(caught.exception.__cause__, failures[1])

    def test_parser_internal_import_failure_is_not_hidden(self) -> None:
        failure = ModuleNotFoundError("parser dependency missing", name="parser_dependency")
        with mock.patch("toml_test_support.importlib.import_module", side_effect=failure) as importer:
            with self.assertRaises(ModuleNotFoundError) as caught:
                load_toml(Path("synthetic.toml"))
        self.assertIs(caught.exception, failure)
        importer.assert_called_once_with("tomllib")
