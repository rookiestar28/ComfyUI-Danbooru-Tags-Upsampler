from __future__ import annotations

import unittest
from pathlib import Path

from danbooru_upsampler.dart.settings import DART_MODELS

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback for local smoke only.
    tomllib = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _requirements() -> set[str]:
    requirements_path = PROJECT_ROOT / "requirements.txt"
    requirements: set[str] = set()
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.add(stripped)
    return requirements


@unittest.skipIf(tomllib is None, "tomllib is unavailable on this Python runtime")
class PackageMetadataTests(unittest.TestCase):
    def test_model_allowlist_uses_approved_revisions_and_scoped_remote_code(self) -> None:
        expected = {
            "dart-v1-sft": (
                "dd5a3f34f3baa15b5266b5f5e2371a97c8ac7702",  # pragma: allowlist secret
                True,
            ),
            "dart-v2-sft": (
                "df62d486a9308fde0b4ddbf23742a18f7bc0b8e6",  # pragma: allowlist secret
                False,
            ),
            "dart-v2-moe-sft": (
                "167fdb177a6d68e2d4adca0be5f05d21f74e4d41",  # pragma: allowlist secret
                False,
            ),
        }

        self.assertEqual(set(DART_MODELS), set(expected))
        for model_name, (revision, trust_remote_code) in expected.items():
            model_info = DART_MODELS[model_name]
            self.assertEqual(model_info["revision"], revision)
            self.assertRegex(model_info["revision"], r"^[0-9a-f]{40}$")
            self.assertIs(model_info["trust_remote_code"], trust_remote_code)

    def test_legacy_host_mutating_installer_is_absent(self) -> None:
        self.assertFalse((PROJECT_ROOT / "install.py").exists())

    def test_pyproject_dependencies_match_requirements_file(self) -> None:
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(set(pyproject["project"]["dependencies"]), _requirements())

    def test_host_managed_dependencies_are_not_declared_by_node_package(self) -> None:
        requirements = _requirements()

        self.assertNotIn("torch", requirements)
        self.assertNotIn("torchvision", requirements)
        self.assertNotIn("torchaudio", requirements)
        self.assertNotIn("optimum-onnx", requirements)

    def test_comfy_registry_metadata_declares_realistic_host_floor(self) -> None:
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["requires-python"], ">=3.10")
        self.assertEqual(pyproject["tool"]["comfy"]["requires-comfyui"], ">=0.22.3")
        self.assertIn("Operating System :: OS Independent", pyproject["project"]["classifiers"])
