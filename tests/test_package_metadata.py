from __future__ import annotations

import unittest
from pathlib import Path

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
