from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

from toml_test_support import load_toml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_PREFIXES = (".planning/", "reference/", ".sessions/")
COMFYIGNORE_PREFIXES = (".github/", "tests/")
COMFYIGNORE_FILES = {
    ".comfyignore",
    ".gitignore",
    ".pre-commit-config.yaml",
    "AGENTS.md",
    "ROADMAP.md",
}
EXPECTED_PAYLOAD_ROOTS = {
    "LICENSE",
    "README.md",
    "__init__.py",
    "danbooru_upsampler",
    "pyproject.toml",
    "requirements.txt",
    "tags",
}


class ReleasePayloadTests(unittest.TestCase):
    def test_release_candidate_uses_new_three_part_semver(self) -> None:
        pyproject = load_toml(PROJECT_ROOT / "pyproject.toml")
        version = pyproject["project"]["version"]

        self.assertEqual(version, "2.4.0")
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertEqual(pyproject["tool"]["comfy"]["includes"], [])

    def test_comfyignore_excludes_development_and_internal_paths(self) -> None:
        comfyignore_path = PROJECT_ROOT / ".comfyignore"
        self.assertTrue(comfyignore_path.is_file())
        patterns = {
            line.strip()
            for line in comfyignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertTrue(set(COMFYIGNORE_PREFIXES).issubset(patterns))
        self.assertTrue(COMFYIGNORE_FILES.issubset(patterns))
        self.assertTrue(set(PRIVATE_PREFIXES).issubset(patterns))

    def test_registry_payload_candidate_has_only_runtime_roots(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = {
            path.replace("\\", "/")
            for path in result.stdout.splitlines()
            if (PROJECT_ROOT / path).is_file()
        }
        payload = {
            path
            for path in candidates
            if path not in COMFYIGNORE_FILES
            and not path.startswith(COMFYIGNORE_PREFIXES + PRIVATE_PREFIXES)
        }
        payload_roots = {path.split("/", 1)[0] for path in payload}

        self.assertEqual(payload_roots, EXPECTED_PAYLOAD_ROOTS)
        self.assertFalse(any(path.startswith(PRIVATE_PREFIXES) for path in payload))
        self.assertFalse(any(re.search(r"(?i)(prompt|command[_-]?log)", path) for path in payload))

    def test_public_compatibility_matrix_is_evidence_bounded(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("ComfyUI 0.31.0", readme)
        self.assertIn("packaged frontend 1.48.7", readme)
        self.assertIn("standalone frontend 1.50.3", readme)
        self.assertIn("Comfy Desktop 1.0.37", readme)
        self.assertIn("channel-resolved", readme)
        self.assertIn("DanbooruTagsUpsamplerNodeRay", readme)
        self.assertIn("No live model download", readme)
        self.assertIn("2026-10-04", readme)
        self.assertIn("ComfyUI 0.22.3", readme)
        self.assertIn("ComfyUI 0.38.0 source snapshot", readme)
        self.assertIn("packaged frontend 1.53.6", readme)
        self.assertIn("official frontend release assets v1.56.2", readme)
        self.assertIn("test samples, not version locks", readme)
        self.assertIn("do not add live inference claims", readme)
        self.assertIn("an installed Desktop runtime was not tested", readme)


if __name__ == "__main__":
    unittest.main()
