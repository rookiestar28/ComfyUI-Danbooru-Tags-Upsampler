from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
FULL_ACTION_REF = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s*#.*)?$")


class WorkflowSecurityTests(unittest.TestCase):
    def test_ci_runs_deterministic_replacement_gate_without_secrets(self) -> None:
        ci_path = WORKFLOWS_DIR / "ci.yml"
        self.assertTrue(ci_path.is_file())
        ci = ci_path.read_text(encoding="utf-8")

        self.assertIn("pull_request:", ci)
        self.assertRegex(ci, r"(?m)^\s+-\s+dev\s*$")
        self.assertRegex(ci, r"(?m)^\s+permissions:\s*$")
        self.assertRegex(ci, r"(?m)^\s+contents:\s+read\s*$")
        self.assertIn("timeout-minutes:", ci)
        self.assertIn("pre-commit run detect-secrets --all-files", ci)
        self.assertIn("pre-commit run --all-files --show-diff-on-failure", ci)
        self.assertIn("python -m compileall danbooru_upsampler __init__.py", ci)
        self.assertIn("python -m unittest discover", ci)
        self.assertIn("E2E replacement lane", ci)
        self.assertNotIn("REGISTRY_ACCESS_TOKEN", ci)
        self.assertNotIn("comfy node publish", ci)

    def test_publish_is_main_only_least_privilege_and_uses_pinned_cli(self) -> None:
        publish = (WORKFLOWS_DIR / "publish.yml").read_text(encoding="utf-8")

        self.assertRegex(publish, r"(?m)^\s+permissions:\s*$")
        self.assertRegex(publish, r"(?m)^\s+contents:\s+read\s*$")
        self.assertNotIn("issues: write", publish)
        self.assertNotRegex(publish, r"(?m)^\s+-\s+(?:dev|master)\s*$")
        self.assertIn("timeout-minutes:", publish)
        self.assertIn("persist-credentials: false", publish)
        self.assertNotIn("submodules: true", publish)
        self.assertNotIn("Comfy-Org/publish-node-action", publish)
        self.assertIn("comfy-cli==1.15.0", publish)
        self.assertEqual(publish.count("REGISTRY_ACCESS_TOKEN"), 1)
        self.assertIn("COMFY_REGISTRY_TOKEN", publish)

    def test_every_workflow_action_is_pinned_to_reviewed_full_sha(self) -> None:
        expected_refs = {
            "actions/checkout": "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",  # pragma: allowlist secret
            "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",  # pragma: allowlist secret
        }
        observed_actions: set[str] = set()

        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            for line_number, line in enumerate(
                workflow_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if "uses:" not in line:
                    continue
                match = FULL_ACTION_REF.match(line)
                self.assertIsNotNone(
                    match,
                    f"{workflow_path.name}:{line_number} must use a full action SHA",
                )
                action_name = line.split("uses:", 1)[1].strip().split("@", 1)[0]
                observed_actions.add(action_name)
                self.assertEqual(match.group(1), expected_refs[action_name])

        self.assertEqual(observed_actions, set(expected_refs))


if __name__ == "__main__":
    unittest.main()
