# TEST_SOP.md

### Problem-First Test Design Rule (Mandatory)

All test scripts, test harnesses, and validation flows must be designed first to reproduce real failures and catch bugs early.

The purpose of testing is to expose defects, regressions, drift, and broken assumptions before users hit them. Tests must not be designed merely to produce a green validation result, satisfy a checklist, or prove that a happy path still passes. Do not waste validation time on pass-only checks that cannot fail for the bug class under review.

Every bugfix or high-risk change must start from the question: "Which test would have caught this before release?" If the existing gate missed the bug, update the targeted test or SOP flow so the same class of bug fails deterministically next time.
## Purpose

This repository ships a ComfyUI custom node with Python-only runtime behavior. The local validation goal is to prove:

- import and packaging integrity,
- package-style root import compatibility,
- deterministic mocked unit coverage for host-delegation seams,
- no reliance on live model downloads during automated tests.

## Environment Rules

- Prefer a repo-local virtual environment: `.venv`
- Use one Python interpreter consistently for install, test, and smoke commands.
- Do not rely on automated tests that require live Hugging Face downloads or external authentication.

## Validation Sequence

### 1. Environment bootstrap

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. Import and syntax smoke

```powershell
.venv\Scripts\python -m compileall danbooru_upsampler __init__.py install.py
```

### 2b. Package-style import smoke

```powershell
.venv\Scripts\python -c "import importlib.util, pathlib, sys; root = pathlib.Path('__init__.py').resolve(); name = 'comfyui_danbooru_tags_upsampler_root'; spec = importlib.util.spec_from_file_location(name, root, submodule_search_locations=[str(root.parent)]); module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); print(sorted(module.NODE_CLASS_MAPPINGS.keys()))"
```

### 3. Unit tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py" -v
```

### 4. E2E applicability

This repository currently has no frontend application, `package.json`, npm script, Playwright suite, or browser harness. Frontend E2E is therefore non-applicable for the current Python-only custom node.

Replacement validation lane:

- `compileall` verifies Python syntax and importable files.
- package-style root import smoke verifies ComfyUI custom-node bootstrap compatibility.
- `unittest` verifies the service seam, node wrapper, registry aliases, runtime lock, and package metadata.
- manual host smoke remains the optional live ComfyUI check when node registration or bootstrap behavior changes.

Do not run `npm test` for this repository unless a frontend harness is intentionally added in a future change. If a frontend harness is added, follow `tests/E2E_TESTING_NOTICE.md` and `tests/E2E_TESTING_SOP.md`.

## Mocking Rules

- Unit tests must mock `DartGenerator`, `DartAnalyzer`, and any heavy runtime/model-loading path.
- Automated tests must not trigger real Hugging Face downloads, ONNX fetches, or CUDA-only execution.
- Concurrency tests should validate lock/ordering behavior with fakes or mocks rather than real model threads.

## Manual Host Smoke

Run this when changing node registration or wrapper/runtime behavior:

1. Install the custom node into a local ComfyUI `custom_nodes` directory.
2. Start ComfyUI and confirm the node loads without import-time failure.
3. Verify the node appears with the expected canonical display name and legacy compatibility mapping.
4. Execute one mocked or lightweight prompt-upsample flow if the local environment already has the required runtime dependencies cached.

## Evidence Recording

Implementation records must capture:

- date,
- operating environment,
- exact commands run,
- pass/fail result for each validation step,
- any skipped steps plus the reason.
<!-- ROOKIEUI-GLOBAL-TEST-SOP-RULES:START -->
## RookieUI-Derived Global Testing Rules

These rules preserve this repository's existing test lanes while adding the shared testing baseline used across this workspace.

### Required Reading Order

1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`

### Acceptance Rule

A change is not accepted until required checks pass and evidence is recorded. Existing repo-specific gates remain authoritative; this section adds the shared minimum expectations.

Required shared gate:

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend/unit tests through the repo's documented runner, preferring `scripts/run_unittests.py` when present
4. frontend/E2E tests through the repo's documented Playwright or harness lane, usually `npm test` when a Node harness exists
5. targeted type/static validation when the changed surface has a typed frontend or equivalent static contract

If a repo has no frontend/E2E harness, the SOP must state the non-applicability and identify the replacement smoke, unit, or integration lane that catches the same user-facing risk.

### Problem-First Test Design Rule

All test scripts, test harnesses, and validation flows must be designed first to reproduce real failures and catch bugs early.

The purpose of testing is to expose defects, regressions, drift, and broken assumptions before users hit them. Tests must not be designed merely to produce a green validation result, satisfy a checklist, or prove that a happy path still passes. Do not waste validation time on pass-only checks that cannot fail for the bug class under review.

Every bugfix or high-risk change must start from the question: "Which test would have caught this before release?" If the existing gate missed the bug, update the targeted test or SOP flow so the same class of bug fails deterministically next time.

### Bugfix/Hotfix Rule (Reproduce -> Pin -> Sweep)

For bugfix/hotfix work, acceptance evidence must include:

1. pre-fix reproduction evidence
2. post-fix targeted regression evidence
3. final full-gate evidence

A green full gate alone is not sufficient bugfix evidence unless the record also shows how the specific failure was reproduced and pinned.

### Documentation-only Exception

If all touched files are documentation/planning text only and no code, tests, scripts, config, generated artifacts, dependency manifests, or runtime behavior changed, full test execution is optional. Once executable or runtime-affecting files change, this exception does not apply.

### Environment Guardrails

- Keep the Python interpreter consistent across all commands.
- Prefer a project-local virtual environment: `.venv` on Windows and `.venv-wsl` on WSL/Linux when the repo supports dual-OS validation.
- Do not mix global and venv-installed `pre-commit` accidentally.
- Node.js must be 18+ before running frontend/E2E tests.
- On Windows, prefer repo-local `PRE_COMMIT_HOME` to avoid cache lock issues.
- On WSL, if `python` is missing but `python3` exists, create a local shim before running Playwright or harness commands.
- If pre-commit modifies files, review/stage/commit those changes and rerun hooks until clean.

### Evidence Recording

Implementation records must include date/time, OS/environment, command log reference, and pass/fail result for each required stage. If a gate is intentionally skipped as non-applicable, record why and name the replacement validation lane.
<!-- ROOKIEUI-GLOBAL-ONE-COMMAND:START -->
### One-command Full Gate

Prefer repo wrapper scripts when they exist:

- Linux/WSL: `bash scripts/run_full_tests_linux.sh`
- Windows: `powershell -File scripts/run_full_tests_windows.ps1`

If wrapper scripts do not exist in this repository, use the manual staged workflow in this SOP and record that the wrapper lane is not available.<!-- ROOKIEUI-GLOBAL-ONE-COMMAND:END -->
<!-- ROOKIEUI-GLOBAL-TEST-SOP-RULES:END -->
