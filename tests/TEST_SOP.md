# TEST_SOP.md

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
