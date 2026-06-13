# E2E Testing SOP

This SOP defines the verified E2E workflow for this repository. If this repository does not currently have a frontend or Playwright harness, keep this file as the shared policy and record the replacement validation lane in `tests/TEST_SOP.md`.

<!-- ROOKIEUI-GLOBAL-E2E-SOP-RULES:START -->
## RookieUI-Derived Global E2E Rules

This section preserves the repo's existing E2E procedure while adding the shared Playwright/harness baseline used across this workspace.

### Problem-First Test Design Rule

E2E scripts and mocked harness flows must be designed to reproduce failures and catch bugs early. The goal is not to make the harness pass; the goal is to make the harness fail when a real user-facing contract breaks.

When adding or reviewing E2E coverage, prefer assertions that prove final user-visible behavior, request routing, payload shape, state synchronization, and failure feedback. Avoid pass-only checks that only prove the page loaded or a mocked happy path returned.

### Requirements

- Node.js 18+
- npm 9+ when the repo uses npm
- Python command available (`python` or a local shim to `python3`) when the harness serves files through Python
- Playwright Chromium installed with `npx playwright install chromium` when Playwright is used

### Windows (PowerShell)

```powershell
node -v
npm -v
python --version

npm install
npx playwright install chromium
npm test
```

### WSL2 (bash)

```bash
source ~/.nvm/nvm.sh
nvm use 18
node -v
python3 --version

mkdir -p .tmp/bin
ln -sf "$(command -v python3)" .tmp/bin/python

npm install
npx playwright install chromium

mkdir -p .tmp/playwright
TMPDIR=.tmp/playwright TMP=.tmp/playwright TEMP=.tmp/playwright \
  PATH=".tmp/bin:$PATH" npm test
```

### Troubleshooting

- `python: command not found` on WSL: create `.tmp/bin/python` as a shim to `python3`.
- Port bind failure: use the repo-documented E2E port override or stop the conflicting process.
- Browser missing: run `npx playwright install chromium`.
- Dependency drift: remove `node_modules` and rerun `npm install`.

### Non-applicable E2E

If the repo does not have a frontend or Playwright harness, document the non-applicability in `tests/TEST_SOP.md` and identify the replacement smoke, unit, or integration lane. Do not treat a missing E2E harness as an unrecorded pass.
<!-- ROOKIEUI-GLOBAL-E2E-SOP-RULES:END -->
