# AGENT KERNEL

MODE: GREENFIELD, strict no-compat.

## 0. PRECEDENCE

- Follow instruction priority: system > developer > AGENTS.md > user.
- If this file conflicts with a higher-priority instruction, follow the higher-priority instruction and continue.

## 1. NON-NEGOTIABLES

- Use `uv` for all Python environment and package work (`uv run`, `uv add`, `uv sync`); never use `pip` or manual venv workflows.
- Use `make upgrade-dev-tools` to refresh locked dev tooling; do not rely on `uv pip install` for project tool upgrades.
- Greenfield strict no-compat: no compatibility shims, legacy fallbacks, alias-preserving paths, or dual old/new behavior unless explicitly requested. Canonicalize non-canonical paths while preserving intended capability.
- Type fixes must not rely on `Any` or `type ignore` as the primary solution. Prefer explicit protocols, generics, and Pydantic models.
- Use scoped `rg` searches excluding `.venv`; never run broad recursive searches from the repo root.

## 2. EXECUTION MODEL

1. At session start, run `bd prime` from the repo root. Use `bd --help` for additional CLI usage when needed.
2. Track work in `bd` before coding, then build a dependency DAG for non-trivial changes.
3. If a required command fails, capture the error, stop that phase, and continue only with safe non-conflicting work.

## 3. VALIDATION

- Code changes require the full static gate before completion:
  - `make quality`
  - This delegates to tool configuration in `pyproject.toml` for Ruff, Pyright, and Vulture.
  - Use `make check` for the read-only CI-equivalent gate.
- Also run validation appropriate to the changed scope: targeted tests for touched modules or features, and integration or demo validation when affected.
- Docs, config, or workflow-only changes require only changed-scope validation that proves the edited files are well-formed and the workflow still makes sense.

## 4. COMPLETION

- Close completed `bd` issues, run required validation, and inspect `git status`.
- Stage and commit work before any required push sequence.
- When push is required, run `git pull --rebase`, `git push`, and verify the branch is up to date with origin. If push fails, resolve it and retry.
- Provide concise handoff context for the next session.
