PYTHON_PATHS := plugins/filters plugins/pipes utils dev tests examples

.PHONY: sync quality check test upgrade-dev-tools

sync:
	uv sync

quality:
	uv run ruff format $(PYTHON_PATHS)
	uv run ruff check --fix $(PYTHON_PATHS)
	$(MAKE) check

check:
	uv lock --check
	uv run ruff format --check $(PYTHON_PATHS)
	uv run ruff check $(PYTHON_PATHS)
	uv run pyright
	uv run vulture
	uv run pytest

test:
	uv run pytest

upgrade-dev-tools:
	uv lock --upgrade-package pyright --upgrade-package pytest --upgrade-package pytest-asyncio --upgrade-package ruff --upgrade-package vulture
	uv sync
