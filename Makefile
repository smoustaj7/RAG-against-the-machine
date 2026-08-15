export UV_CACHE_DIR ?= /home/smoustaj/goinfre/cache

.PHONY: all install run debug clean lint lint-strict

all: install


install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/processed/*

lint:
	uv run flake8 src
	uv run mypy src

lint-strict:
	uv run flake8 src
	uv run mypy src --strict
