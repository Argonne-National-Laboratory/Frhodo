# Contributing to Frhodo

Thanks for your interest in improving Frhodo.

## Development setup

Dependencies are declared in `pyproject.toml`. Use either uv or conda.

### uv

```bash
uv sync --extra gui --extra dev
```

### conda

```bash
conda env create -f environment.yml
conda activate frhodo
pip install -e ".[dev]"
```

## Running the tests

The suite uses pytest. GUI tests need Qt to run offscreen:

```bash
QT_QPA_PLATFORM=offscreen pytest
```

- Mark Qt-dependent tests with `@pytest.mark.gui` and long-running ones with
  `@pytest.mark.slow`.
- Run the suite serially; avoid launching several heavy Python processes at once.

## Linting and types

```bash
ruff check src tests
pyright
```

`pyright` runs on the typed surface listed under `[tool.pyright]` in
`pyproject.toml`. New modules should type-check cleanly so that surface can be
widened over time.

## Pull requests

- Branch from `master`, keep changes focused, and add tests for new behavior.
- Make sure `pytest`, `ruff check`, and `pyright` pass before opening the PR.
- Use clear, single-purpose commits.
