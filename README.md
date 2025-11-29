### Installation

This repository uses [uv](https://docs.astral.sh/uv/) Python package manager, which can be installed with
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install repository packages run
```
uv sync --frozen
source .venv/bin/activate
```

To add dependencies, update `pyproject.toml`, then run `uv lock --upgrade && uv sync --frozen` and commit the changes.