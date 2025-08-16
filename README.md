# diffusion-sandbox

A minimal yet production-style Diffusion learning template: YAML configuration, determinism, tracking and logging, unit tests, type checking and style tools, and complete visualization examples.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest
python -m diffusion_sandbox.train --config configs/default.yaml
tensorboard --logdir runs
```

## Configuration

Modify dataset/model/diffusion/training/tracking parameters in `configs/default.yaml`.

## Style & Typing

- Formatting: `black . && isort .`
- Static checks: `ruff check .`
- Type checking: `mypy src tests`

## Structure

See the repository directory and docstrings in the source code.

## Dataset Visualization

```bash
python scripts/visualize_datasets.py
```

After running, the following will be generated under `examples/figures/`:

- gmm.png
- ring.png