# Repository Guidelines

## Project Structure & Modules
- `main.py`: single entry point for training, validation, and utilities.
- `config/`: configuration files (primary: `config/config.yaml`).
- `data/`: loaders, preprocessing, feature engineering, datasets.
- `features/`: reusable feature transforms for time-series.
- `models/`: PatchTST-based model and heads (`patchtst_unified.py`).
- `training/`: trainers, optimizers, fine-tuning utilities.
- `trading/`: signals, backtester, model adapter, risk manager.
- `validation/`: comprehensive offline evaluation and stress tests.
- `utils/`: logging, metrics, config helpers, diagnostics.

## Build, Test, and Run
- Setup:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run:
  - Download data: `python download_data.py`
  - Prepare features: `python prepare_trading_data.py`
  - Train: `python main.py --mode train`
  - Full pipeline: `python main.py --mode full`
  - Demo / config check: `python main.py --mode demo` or `python main.py --validate-only`
  - Validation suite: `python validation/run_comprehensive_validation.py`
  - Monitor: `tensorboard --logdir logs/` or `python -m tools.cli monitor`
  - Eval: `python -m tools.cli eval` (wraps `evaluate_model_simple.py`)

## Coding Style & Naming
- Use PEP 8 with 4-space indentation; add type hints for public APIs.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Formatting: run `black .` before commits (Black is included in requirements).
- Keep functions cohesive; prefer explicit configuration via `config/config.yaml`.

## Testing Guidelines
- Framework: `pytest` (included).
- Location: place tests under `tests/` mirroring module paths.
- Naming: files `test_<module>.py`, functions `test_<behavior>`.
- Run: `pytest -q` (aim for ≥80% coverage on changed code).

## Commit & Pull Request Guidelines
- Use conventional commit prefixes: `feat|fix|refactor|docs|chore|cleanup: message`.
  - Example: `feat(training): improve class balancing in staged trainer`
- PRs must include: clear description, linked issues, reproduction/validation steps, relevant logs/metrics (F1, WinRate), and config diffs when applicable. Ensure `black` and `pytest` pass.

## Security & Configuration
- Do not commit secrets. Provide DB/API credentials via environment or local overrides; review `config/config.yaml` before runs.
- Heavy training is GPU-intensive—use smaller batches/symbol subsets for local tests.

