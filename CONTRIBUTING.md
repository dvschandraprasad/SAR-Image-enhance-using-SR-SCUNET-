# Contributing

Thank you for your interest! This repository is a finished course project, but issues and pull requests are welcome.

## Reporting a problem

Open a GitHub issue. Say which script you ran, the full command, your OS, your Python / PyTorch / CUDA versions, and the full error message. Please check the [known issues](docs/method.md#known-issues) first.

## Making a change

1. Fork the repo and create a branch: `git checkout -b fix/short-description`.
2. Set up the environment as described in the [README](README.md#installation) (`.venv`, PyTorch from pytorch.org, then `pip install -r requirements.txt`).
3. Keep changes small and focused. Keep the `src/` layout, because the scripts import each other by module name.
4. Run the smoke tests. They need no data and no GPU:
   ```bash
   pip install pytest
   pytest
   ```
   The same tests run in GitHub Actions on every push and pull request (`.github/workflows/tests.yml`).
5. If you fix a known issue, remove the matching `skip` in `tests/test_smoke.py` (if there is one) and update `docs/method.md`.
6. If your change affects training or evaluation numbers, say so in the pull request and explain how you measured them.
7. Never commit data, checkpoints, logs or secrets. `.gitignore` already blocks `*.tif`, `*.pth`, `checkpoints/`, `runs/` and `diagnostics/`.

## Style

- Follow PEP 8 where you can, and match the style of the code around your change.
- Write clear docstrings and comments in simple English.

## License

By contributing, you agree that your contribution is released under the MIT License of this repository.
