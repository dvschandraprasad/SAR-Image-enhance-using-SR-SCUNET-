# Changelog

All notable changes to this project are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-13

First clean public release of the spring 2026 Deep Learning course project.

### Added
- Full `README.md`: summary, highlights, results (each number marked as checked or author-reported), quick start, pipeline diagram, degradation model, model comparison, dataset and license notes, installation, step-by-step reproduction, script reference, project structure, known issues, limitations, citations.
- `docs/method.md`: technical notes on sampling, degradation (with formulas), curriculum, normalisation, SCUNet design, losses, training setup, metrics (PSNR, SSIM, ENL), evaluation, inference and a full list of known issues.
- `tests/test_smoke.py` and `pytest.ini`: CPU smoke tests that need no data and no GPU (model forward passes and parameter counts, degradation and normalisation, curriculum, constants that must stay in sync). The two known-broken training scripts are skipped with a reason.
- `.github/workflows/tests.yml`: runs the smoke tests on every push and pull request with CPU-only PyTorch.
- `requirements.txt`, built from the real imports.
- `LICENSE` (MIT, code only; the Capella data is CC BY 4.0).
- `CHANGELOG.md` and `CONTRIBUTING.md`.

### Changed
- Moved the notebook from `src/STAC_to_CSV.ipynb_/STAC_to_CSV.ipynb` to `notebooks/STAC_to_CSV.ipynb`. No code refers to the old path.
- Made `.gitignore` stronger (checkpoints, runs, diagnostics, imagery, `*.tif`, weights, macOS files). The CSV files in `data/` stay tracked.

### Removed
- `src/STAC_to_CSV.ipynb_/__MACOSX/._STAC_to_CSV.ipynb`, a macOS metadata file. It held browser download history, including a signed download URL.

### Not changed
- The program logic of every `.py` file and of the notebook is the same as before. Known problems are documented in `docs/method.md`, not fixed.
