# SAR Image Enhancement Using SR, SCUNet, and SwinIR

This repository contains the source code for a deep learning project focused on single-image super-resolution for remote sensing and SAR imagery. The project includes data preparation, degradation pipelines, model training, inference, and evaluation for multiple restoration architectures including RRDB, SCUNet, and SwinIR.

## Repository Contents

- source code for training, inference, and evaluation in `src/`
- dataset metadata and CSV manifests in `data/`
- experiment-related scripts and utilities in the project root

## Excluded Artifacts

Large generated artifacts are not tracked in Git, including:

- model checkpoints in `checkpoints/` and `backup checkpoints/`
- training logs in `runs/`
- generated diagnostics in `diagnostics/`
- large GeoTIFF datasets and super-resolved outputs in `data/samples/` and `data/samples_SR/`
- local environment files in `.venv/`
