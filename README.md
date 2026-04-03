# Deep Learning Project

This repository is intended to host the source code for the project.

Large training artifacts are intentionally not tracked in Git:

- model checkpoints in `checkpoints/` and `backup checkpoints/`
- training logs in `runs/`
- generated diagnostics in `diagnostics/`
- large GeoTIFF datasets and super-resolved outputs under `data/samples/` and `data/samples_SR/`
- local virtual environment files in `.venv/`

## Repository policy

GitHub is being used for:

- source code in `src/`
- small metadata files such as CSV manifests
- documentation and experiment notes

Large files should be shared separately using one of these approaches:

1. GitHub Releases for a small number of deliverable checkpoints
2. Git LFS for a few essential model files that must stay tied to the repo
3. Google Drive, OneDrive, Kaggle, Hugging Face, or another external storage location for datasets and many checkpoints

## Recommended submission setup

For a course or project submission, the safest layout is:

1. Keep this GitHub repository code-only
2. Upload required checkpoints and sample outputs to external storage
3. Put the download links in this README
4. Document which script reproduces training and evaluation

## Next additions

Useful follow-up files to add:

- `requirements.txt` or `environment.yml`
- a short section describing how to train and evaluate the models
- links to externally hosted checkpoints and datasets