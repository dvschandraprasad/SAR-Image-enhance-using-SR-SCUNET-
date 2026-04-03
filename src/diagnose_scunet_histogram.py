"""
Diagnose SCUNet output distributions against Bicubic and HR targets.

Run from project root:
    python src/diagnose_scunet_histogram.py

Outputs written to diagnostics/:
  - scunet_histogram_comparison.png
  - scunet_patch_stats.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from sar_dataset import SARDataset
from scunet_model import SCUNet


SCALE = 2
PATCH_SIZE = 128
LR_SIZE = PATCH_SIZE // SCALE


def upsample_bicubic(lr_batch: torch.Tensor, scale: int = SCALE) -> torch.Tensor:
    return F.interpolate(lr_batch, scale_factor=scale, mode="bicubic", align_corners=False)


def load_model(checkpoint_path: Path, device: torch.device) -> SCUNet:
    model = SCUNet(
        in_channels=1,
        out_channels=1,
        img_size=LR_SIZE,
        nc=96,
        depth=(2, 2, 2, 2),
        num_heads=6,
        window_size=8,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        scale=SCALE,
        residual_output=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_psnr = ckpt.get("val_psnr", float("nan"))
    print(f"Loaded SCUNet checkpoint: {checkpoint_path}")
    print(f"  saved epoch={epoch}, val_psnr={val_psnr:.2f} dB")
    return model


def _append_patch_stats(batch_1ch: torch.Tensor, stats_dict: dict[str, list[np.ndarray]], key: str) -> None:
    patch_means = batch_1ch.mean(dim=(1, 2)).cpu().numpy()
    patch_stds = batch_1ch.std(dim=(1, 2)).cpu().numpy()
    stats_dict[f"{key}_mean"].append(patch_means)
    stats_dict[f"{key}_std"].append(patch_stds)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tif_paths = sorted(data_dir.glob("*.tif")) + sorted(data_dir.glob("*.tiff"))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFF files found in: {data_dir}")

    print(f"Found {len(tif_paths)} GeoTIFF files")

    dataset = SARDataset(
        tif_paths=tif_paths,
        patch_size=PATCH_SIZE,
        patches_per_image=args.patches_per_image,
        augment=False,
    )

    if args.max_samples > 0 and args.max_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=args.max_samples, replace=False).tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Using {len(dataset)} sampled patches")

    model = load_model(checkpoint_path, device)

    intensities: dict[str, list[np.ndarray]] = {"bicubic": [], "scunet": [], "hr": []}
    patch_stats: dict[str, list[np.ndarray]] = {
        "bicubic_mean": [],
        "bicubic_std": [],
        "scunet_mean": [],
        "scunet_std": [],
        "hr_mean": [],
        "hr_std": [],
    }

    with torch.no_grad():
        for batch_idx, (lr_batch, hr_batch) in enumerate(loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            bic_batch = upsample_bicubic(lr_batch, SCALE).clamp(0.0, 1.0)
            residual_batch = model(lr_batch)
            sr_batch = (bic_batch + residual_batch).clamp(0.0, 1.0)

            bic_np = bic_batch.squeeze(1).cpu().numpy()
            sr_np = sr_batch.squeeze(1).cpu().numpy()
            hr_np = hr_batch.squeeze(1).cpu().numpy()

            intensities["bicubic"].append(bic_np.reshape(-1))
            intensities["scunet"].append(sr_np.reshape(-1))
            intensities["hr"].append(hr_np.reshape(-1))

            _append_patch_stats(bic_batch.squeeze(1), patch_stats, "bicubic")
            _append_patch_stats(sr_batch.squeeze(1), patch_stats, "scunet")
            _append_patch_stats(hr_batch.squeeze(1), patch_stats, "hr")

            if (batch_idx + 1) % 10 == 0:
                print(f"  processed {batch_idx + 1}/{len(loader)} batches")

    for key in intensities:
        intensities[key] = np.concatenate(intensities[key], axis=0)
    for key in patch_stats:
        patch_stats[key] = np.concatenate(patch_stats[key], axis=0)

    print("\nIntensity summary:")
    for key in ("bicubic", "scunet", "hr"):
        arr = intensities[key]
        print(
            f"  {key:8s} mean={arr.mean():.4f} std={arr.std():.4f} "
            f"p5={np.percentile(arr, 5):.4f} p50={np.percentile(arr, 50):.4f} p95={np.percentile(arr, 95):.4f}"
        )

    hist_out = out_dir / "scunet_histogram_comparison.png"
    fig1, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    ax1.hist(intensities["bicubic"], bins=args.bins, density=True, alpha=0.45, color="tab:blue", label="Bicubic")
    ax1.hist(intensities["scunet"], bins=args.bins, density=True, alpha=0.45, color="tab:green", label="SCUNet")
    ax1.hist(intensities["hr"], bins=args.bins, density=True, alpha=0.45, color="tab:red", label="HR")
    ax1.set_title("Pixel Intensity Distribution: Bicubic vs SCUNet vs HR")
    ax1.set_xlabel("Normalized intensity")
    ax1.set_ylabel("Density")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(hist_out, dpi=140)
    plt.close(fig1)

    stats_out = out_dir / "scunet_patch_stats.png"
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(patch_stats["bicubic_mean"], bins=60, density=True, alpha=0.45, color="tab:blue", label="Bicubic")
    axes[0].hist(patch_stats["scunet_mean"], bins=60, density=True, alpha=0.45, color="tab:green", label="SCUNet")
    axes[0].hist(patch_stats["hr_mean"], bins=60, density=True, alpha=0.45, color="tab:red", label="HR")
    axes[0].set_title("Patch Mean Distribution")
    axes[0].set_xlabel("Patch mean")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].hist(patch_stats["bicubic_std"], bins=60, density=True, alpha=0.45, color="tab:blue", label="Bicubic")
    axes[1].hist(patch_stats["scunet_std"], bins=60, density=True, alpha=0.45, color="tab:green", label="SCUNet")
    axes[1].hist(patch_stats["hr_std"], bins=60, density=True, alpha=0.45, color="tab:red", label="HR")
    axes[1].set_title("Patch Std Distribution")
    axes[1].set_xlabel("Patch std")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig2.suptitle("Patch Statistics: Bicubic vs SCUNet vs HR", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(stats_out, dpi=140)
    plt.close(fig2)

    print(f"\nSaved histogram figure: {hist_out}")
    print(f"Saved patch-stats figure: {stats_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose SCUNet output distribution against Bicubic and HR"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BASE_DIR / ".." / "checkpoints" / "best_scunet.pth"),
        help="Path to SCUNet checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(BASE_DIR / ".." / "data" / "samples" / "geotiffs"),
        help="Directory containing input GeoTIFFs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(BASE_DIR / ".." / "diagnostics"),
        help="Output directory for diagnostic plots",
    )
    parser.add_argument("--patches_per_image", type=int, default=120)
    parser.add_argument("--max_samples", type=int, default=1200,
                        help="Max sampled patches (0 disables cap)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()
    run(cli)
