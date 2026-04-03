"""
quick_benchmark.py - fast model comparison for SAR SR.

Purpose:
    - Compare SCUNet, SwinIR, RRDBNet, and bicubic quickly on a small patch subset.
  - Report PSNR/SSIM and rough runtime per patch.
  - Keep runtime short by default for rapid iteration.

Example:
  python src/quick_benchmark.py
    python src/quick_benchmark.py --models scunet swinir rrdb bicubic --max_samples 200
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from model import RRDBNet
from sar_dataset import SARDataset
from scunet_model import SCUNet
from swinIR_model import SwinIR


SCALE = 2
PATCH_SIZE = 128
LR_SIZE = PATCH_SIZE // SCALE


def compute_psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    return skimage_psnr(hr, sr, data_range=1.0)


def compute_ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    return skimage_ssim(hr, sr, data_range=1.0)


def _load_state_dict_from_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        # Support multiple checkpoint conventions used across this repo.
        for key in ("model_state_dict", "model_state", "state_dict", "params_ema", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip DataParallel prefixes if present.
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint did not contain a valid state_dict: {ckpt_path}")

    return state_dict


def _remap_legacy_rrdb_head(state_dict: dict) -> dict:
    # Older RRDB checkpoints used conv_last.{0,2} for [conv_hr, conv_last].
    if "conv_hr.weight" in state_dict:
        return state_dict
    if "conv_last.0.weight" not in state_dict or "conv_last.2.weight" not in state_dict:
        return state_dict

    remapped = dict(state_dict)
    remapped["conv_hr.weight"] = remapped.pop("conv_last.0.weight")
    remapped["conv_hr.bias"] = remapped.pop("conv_last.0.bias")
    remapped["conv_last.weight"] = remapped.pop("conv_last.2.weight")
    remapped["conv_last.bias"] = remapped.pop("conv_last.2.bias")
    return remapped


def load_scunet(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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
    state_dict = _load_state_dict_from_checkpoint(ckpt_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_rrdb(ckpt_path: Path, device: torch.device, num_blocks: int) -> torch.nn.Module:
    model = RRDBNet(
        in_channels=1,
        out_channels=1,
        num_feat=64,
        num_blocks=num_blocks,
        growth_rate=32,
        scale=4,
    ).to(device)

    state_dict = _remap_legacy_rrdb_head(_load_state_dict_from_checkpoint(ckpt_path, device))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if num_blocks != 8:
            print(
                "RRDB load with num_blocks={} failed, retrying with num_blocks=8.\n"
                "  Reason: {}".format(num_blocks, exc)
            )
            model = RRDBNet(
                in_channels=1,
                out_channels=1,
                num_feat=64,
                num_blocks=8,
                growth_rate=32,
                scale=4,
            ).to(device)
            model.load_state_dict(state_dict)
        else:
            raise

    model.eval()
    return model


def load_swinir(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = SwinIR(
        in_channels=1,
        out_channels=1,
        img_size=LR_SIZE,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        scale=4,
        drop_path_rate=0.1,
    ).to(device)
    state_dict = _load_state_dict_from_checkpoint(ckpt_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_batch(name: str, lr_batch: torch.Tensor, model: Optional[torch.nn.Module]) -> torch.Tensor:
    if name == "bicubic":
        return F.interpolate(lr_batch, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0.0, 1.0)

    if name == "scunet":
        base = F.interpolate(lr_batch, scale_factor=SCALE, mode="bicubic", align_corners=False)
        residual = model(lr_batch)
        return (base + residual).clamp(0.0, 1.0)

    if name == "rrdb":
        return model(lr_batch).clamp(0.0, 1.0)

    if name == "swinir":
        return model(lr_batch).clamp(0.0, 1.0)

    raise ValueError(f"Unknown model name: {name}")


def match_target_size(sr: torch.Tensor, hr: torch.Tensor, model_name: str, warned: Dict[str, bool]) -> torch.Tensor:
    if sr.shape[-2:] == hr.shape[-2:]:
        return sr
    if not warned.get(model_name, False):
        print(
            f"Warning: {model_name} output size {tuple(sr.shape[-2:])} differs from target {tuple(hr.shape[-2:])}. "
            "Resizing output for approximate metric comparison."
        )
        warned[model_name] = True
    return F.interpolate(sr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def evaluate_one(
    name: str,
    model: Optional[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    timed_batches: int,
    warmup_batches: int,
) -> dict:
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    warned = {}

    warmup_left = warmup_batches
    timed_left = timed_batches
    timed_images = 0
    timed_seconds = 0.0

    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        # Warmup iterations for more stable timing.
        if warmup_left > 0:
            with torch.no_grad():
                _ = predict_batch(name, lr_batch, model)
            warmup_left -= 1
            continue

        start = time.perf_counter()
        with torch.no_grad():
            sr_batch = predict_batch(name, lr_batch, model)
            sr_batch = match_target_size(sr_batch, hr_batch, name, warned)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if timed_left > 0:
            timed_seconds += elapsed
            timed_images += lr_batch.shape[0]
            timed_left -= 1

        sr_np = sr_batch.squeeze(1).detach().cpu().float().numpy()
        hr_np = hr_batch.squeeze(1).detach().cpu().float().numpy()

        for i in range(sr_np.shape[0]):
            psnr_vals.append(compute_psnr(sr_np[i], hr_np[i]))
            ssim_vals.append(compute_ssim(sr_np[i], hr_np[i]))

    psnr_arr = np.array(psnr_vals, dtype=np.float64)
    ssim_arr = np.array(ssim_vals, dtype=np.float64)

    ms_per_patch = float("nan")
    patches_per_sec = float("nan")
    if timed_images > 0 and timed_seconds > 0:
        patches_per_sec = timed_images / timed_seconds
        ms_per_patch = 1000.0 / patches_per_sec

    return {
        "count": int(len(psnr_arr)),
        "psnr_mean": float(psnr_arr.mean()) if len(psnr_arr) else float("nan"),
        "psnr_std": float(psnr_arr.std()) if len(psnr_arr) else float("nan"),
        "ssim_mean": float(ssim_arr.mean()) if len(ssim_arr) else float("nan"),
        "ssim_std": float(ssim_arr.std()) if len(ssim_arr) else float("nan"),
        "ms_per_patch": ms_per_patch,
        "patches_per_sec": patches_per_sec,
    }


def build_loader(args: argparse.Namespace) -> DataLoader:
    tif_dir = Path(args.data_dir)
    tif_paths = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff"))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFFs found in: {tif_dir}")

    if args.max_images > 0:
        tif_paths = tif_paths[: args.max_images]

    dataset = SARDataset(
        tif_paths=tif_paths,
        patch_size=PATCH_SIZE,
        patches_per_image=args.patches_per_image,
        augment=False,
    )

    if args.max_samples > 0 and args.max_samples < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), args.max_samples, replace=False).tolist()
        dataset = Subset(dataset, indices)

    print(f"GeoTIFFs used      : {len(tif_paths)}")
    print(f"Patches evaluated  : {len(dataset)}")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")

    loader = build_loader(args)

    models: Dict[str, Optional[torch.nn.Module]] = {}

    for name in args.models:
        if name == "bicubic":
            models[name] = None
            continue

        if name == "scunet":
            ckpt = Path(args.scunet_checkpoint)
            if not ckpt.exists():
                print(f"Skipping scunet: checkpoint not found: {ckpt}")
                continue
            print(f"Loading SCUNet from {ckpt}")
            models[name] = load_scunet(ckpt, device)
            continue

        if name == "rrdb":
            ckpt = Path(args.rrdb_checkpoint)
            if not ckpt.exists():
                print(f"Skipping rrdb: checkpoint not found: {ckpt}")
                continue
            print(f"Loading RRDBNet from {ckpt}")
            models[name] = load_rrdb(ckpt, device, num_blocks=args.rrdb_blocks)
            continue

        if name == "swinir":
            ckpt = Path(args.swinir_checkpoint)
            if not ckpt.exists():
                print(f"Skipping swinir: checkpoint not found: {ckpt}")
                continue
            print(f"Loading SwinIR from {ckpt}")
            models[name] = load_swinir(ckpt, device)
            continue

    if not models:
        raise RuntimeError("No models loaded. Check --models and checkpoint paths.")

    print("\n" + "=" * 72)
    print(" QUICK BENCHMARK")
    print("=" * 72)
    print("Notes:")
    print("  - Uses a small random subset for fast turnaround.")
    print("  - RRDB and SwinIR are trained as 4x in this repo; if output size mismatches target,")
    print("    it is resized to target size for approximate metric comparison.")
    print("=" * 72)

    results = {}
    for name in models:
        print(f"\nEvaluating: {name}")
        results[name] = evaluate_one(
            name=name,
            model=models[name],
            loader=loader,
            device=device,
            timed_batches=args.timed_batches,
            warmup_batches=args.warmup_batches,
        )

    print("\n" + "=" * 72)
    print(" BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"{'Model':<10} {'PSNR (dB)':<20} {'SSIM':<20} {'Speed':<20}")
    print("-" * 72)
    for name, r in results.items():
        psnr_txt = f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}"
        ssim_txt = f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
        if np.isnan(r["ms_per_patch"]):
            speed_txt = "n/a"
        else:
            speed_txt = f"{r['ms_per_patch']:.2f} ms/patch"
        print(f"{name:<10} {psnr_txt:<20} {ssim_txt:<20} {speed_txt:<20}")
    print("=" * 72)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Fast benchmark for SAR SR models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["scunet", "swinir", "rrdb", "bicubic"],
        choices=["scunet", "swinir", "rrdb", "bicubic"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(base_dir / ".." / "data" / "samples" / "geotiffs"),
    )
    parser.add_argument(
        "--scunet_checkpoint",
        type=str,
        default=str(base_dir / ".." / "checkpoints" / "best_scunet.pth"),
    )
    parser.add_argument(
        "--rrdb_checkpoint",
        type=str,
        default=str(base_dir / ".." / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument(
        "--swinir_checkpoint",
        type=str,
        default=str(base_dir / ".." / "checkpoints" / "best_swinir.pth"),
    )
    parser.add_argument(
        "--rrdb_blocks",
        type=int,
        default=23,
        help="RRDB blocks to try first; loader auto-falls back to 8 if needed",
    )
    parser.add_argument("--patches_per_image", type=int, default=20)
    parser.add_argument("--max_images", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--timed_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())