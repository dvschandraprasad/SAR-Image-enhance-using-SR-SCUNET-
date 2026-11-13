"""
evaluate.py — Evaluation script for SCUNet SAR Super-Resolution (v16)
Metrics: PSNR, SSIM, ENL (Equivalent Number of Looks), Bicubic baseline
Run from src/: python evaluate.py
  or from project root: python src/evaluate.py

Model: SCUNet with residual_output=True
  sr = upsample_bicubic(lr) + model(lr)

Outputs:
  - Running per-batch metrics printed to console
  - Aggregate mean ± std summary
  - Optional: saves LR / Bicubic / SR / HR 4-panel PNGs
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

sys.path.insert(0, str(Path(__file__).parent))
from sar_dataset import SARDataset
from scunet_model import SCUNet


# ── Constants — must match training config ───────────────────────────────────
SCALE        = 2
PATCH_SIZE   = 128          # HR patch
LR_SIZE      = PATCH_SIZE // SCALE   # 64


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    return skimage_psnr(hr, sr, data_range=1.0)


def compute_ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    return skimage_ssim(hr, sr, data_range=1.0)


def compute_enl(image: np.ndarray) -> float:
    """ENL = mean² / variance — SAR speckle homogeneity metric."""
    mu  = np.mean(image)
    var = np.var(image)
    if var < 1e-10:
        return float('nan')
    return float((mu ** 2) / var)


def upsample_bicubic(lr_batch: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr_batch, scale_factor=scale,
                         mode='bicubic', align_corners=False)


def _smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    result = np.empty_like(values, dtype=np.float64)
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result[i] = values[lo:hi].mean()
    return result


def plot_loss_curves(run_dir: str, out_path: str, smooth_window: int = 1) -> None:
    """Plot train/val loss curves from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        print("TensorBoard event parser not available. Install tensorboard to enable loss plots.")
        return

    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Run directory not found: {run_path}")
        return

    event_files = sorted(run_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {run_path}")
        return

    train_by_step = {}
    val_by_step = {}
    found_any_tags = False

    for ef in event_files:
        ea = event_accumulator.EventAccumulator(str(ef))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if 'train/loss' in tags:
            found_any_tags = True
            for e in ea.Scalars('train/loss'):
                train_by_step[int(e.step)] = float(e.value)
        if 'val/loss' in tags:
            found_any_tags = True
            for e in ea.Scalars('val/loss'):
                val_by_step[int(e.step)] = float(e.value)

    if not found_any_tags:
        print("Could not find train/loss or val/loss tags in TensorBoard logs.")
        return

    if not train_by_step and not val_by_step:
        print("No scalar events found for train/loss or val/loss.")
        return

    train_steps = np.array(sorted(train_by_step.keys()), dtype=np.int32)
    train_vals = np.array([train_by_step[s] for s in train_steps], dtype=np.float64)
    val_steps = np.array(sorted(val_by_step.keys()), dtype=np.int32)
    val_vals = np.array([val_by_step[s] for s in val_steps], dtype=np.float64)

    train_plot = _smooth_curve(train_vals, smooth_window)
    val_plot = _smooth_curve(val_vals, smooth_window)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_plot, label='Train Loss', linewidth=2.0)
    plt.plot(val_steps, val_plot, label='Val Loss', linewidth=2.0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"Saved loss curve plot: {out_file}")


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> SCUNet:
    model = SCUNet(
        in_channels    = 1,
        out_channels   = 1,
        img_size       = LR_SIZE,
        nc             = 96,
        depth          = (2, 2, 2, 2),
        num_heads      = 6,
        window_size    = 8,
        mlp_ratio      = 4.,
        drop_path_rate = 0.0,
        scale          = SCALE,
        residual_output = True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    epoch    = ckpt.get('epoch', '?')
    val_psnr = ckpt.get('val_psnr', float('nan'))
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Saved at epoch {epoch}, val PSNR {val_psnr:.2f} dB")
    return model


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    if args.plot_losses_only:
        print("Generating loss curve plot only (evaluation skipped) ...")
        plot_loss_curves(
            run_dir=args.run_dir,
            out_path=args.loss_plot_out,
            smooth_window=args.loss_smooth_window,
        )
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = load_model(args.checkpoint, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    tif_dir   = Path(args.data_dir)
    tif_paths = sorted(tif_dir.glob('*.tif')) + sorted(tif_dir.glob('*.tiff'))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFFs found in {tif_dir}")
    print(f"Found {len(tif_paths)} GeoTIFF(s) in {tif_dir}")

    dataset = SARDataset(
        tif_paths          = tif_paths,
        patch_size         = PATCH_SIZE,
        patches_per_image  = args.patches_per_image,
        augment            = False,
    )

    if args.max_samples and args.max_samples < len(dataset):
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(dataset), args.max_samples, replace=False).tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = True,
    )
    print(f"Evaluating on {len(dataset)} patches …\n")

    # ── Accumulators ─────────────────────────────────────────────────────────
    sr_psnr_vals, sr_ssim_vals       = [], []
    bic_psnr_vals, bic_ssim_vals     = [], []
    enl_sr_vals, enl_hr_vals         = [], []

    with torch.no_grad():
        for batch_idx, (lr_batch, hr_batch) in enumerate(loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # ── Forward pass — mirrors training exactly ──────────────────────
            base_batch     = upsample_bicubic(lr_batch, SCALE)
            residual_batch = model(lr_batch)
            sr_batch       = (base_batch + residual_batch).clamp(0.0, 1.0)
            bic_batch      = base_batch.clamp(0.0, 1.0)

            sr_np  = sr_batch.squeeze(1).cpu().float().numpy()
            bic_np = bic_batch.squeeze(1).cpu().float().numpy()
            hr_np  = hr_batch.squeeze(1).cpu().float().numpy()

            for i in range(sr_np.shape[0]):
                s = sr_np[i]
                b = bic_np[i]
                h = hr_np[i]

                sr_psnr_vals.append(compute_psnr(s, h))
                sr_ssim_vals.append(compute_ssim(s, h))
                bic_psnr_vals.append(compute_psnr(b, h))
                bic_ssim_vals.append(compute_ssim(b, h))
                enl_sr_vals.append(compute_enl(s))
                enl_hr_vals.append(compute_enl(h))

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  [{batch_idx+1}/{len(loader)}]  "
                    f"SR PSNR {np.mean(sr_psnr_vals):.2f} dB  "
                    f"SR SSIM {np.mean(sr_ssim_vals):.4f}  "
                    f"Bicubic PSNR {np.mean(bic_psnr_vals):.2f} dB"
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    sr_psnr  = np.array(sr_psnr_vals)
    sr_ssim  = np.array(sr_ssim_vals)
    bic_psnr = np.array(bic_psnr_vals)
    bic_ssim = np.array(bic_ssim_vals)
    enl_sr   = np.array([v for v in enl_sr_vals if not np.isnan(v)])
    enl_hr   = np.array([v for v in enl_hr_vals if not np.isnan(v)])

    gain_psnr = sr_psnr.mean() - bic_psnr.mean()
    gain_ssim = sr_ssim.mean() - bic_ssim.mean()

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Patches evaluated  : {len(sr_psnr)}")
    print(f"  SCUNet  PSNR       : {sr_psnr.mean():.2f} ± {sr_psnr.std():.2f} dB")
    print(f"  SCUNet  SSIM       : {sr_ssim.mean():.4f} ± {sr_ssim.std():.4f}")
    print(f"  Bicubic PSNR       : {bic_psnr.mean():.2f} ± {bic_psnr.std():.2f} dB")
    print(f"  Bicubic SSIM       : {bic_ssim.mean():.4f} ± {bic_ssim.std():.4f}")
    print(f"  PSNR gain vs bicubic : {gain_psnr:+.2f} dB")
    print(f"  SSIM gain vs bicubic : {gain_ssim:+.4f}")
    print(f"  ENL (SR)           : {enl_sr.mean():.2f} ± {enl_sr.std():.2f}")
    print(f"  ENL (HR)           : {enl_hr.mean():.2f} ± {enl_hr.std():.2f}")
    print("=" * 60)

    if args.plot_losses:
        print("\nGenerating loss curve plot from TensorBoard logs ...")
        plot_loss_curves(
            run_dir=args.run_dir,
            out_path=args.loss_plot_out,
            smooth_window=args.loss_smooth_window,
        )

    if args.save_images:
        _save_comparison_images(args, model, device, dataset)


# ── Optional image saving ─────────────────────────────────────────────────────

def _save_comparison_images(args, model, device, dataset):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping image saving.")
        return

    out_dir = Path(args.image_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    saved  = 0

    with torch.no_grad():
        for lr_batch, hr_batch in loader:
            if saved >= args.save_images:
                break

            lr_d           = lr_batch.to(device)
            base_batch     = upsample_bicubic(lr_d, SCALE)
            residual_batch = model(lr_d)
            sr_batch       = (base_batch + residual_batch).clamp(0.0, 1.0).cpu()
            bic_batch      = base_batch.clamp(0.0, 1.0).cpu()

            lr_img  = lr_batch[0, 0].numpy()
            bic_img = bic_batch[0, 0].numpy()
            sr_img  = sr_batch[0, 0].numpy()
            hr_img  = hr_batch[0, 0].numpy()

            sr_psnr  = compute_psnr(sr_img, hr_img)
            bic_psnr = compute_psnr(bic_img, hr_img)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, img, title in zip(
                axes,
                [lr_img, bic_img, sr_img, hr_img],
                [f"LR ({LR_SIZE}×{LR_SIZE})",
                 f"Bicubic ({PATCH_SIZE}×{PATCH_SIZE})\n{bic_psnr:.2f} dB",
                 f"SCUNet ({PATCH_SIZE}×{PATCH_SIZE})\n{sr_psnr:.2f} dB",
                 f"HR ({PATCH_SIZE}×{PATCH_SIZE})"],
            ):
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis('off')

            plt.tight_layout()
            fig.savefig(out_dir / f'sample_{saved:04d}.png', dpi=150)
            plt.close(fig)
            saved += 1

    print(f"Saved {saved} comparison image(s) to {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Evaluate SCUNet SAR SR model')
    parser.add_argument('--checkpoint',        type=str,
                        default=str(BASE_DIR / '..' / 'checkpoints' / 'best_scunet.pth'))
    parser.add_argument('--data_dir',          type=str,
                        default=str(BASE_DIR / '..' / 'data' / 'samples' / 'geotiffs'))
    parser.add_argument('--patches_per_image', type=int, default=200)
    parser.add_argument('--max_samples',       type=int, default=2000,
                        help='Hard cap on patches to eval (0 = no cap)')
    parser.add_argument('--batch_size',        type=int, default=16)
    parser.add_argument('--workers',           type=int, default=4)
    parser.add_argument('--save_images',       type=int, default=0, metavar='N',
                        help='Save N 4-panel comparison PNGs (0 = disabled)')
    parser.add_argument('--image_out_dir',     type=str, default='eval_images')
    parser.add_argument('--plot_losses',       action='store_true',
                        help='Generate train/val loss graph from TensorBoard logs')
    parser.add_argument('--plot_losses_only',  action='store_true',
                        help='Generate train/val loss graph and skip full evaluation')
    parser.add_argument('--run_dir',           type=str,
                        default=str(BASE_DIR / '..' / 'runs' / 'scunet-128patch-2x-v17-progressive-ease'),
                        help='TensorBoard run directory containing events.out.tfevents.*')
    parser.add_argument('--loss_plot_out',     type=str,
                        default=str(BASE_DIR / '..' / 'diagnostics' / 'loss_curve.png'),
                        help='Output path for saved loss graph PNG')
    parser.add_argument('--loss_smooth_window', type=int, default=1,
                        help='Moving-average window for smoothing loss curves (1 disables smoothing)')

    args = parser.parse_args()
    if args.max_samples == 0:
        args.max_samples = None

    evaluate(args)