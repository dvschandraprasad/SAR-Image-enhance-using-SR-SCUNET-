"""
Dataset diagnostic -- run this before any further training changes.
Saves a grid of LR / bicubic-upsampled LR / HR patch triplets to disk
and measures bicubic baseline PSNR so we know what the model is competing against.

Run from the src/ directory:
    python diagnose_dataset.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# --- locate sar_dataset relative to this script
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'data')
GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')
OUT_DIR     = os.path.join(BASE_DIR, '..', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from sar_dataset import SARDataset

tif_paths = sorted([
    os.path.join(GEOTIFF_DIR, f)
    for f in os.listdir(GEOTIFF_DIR) if f.endswith('.tif')
])
print(f"Found {len(tif_paths)} GeoTIFFs")

dataset = SARDataset(tif_paths, patch_size=256, patches_per_image=50, augment=False)
loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

lr_batch, hr_batch = next(iter(loader))
print(f"LR batch: {lr_batch.shape}  min={lr_batch.min():.4f}  max={lr_batch.max():.4f}")
print(f"HR batch: {hr_batch.shape}  min={hr_batch.min():.4f}  max={hr_batch.max():.4f}")

# ── 1. Bicubic baseline PSNR / SSIM ───────────────────────────────────────────
lr_up = F.interpolate(lr_batch, size=(256, 256), mode='bicubic', align_corners=False)
lr_up = lr_up.clamp(0, 1)

bicubic_psnrs, bicubic_ssims = [], []
model_floor_psnrs = []

for i in range(lr_batch.shape[0]):
    lr_np  = lr_up[i, 0].numpy()
    hr_np  = hr_batch[i, 0].numpy()
    bicubic_psnrs.append(compute_psnr(hr_np, lr_np, data_range=1.0))
    bicubic_ssims.append(compute_ssim(hr_np, lr_np, data_range=1.0))
    # also check mean-prediction floor (what a model predicting constant 0.5 gets)
    mean_pred = np.full_like(hr_np, hr_np.mean())
    model_floor_psnrs.append(compute_psnr(hr_np, mean_pred, data_range=1.0))

print(f"\n── Bicubic baseline ─────────────────────────────")
print(f"  PSNR : {np.mean(bicubic_psnrs):.2f} dB  (std {np.std(bicubic_psnrs):.2f})")
print(f"  SSIM : {np.mean(bicubic_ssims):.4f}  (std {np.std(bicubic_ssims):.4f})")
print(f"\n── Mean-prediction floor (constant output) ──────")
print(f"  PSNR : {np.mean(model_floor_psnrs):.2f} dB")
print(f"\nIf model PSNR ≈ bicubic PSNR → model = bicubic upsampler, not learning SR")
print(f"If model PSNR ≈ mean-floor PSNR → model collapsed to constant output")

# ── 2. LR / HR correlation check ──────────────────────────────────────────────
# Pearson correlation between bicubic-upsampled LR and HR
# High correlation (>0.7) means the mapping is learnable
# Low correlation (<0.4) means degradation destroyed too much signal
correlations = []
for i in range(lr_batch.shape[0]):
    lr_flat = lr_up[i, 0].numpy().ravel()
    hr_flat = hr_batch[i, 0].numpy().ravel()
    corr    = np.corrcoef(lr_flat, hr_flat)[0, 1]
    correlations.append(corr)

print(f"\n── LR↔HR Pearson correlation ────────────────────")
print(f"  Mean : {np.mean(correlations):.4f}  (std {np.std(correlations):.4f})")
print(f"  Min  : {np.min(correlations):.4f}  Max: {np.max(correlations):.4f}")
print(f"  >0.7 = learnable mapping  |  <0.4 = degradation too aggressive")

# ── 3. Pixel intensity distribution check ─────────────────────────────────────
lr_flat_all = lr_batch.numpy().ravel()
hr_flat_all = hr_batch.numpy().ravel()
print(f"\n── Pixel distribution ───────────────────────────")
print(f"  LR percentiles  p5={np.percentile(lr_flat_all,5):.3f}  "
      f"p50={np.percentile(lr_flat_all,50):.3f}  "
      f"p95={np.percentile(lr_flat_all,95):.3f}  "
      f"mean={lr_flat_all.mean():.3f}  std={lr_flat_all.std():.3f}")
print(f"  HR percentiles  p5={np.percentile(hr_flat_all,5):.3f}  "
      f"p50={np.percentile(hr_flat_all,50):.3f}  "
      f"p95={np.percentile(hr_flat_all,95):.3f}  "
      f"mean={hr_flat_all.mean():.3f}  std={hr_flat_all.std():.3f}")

# ── 4. Visual grid: LR_bicubic | HR | difference ──────────────────────────────
n_show = min(4, lr_batch.shape[0])
fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
fig.suptitle('Left: LR (bicubic up) | Middle: HR | Right: |LR-HR| diff', fontsize=12)

for i in range(n_show):
    lr_img   = lr_up[i, 0].numpy()
    hr_img   = hr_batch[i, 0].numpy()
    diff_img = np.abs(lr_img - hr_img)

    axes[i, 0].imshow(lr_img, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f'LR bicubic  PSNR={bicubic_psnrs[i]:.1f}dB', fontsize=9)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(hr_img, cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_title(f'HR  corr={correlations[i]:.3f}', fontsize=9)
    axes[i, 1].axis('off')

    axes[i, 2].imshow(diff_img, cmap='hot', vmin=0, vmax=0.5)
    axes[i, 2].set_title(f'|LR-HR|  mean={diff_img.mean():.3f}', fontsize=9)
    axes[i, 2].axis('off')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'lr_hr_pairs.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nVisual grid saved to: {out_path}")

# ── 5. Histogram comparison ────────────────────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(lr_flat_all, bins=100, alpha=0.6, label='LR', color='blue', density=True)
ax1.hist(hr_flat_all, bins=100, alpha=0.6, label='HR', color='red',  density=True)
ax1.set_title('LR vs HR pixel distribution')
ax1.set_xlabel('Normalised value')
ax1.legend()

ax2.hist(bicubic_psnrs, bins=20, color='green', alpha=0.8)
ax2.axvline(np.mean(bicubic_psnrs), color='red', linestyle='--',
            label=f'Mean {np.mean(bicubic_psnrs):.1f} dB')
ax2.set_title('Bicubic PSNR distribution across patches')
ax2.set_xlabel('PSNR (dB)')
ax2.legend()

plt.tight_layout()
hist_path = os.path.join(OUT_DIR, 'distributions.png')
plt.savefig(hist_path, dpi=120, bbox_inches='tight')
print(f"Histogram saved to: {hist_path}")

print("\n── Summary ──────────────────────────────────────")
print(f"  Bicubic PSNR        : {np.mean(bicubic_psnrs):.2f} dB")
print(f"  Mean-floor PSNR     : {np.mean(model_floor_psnrs):.2f} dB")
print(f"  LR↔HR correlation   : {np.mean(correlations):.4f}")
print(f"  Current model PSNR  : ~15.3 dB")
if np.mean(bicubic_psnrs) >= 14.5:
    print(f"\n  ⚠ Bicubic baseline is {np.mean(bicubic_psnrs):.1f} dB — model is not beating bicubic.")
    print(f"    The model is acting as a learned bicubic upsampler, not SR.")
if np.mean(correlations) < 0.5:
    print(f"\n  ⚠ LR↔HR correlation {np.mean(correlations):.3f} is low.")
    print(f"    Degradation is destroying too much signal for the model to recover.")
if np.mean(correlations) >= 0.5:
    print(f"\n  ✓ LR↔HR correlation {np.mean(correlations):.3f} is acceptable.")
    print(f"    Signal is present — problem is in training, not data.")