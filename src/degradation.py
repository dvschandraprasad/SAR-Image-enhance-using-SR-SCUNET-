import os
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
SCALE_FACTOR   = 4          # SR upscale factor (LR will be 1/4 the HR size)
PSF_SIGMA      = 1.2        # Gaussian PSF blur sigma (in LR pixels)
SPECKLE_LOOKS  = 4          # Equivalent number of looks for speckle simulation
NOISE_STD      = 0.01       # Additive Gaussian noise std (normalized 0-1 space)
PATCH_SIZE     = 128        # HR patch size in pixels
PATCH_LR_SIZE  = PATCH_SIZE // SCALE_FACTOR   # = 32
NUM_PATCHES    = 16         # Number of patches to visualize/test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')
OUT_DIR = os.path.join(DATA_DIR, 'samples', 'degradation_preview')
os.makedirs(OUT_DIR, exist_ok=True)

tif_path = os.path.join(GEOTIFF_DIR, 'CAPELLA_C13_SP_GEO_HH_20250830031430_20250830031509.tif')


def load_valid_patch(src, patch_size, max_attempts=200):
    """Randomly sample a patch that contains actual SAR data (not black border)."""
    h, w = src.height, src.width
    for _ in range(max_attempts):
        row = np.random.randint(0, h - patch_size)
        col = np.random.randint(0, w - patch_size)
        patch = src.read(1, window=rasterio.windows.Window(col, row, patch_size, patch_size))
        # Reject patches that are mostly black (border / no-data)
        if np.count_nonzero(patch) > 0.8 * patch_size * patch_size:
            return patch.astype(np.float32)
    raise RuntimeError("Could not find valid patch after max_attempts")


def normalize(patch):
    """Normalize uint16 SAR patch to [0, 1] using log1p + percentile stretch."""
    patch = np.log1p(patch)
    p1, p99 = np.percentile(patch, 1), np.percentile(patch, 99)
    patch = np.clip(patch, p1, p99)
    patch = (patch - p1) / (p99 - p1 + 1e-8)
    return patch.astype(np.float32)


def degrade(hr_patch):
    """
    Synthetic degradation pipeline:
      1. Gaussian PSF blur (sensor blur)
      2. Bicubic downsample by SCALE_FACTOR
      3. Speckle noise (multiplicative, gamma-distributed)
      4. Additive Gaussian noise
    Returns lr_patch in same normalized [0,1] space.
    """
    lr_size = hr_patch.shape[0] // SCALE_FACTOR

    # Step 1: PSF blur on HR
    blurred = gaussian_filter(hr_patch, sigma=PSF_SIGMA)

    # Step 2: Bicubic downsample
    lr = resize(blurred, (lr_size, lr_size), order=3, anti_aliasing=True,
                preserve_range=True).astype(np.float32)

    # Step 3: Speckle noise (multiplicative gamma noise, ENL = SPECKLE_LOOKS)
    speckle = np.random.gamma(shape=SPECKLE_LOOKS, scale=1.0 / SPECKLE_LOOKS,
                               size=lr.shape).astype(np.float32)
    lr = lr * speckle

    # Step 4: Additive Gaussian noise
    lr = lr + np.random.normal(0, NOISE_STD, lr.shape).astype(np.float32)
    lr = np.clip(lr, 0, 1)

    return lr


# ── Test on NUM_PATCHES random patches ────────────────────────────────────────
print(f"Loading patches from: {tif_path}")
np.random.seed(42)

fig, axes = plt.subplots(NUM_PATCHES, 3, figsize=(10, NUM_PATCHES * 3))
fig.suptitle(f'Degradation Pipeline (scale x{SCALE_FACTOR})  |  HR:{PATCH_SIZE}px  LR:{PATCH_LR_SIZE}px',
             fontsize=13, fontweight='bold')

with rasterio.open(tif_path) as src:
    for i in range(NUM_PATCHES):
        hr_raw = load_valid_patch(src, PATCH_SIZE)
        hr = normalize(hr_raw)
        lr = degrade(hr)

        # Upsample LR back to HR size for visual comparison
        lr_up = resize(lr, (PATCH_SIZE, PATCH_SIZE), order=3,
                       anti_aliasing=True, preserve_range=True)

        axes[i, 0].imshow(hr, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('HR (ground truth)', fontsize=8)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(lr, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'LR (degraded {PATCH_LR_SIZE}px)', fontsize=8)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(lr_up, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('LR upsampled (bicubic)', fontsize=8)
        axes[i, 2].axis('off')

        if i % 4 == 0:
            print(f"  Patch {i+1}/{NUM_PATCHES} done")

plt.tight_layout()
preview_path = os.path.join(OUT_DIR, 'degradation_preview.png')
plt.savefig(preview_path, dpi=120, bbox_inches='tight')
plt.show()
print(f"\nPreview saved to: {preview_path}")
print("Degradation pipeline validated.")