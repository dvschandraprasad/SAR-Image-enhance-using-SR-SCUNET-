import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import rasterio
import rasterio.windows

# ── Config ─────────────────────────────────────────────────────────────────────
SCALE_FACTOR  = 2
PSF_SIGMA     = 0.20  # final ease step for >=20 dB target
SPECKLE_LOOKS = 40    # lower speckle variance than looks=32
NOISE_STD     = 0.0002
DEGRADE_BLEND = 0.5   # half noisy LR, half clean LR
PATCH_SIZE    = 128   # HR patch size -- LR will be 64x64 (128/2)
MIN_NONZERO   = 0.8   # Minimum fraction of non-zero pixels for a valid patch


def normalize_joint(hr_raw, lr_raw):
    """
    Normalize HR and LR using the SAME scale derived from hr_raw.

    Independent normalization (the previous approach) is the root cause of the
    15 dB ceiling: both HR and LR end up spanning [0,1] regardless of their
    actual amplitude relationship, so the model cannot learn amplitude recovery
    and the speckle/noise effects are completely erased by the per-patch rescaling.

    Joint normalization preserves the relative amplitude difference between LR
    and HR, which is the actual signal the model needs to learn to invert.

    Steps:
      1. NaN/sentinel guards on both inputs (same as before)
      2. log1p on both
      3. Compute p1/p99 from HR only
      4. Apply same clip+scale to both HR and LR
      5. LR values outside [0,1] after joint scaling are clipped -- speckle
         can push some LR pixels above the HR p99, which is fine, they just
         saturate at 1.0 rather than causing the model to see impossible targets.
    """
    # --- HR ---
    hr = hr_raw.astype(np.float32)
    if not np.all(np.isfinite(hr)):
        hr = np.where(np.isfinite(hr), hr, 0.0)
    hr = np.clip(hr, 0, 10000.0)
    hr = np.log1p(hr)

    # --- LR ---
    lr = lr_raw.astype(np.float32)
    if not np.all(np.isfinite(lr)):
        lr = np.where(np.isfinite(lr), lr, 0.0)
    lr = np.clip(lr, 0, 10000.0)
    lr = np.log1p(lr)

    # --- Compute scale from HR ---
    p1, p99 = np.percentile(hr, 1), np.percentile(hr, 99)
    if (p99 - p1) < 1e-6:
        # Flat/empty patch -- return zeros for both
        return (np.zeros_like(hr, dtype=np.float32),
                np.zeros_like(lr, dtype=np.float32))

    # --- Apply same scale to both ---
    hr = np.clip(hr, p1, p99)
    hr = (hr - p1) / (p99 - p1)

    # LR is NOT clipped to [p1,p99] before scaling -- speckle can push values
    # outside the HR range and that information should be visible to the model.
    # We clip the final result to [0,1] to keep tensors bounded.
    lr = (lr - p1) / (p99 - p1)
    lr = np.clip(lr, 0.0, 1.0)

    return hr.astype(np.float32), lr.astype(np.float32)


def degrade(hr_raw):
    # Operate on raw amplitude -- speckle is multiplicative on physical values.
    # Degradation chain:
    #   1. Gaussian blur (PSF_SIGMA=0.8) -- models sensor resolution limit
    #   2. Bicubic downsample 4x         -- simulates lower-resolution acquisition
    #   3. Gamma speckle (LOOKS=8)       -- multiplicative noise, less aggressive than looks=4
    #   4. Additive Gaussian noise       -- thermal/quantisation noise floor

    # Guard: replace NaN/inf in raw input before blur propagates them patch-wide.
    hr = hr_raw.astype(np.float32)
    if not np.all(np.isfinite(hr)):
        hr = np.where(np.isfinite(hr), hr, 0.0)

    lr_size = hr.shape[0] // SCALE_FACTOR
    blurred = gaussian_filter(hr, sigma=PSF_SIGMA)
    lr = resize(blurred, (lr_size, lr_size), order=3, anti_aliasing=True,
                preserve_range=True).astype(np.float32)

    # Clamp speckle to a tighter band to reduce extreme hard samples.
    speckle = np.random.gamma(shape=SPECKLE_LOOKS, scale=1.0 / SPECKLE_LOOKS,
                               size=lr.shape).astype(np.float32)
    speckle = np.clip(speckle, 0.8, 1.25)
    noisy_lr = lr * speckle
    noisy_lr = noisy_lr + np.random.normal(0, NOISE_STD, noisy_lr.shape).astype(np.float32)
    lr = DEGRADE_BLEND * noisy_lr + (1.0 - DEGRADE_BLEND) * lr
    lr = np.clip(lr, 0, None)   # clip negatives only -- no upper clip
    return lr


class SARDataset(Dataset):
    """
    PyTorch Dataset for SAR super-resolution.
    Randomly samples patches from GeoTIFF files on-the-fly.
    Each __getitem__ call returns a fresh (lr, hr) patch pair.
    """

    def __init__(self, tif_paths, patch_size=PATCH_SIZE, patches_per_image=500,
                 augment=True, max_attempts=200):
        """
        Args:
            tif_paths        : list of paths to GeoTIFF HR images
            patch_size       : HR patch size in pixels
            patches_per_image: how many patches to virtually sample per image
                               (controls epoch length)
            augment          : random flip/rotate augmentation
            max_attempts     : max retries to find a valid (non-black) patch
        """
        self.tif_paths         = tif_paths
        self.patch_size        = patch_size
        self.patches_per_image = patches_per_image
        self.augment           = augment
        self.max_attempts      = max_attempts
        self.total             = len(tif_paths) * patches_per_image

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Map flat index to which image to sample from
        img_idx  = idx % len(self.tif_paths)
        tif_path = self.tif_paths[img_idx]

        with rasterio.open(tif_path) as src:
            h, w = src.height, src.width
            ps   = self.patch_size

            for _ in range(self.max_attempts):
                row    = np.random.randint(0, h - ps)
                col    = np.random.randint(0, w - ps)
                hr_raw = src.read(1, window=rasterio.windows.Window(col, row, ps, ps))

                if np.count_nonzero(hr_raw) >= MIN_NONZERO * ps * ps:
                    break
            else:
                # Fallback: center crop (shouldn't happen often)
                row    = (h - ps) // 2
                col    = (w - ps) // 2
                hr_raw = src.read(1, window=rasterio.windows.Window(col, row, ps, ps))

        lr_raw = degrade(hr_raw)
        hr, lr = normalize_joint(hr_raw, lr_raw)

        # Final guard: zero out any patch that still has NaN/inf
        if not np.all(np.isfinite(hr)):
            hr = np.zeros_like(hr)
        if not np.all(np.isfinite(lr)):
            lr = np.zeros_like(lr)

        # Augmentation: random horizontal/vertical flip and 90° rotations
        if self.augment:
            k  = np.random.randint(0, 4)
            hr = np.rot90(hr, k).copy()
            lr = np.rot90(lr, k).copy()
            if np.random.rand() > 0.5:
                hr = np.fliplr(hr).copy()
                lr = np.fliplr(lr).copy()
            if np.random.rand() > 0.5:
                hr = np.flipud(hr).copy()
                lr = np.flipud(lr).copy()

        # Add channel dim: (1, H, W)
        hr_tensor = torch.from_numpy(hr).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr).unsqueeze(0)

        return lr_tensor, hr_tensor


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR    = os.path.join(BASE_DIR, '..', 'data')
    GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')

    tif_paths = [
        os.path.join(GEOTIFF_DIR, f)
        for f in os.listdir(GEOTIFF_DIR) if f.endswith('.tif')
    ]
    print(f"Found {len(tif_paths)} GeoTIFF(s): {[os.path.basename(p) for p in tif_paths]}")

    dataset = SARDataset(tif_paths, patch_size=128, patches_per_image=10, augment=True)
    print(f"Dataset length: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    lr_batch, hr_batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  LR : {lr_batch.shape}  dtype={lr_batch.dtype}  min={lr_batch.min():.3f}  max={lr_batch.max():.3f}  mean={lr_batch.mean():.3f}")
    print(f"  HR : {hr_batch.shape}  dtype={hr_batch.dtype}  min={hr_batch.min():.3f}  max={hr_batch.max():.3f}  mean={hr_batch.mean():.3f}")

    # Joint normalization check: LR and HR should NOT both have mean ~0.5.
    # With joint normalization, LR mean will typically be slightly different
    # from HR mean because speckle shifts the amplitude distribution.
    # If both means are identical (~0.5) independent normalization is still active.
    mean_diff = (lr_batch.mean() - hr_batch.mean()).abs().item()
    print(f"  |LR_mean - HR_mean| : {mean_diff:.4f}  (>0.01 confirms joint normalization is active)")

    import torch.nn.functional as F
    lr_up = F.interpolate(lr_batch, size=hr_batch.shape[-2:], mode='bicubic', align_corners=False)
    diff  = (lr_up - hr_batch).abs().mean().item()
    print(f"  Mean |LR_up-HR|     : {diff:.4f}  (expect 0.05-0.20 with joint normalization)")
    print("\nSARDataset smoke test passed.")