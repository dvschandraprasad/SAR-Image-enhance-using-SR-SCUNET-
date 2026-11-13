"""
inference.py — Full-scene SR inference for SCUNet v16 (2× scale, residual output)
Run from src/: python inference.py --input <path/to/scene.tif>
  or from project root: python src/inference.py --input <path/to/scene.tif>

Strategy:
  - Reads the full GeoTIFF scene
  - Normalizes with log1p + p1/p99 stretch (matches sar_dataset.py normalize_joint())
  - Generates a degraded LR version of the full scene (matches training degradation)
  - Runs SCUNet tile-by-tile over the LR scene (64×64 LR tiles → 128×128 SR tiles)
    using the residual formulation: sr_tile = bicubic(lr_tile) + model(lr_tile)
  - Reconstructs the full SR output and writes a GeoTIFF with updated geotransform

Degradation constants must stay in sync with sar_dataset.py:
  PSF_SIGMA=0.20, SPECKLE_LOOKS=40, NOISE_STD=0.0002, DEGRADE_BLEND=0.5, SCALE=2
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).parent))
from scunet_model import SCUNet


# ── Constants — must match sar_dataset.py ────────────────────────────────────
SCALE         = 2
LR_TILE       = 64           # network input tile size (HR patch / scale)
HR_TILE       = LR_TILE * SCALE   # = 128

PSF_SIGMA     = 0.20
SPECKLE_LOOKS = 40
NOISE_STD     = 0.0002
DEGRADE_BLEND = 0.5   # half noisy LR, half clean LR — matches sar_dataset.py


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> SCUNet:
    model = SCUNet(
        in_channels    = 1,
        out_channels   = 1,
        img_size       = LR_TILE,
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


def normalize_scene(arr: np.ndarray):
    """
    log1p → percentile [1, 99] clip → [0, 1] linear stretch.
    Returns (normalized float32, p1_log, p99_log) for inversion.
    Mirrors sar_dataset.py normalize_joint() applied to the HR side.
    """
    log_arr = np.log1p(arr.astype(np.float32))
    p1      = float(np.percentile(log_arr, 1))
    p99     = float(np.percentile(log_arr, 99))
    out     = np.clip(log_arr, p1, p99)
    out     = (out - p1) / (p99 - p1 + 1e-8)
    return out.astype(np.float32), p1, p99


def denormalize_scene(norm_arr: np.ndarray, p1_log: float, p99_log: float) -> np.ndarray:
    """Invert normalize_scene() → uint16 amplitude."""
    log_arr = norm_arr * (p99_log - p1_log + 1e-8) + p1_log
    linear  = np.expm1(log_arr)
    return np.clip(linear, 0, 65535).astype(np.uint16)


def make_preview_uint8(arr_uint16: np.ndarray) -> np.ndarray:
    """Build a display-ready uint8 preview using nonzero percentile stretch."""
    arr = arr_uint16.astype(np.float32)
    mask = arr > 0
    if not np.any(mask):
        return np.zeros_like(arr_uint16, dtype=np.uint8)

    lo = float(np.percentile(arr[mask], 2))
    hi = float(np.percentile(arr[mask], 98))
    if hi <= lo:
        hi = lo + 1.0

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def make_display_uint16(arr_uint16: np.ndarray) -> np.ndarray:
    """
    Build a display-optimized uint16 image using nonzero percentile stretch.
    This is for visualization only; scientific values remain in the main SR TIFF.
    """
    arr = arr_uint16.astype(np.float32)
    mask = arr > 0
    if not np.any(mask):
        return np.zeros_like(arr_uint16, dtype=np.uint16)

    lo = float(np.percentile(arr[mask], 2))
    hi = float(np.percentile(arr[mask], 98))
    if hi <= lo:
        hi = lo + 1.0

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 65535.0).astype(np.uint16)


def write_preview_png(arr_uint16: np.ndarray, output_path: Path) -> None:
    """Write a PNG preview beside the GeoTIFF for quick visual checks."""
    preview_path = output_path.with_name(f"{output_path.stem}_preview.png")
    preview_img = make_preview_uint8(arr_uint16)

    try:
        import imageio.v2 as imageio
        imageio.imwrite(preview_path, preview_img)
    except Exception as exc:
        print(f"Preview PNG skipped for {output_path.name}: {exc}")
        return

    print(f"Preview PNG        : {preview_path}")


def write_display_tiff(arr_uint16: np.ndarray, output_path: Path,
                       profile: dict, transform: Affine, crs) -> None:
    """Write a display-optimized GeoTIFF beside the scientific SR output."""
    display_path = output_path.with_name(f"{output_path.stem}_display.tif")
    display_arr = make_display_uint16(arr_uint16)

    display_profile = profile.copy()
    display_profile.update(
        dtype=rasterio.uint16,
        count=1,
        height=display_arr.shape[0],
        width=display_arr.shape[1],
        transform=transform,
        crs=crs,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    display_profile.pop('bigtiff', None)

    with rasterio.open(display_path, 'w', **display_profile) as dst:
        dst.write(display_arr, 1)

    print(f"Display GeoTIFF    : {display_path}")


def degrade_scene(hr_raw: np.ndarray) -> np.ndarray:
    """
    Apply the degradation pipeline from sar_dataset.py degrade() to raw HR
    amplitude. Returns raw LR amplitude (NOT normalized).

    Must stay in sync with sar_dataset.py degrade():
      1. Guard NaN/inf
      2. PSF blur (Gaussian)
      3. Bicubic downsample ×2
      4. Multiplicative gamma speckle (clipped to [0.8, 1.25])
      5. AWGN
      6. DEGRADE_BLEND mix of noisy and clean LR
    """
    hr = hr_raw.astype(np.float32)
    hr = np.where(np.isfinite(hr), hr, 0.0)

    lr_h = hr.shape[0] // SCALE
    lr_w = hr.shape[1] // SCALE

    blurred = gaussian_filter(hr, sigma=PSF_SIGMA)
    lr      = resize(blurred, (lr_h, lr_w), order=3,
                     anti_aliasing=True, preserve_range=True).astype(np.float32)

    speckle = np.random.gamma(shape=SPECKLE_LOOKS,
                              scale=1.0 / SPECKLE_LOOKS,
                              size=lr.shape).astype(np.float32)
    speckle  = np.clip(speckle, 0.8, 1.25)   # match training clamp
    noisy_lr = lr * speckle
    noisy_lr = noisy_lr + np.random.normal(0, NOISE_STD, noisy_lr.shape).astype(np.float32)
    lr = DEGRADE_BLEND * noisy_lr + (1.0 - DEGRADE_BLEND) * lr
    lr = np.clip(lr, 0, None)   # clip negatives only
    return lr.astype(np.float32)


def normalize_joint_scene(hr_raw: np.ndarray, lr_raw: np.ndarray):
    """
    Joint normalization matching sar_dataset.py normalize_joint().
    Both HR and LR are scaled using stats derived from HR only, so the
    model sees LR data in the same amplitude space it was trained on.
    Returns (norm_hr, norm_lr, p1_log, p99_log) — the last two are needed
    to invert the normalization after inference.
    """
    hr = hr_raw.astype(np.float32)
    hr = np.where(np.isfinite(hr), hr, 0.0)
    hr = np.clip(hr, 0, 10000.0)
    hr = np.log1p(hr)

    lr = lr_raw.astype(np.float32)
    lr = np.where(np.isfinite(lr), lr, 0.0)
    lr = np.clip(lr, 0, 10000.0)
    lr = np.log1p(lr)

    p1  = float(np.percentile(hr, 1))
    p99 = float(np.percentile(hr, 99))

    norm_hr = np.clip(hr, p1, p99)
    norm_hr = (norm_hr - p1) / (p99 - p1 + 1e-8)

    # LR uses same scale as HR (joint) — speckle can push values above 1, clip.
    norm_lr = (lr - p1) / (p99 - p1 + 1e-8)
    norm_lr = np.clip(norm_lr, 0.0, 1.0)

    return norm_hr.astype(np.float32), norm_lr.astype(np.float32), p1, p99


def smooth_sr_scene(sr_norm: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply masked Gaussian smoothing on normalized SR output.
    Uses weight normalization to avoid dark bleed from zero/no-data borders.
    """
    if sigma <= 0:
        return np.clip(sr_norm, 0.0, 1.0).astype(np.float32)

    valid = (sr_norm > 0).astype(np.float32)
    num = gaussian_filter(sr_norm, sigma=sigma)
    den = gaussian_filter(valid, sigma=sigma)
    out = np.where(den > 1e-6, num / (den + 1e-8), 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def pad_to_multiple(arr: np.ndarray, multiple: int):
    """Reflect-pad (H, W) array so both dims are exact multiples of `multiple`."""
    h, w   = arr.shape
    pad_h  = (multiple - h % multiple) % multiple
    pad_w  = (multiple - w % multiple) % multiple
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, pad_h, pad_w


def upsample_bicubic_np(patch: np.ndarray, scale: int) -> np.ndarray:
    """Bicubic upsample a single (H, W) float32 patch using torch."""
    t = torch.from_numpy(patch[None, None]).float()
    u = F.interpolate(t, scale_factor=scale, mode='bicubic', align_corners=False)
    return u.squeeze().numpy()


# ── Main inference routine ────────────────────────────────────────────────────

def infer_single_scene(input_path: Path, output_path: Path, model: SCUNet,
                       device: torch.device, args) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInput  : {input_path}")
    print(f"Output : {output_path}")

    # ── Read scene ────────────────────────────────────────────────────────────
    with rasterio.open(input_path) as src:
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        scene_raw = src.read(1)   # uint16 (H, W)

    H, W = scene_raw.shape
    print(f"Scene size: {H} × {W} pixels")

    # ── Build LR scene then joint-normalize (matches training pipeline) ────────
    if args.use_raw_lr:
        print("Using raw input as LR (--use_raw_lr) — skipping degradation.")
        # No degraded LR available; fall back to independent HR normalization.
        norm_scene, p1_log, p99_log = normalize_scene(scene_raw)
        lr_scene = norm_scene
    else:
        print("Generating LR via degradation pipeline (matches training) ...")
        lr_raw = degrade_scene(scene_raw)           # raw amplitude LR
        norm_scene, lr_scene, p1_log, p99_log = normalize_joint_scene(scene_raw, lr_raw)

    lr_H, lr_W = lr_scene.shape
    print(f"LR scene size: {lr_H} × {lr_W}")

    # ── Pad LR to exact tile multiple ─────────────────────────────────────────
    lr_padded, _pad_h, _pad_w = pad_to_multiple(lr_scene, LR_TILE)
    lr_ph, lr_pw = lr_padded.shape

    tiles_y     = lr_ph // LR_TILE
    tiles_x     = lr_pw // LR_TILE
    total_tiles = tiles_y * tiles_x
    print(f"Tiling: {tiles_y} × {tiles_x} = {total_tiles} LR tiles ({LR_TILE}×{LR_TILE} each)")

    # ── Pre-allocate SR output canvas ─────────────────────────────────────────
    sr_full = np.zeros((lr_ph * SCALE, lr_pw * SCALE), dtype=np.float32)

    # ── Tile inference with batching ──────────────────────────────────────────
    batch_lr  = []
    batch_pos = []   # (tile_row, tile_col)

    def flush_batch():
        if not batch_lr:
            return
        # Stack to (B, 1, LR_TILE, LR_TILE)
        x = torch.from_numpy(
            np.stack(batch_lr)[:, np.newaxis]
        ).float().to(device)

        with torch.no_grad():
            # Residual formulation: sr = bicubic(lr) + model(lr)
            base     = F.interpolate(x, scale_factor=SCALE,
                                     mode='bicubic', align_corners=False)
            residual = model(x)
            sr       = (base + residual).clamp(0.0, 1.0)

        sr_np = sr.squeeze(1).cpu().float().numpy()   # (B, HR_TILE, HR_TILE)

        for (tr, tc), patch in zip(batch_pos, sr_np):
            r0 = tr * HR_TILE
            c0 = tc * HR_TILE
            sr_full[r0:r0 + HR_TILE, c0:c0 + HR_TILE] = patch

    with tqdm(total=total_tiles, desc='Running SR', unit='tile') as pbar:
        for tr in range(tiles_y):
            for tc in range(tiles_x):
                r0    = tr * LR_TILE
                c0    = tc * LR_TILE
                patch = lr_padded[r0:r0 + LR_TILE, c0:c0 + LR_TILE]
                batch_lr.append(patch)
                batch_pos.append((tr, tc))

                if len(batch_lr) == args.batch_size:
                    flush_batch()
                    batch_lr.clear()
                    batch_pos.clear()

                pbar.update(1)

    flush_batch()   # flush remaining tiles

    # ── Crop padding and denormalize ──────────────────────────────────────────
    sr_full   = sr_full[:lr_H * SCALE, :lr_W * SCALE]

    if args.smooth_sigma > 0:
        print(f"Applying post-SR smoothing (sigma={args.smooth_sigma:.2f}) ...")
        sr_full = smooth_sr_scene(sr_full, sigma=float(args.smooth_sigma))

    sr_uint16 = denormalize_scene(sr_full, p1_log, p99_log)

    # ── Update geotransform for 2× higher resolution ──────────────────────────
    new_transform = Affine(
        transform.a / SCALE,   # pixel width  (positive → smaller)
        transform.b,
        transform.c,           # top-left x unchanged
        transform.d,
        transform.e / SCALE,   # pixel height (negative → smaller magnitude)
        transform.f,           # top-left y unchanged
    )

    # ── Write output GeoTIFF ──────────────────────────────────────────────────
    profile.update(
        dtype      = rasterio.uint16,
        count      = 1,
        height     = sr_uint16.shape[0],
        width      = sr_uint16.shape[1],
        transform  = new_transform,
        crs        = crs,
        compress   = 'lzw',
        tiled      = True,
        blockxsize = 256,
        blockysize = 256,
    )
    profile.pop('bigtiff', None)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(sr_uint16, 1)

    if not args.no_display_tiff:
        write_display_tiff(
            sr_uint16,
            output_path,
            profile,
            new_transform,
            crs,
        )

    if not args.no_preview_png:
        write_preview_png(sr_uint16, output_path)

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSR GeoTIFF written : {output_path}  ({size_mb:.1f} MB)")
    print(f"Output dimensions  : {sr_uint16.shape[0]} × {sr_uint16.shape[1]}")
    print(f"Approx pixel size  : {abs(new_transform.a):.4f} m  "
          f"(original: {abs(transform.a):.4f} m)")


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.very_smooth and args.smooth_sigma <= 0:
        args.smooth_sigma = 1.8
        print("Using very smooth preset: smooth_sigma=1.8")

    model = load_model(args.checkpoint, device)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in {'.tif', '.tiff'}:
            raise ValueError(
                f"--input must point to a .tif/.tiff file, got: {input_path}"
            )
        output_path = Path(args.output) if args.output else \
            input_path.parent / (input_path.stem + '_SR.tif')
        infer_single_scene(input_path, output_path, model, device, args)
        return

    tif_paths = sorted(
        [
            p for p in input_path.rglob('*')
            if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}
        ]
    )
    if not tif_paths:
        raise ValueError(
            f"No .tif/.tiff files found under directory: {input_path}"
        )

    if args.output:
        output_dir = Path(args.output)
        if output_dir.suffix.lower() in {'.tif', '.tiff'}:
            raise ValueError(
                "When --input is a directory, --output must be a directory path, "
                "not a .tif/.tiff file."
            )
    else:
        output_dir = input_path.parent / f"{input_path.name}_SR"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDirectory mode: found {len(tif_paths)} GeoTIFF files")
    print(f"Input dir  : {input_path}")
    print(f"Output dir : {output_dir}")

    success_count = 0
    failed = []

    for idx, tif_path in enumerate(tif_paths, start=1):
        rel = tif_path.relative_to(input_path)
        out_path = output_dir / rel.parent / f"{rel.stem}_SR.tif"
        print(f"\n[{idx}/{len(tif_paths)}] Processing {tif_path}")
        try:
            infer_single_scene(tif_path, out_path, model, device, args)
            success_count += 1
        except Exception as exc:
            failed.append((tif_path, str(exc)))
            print(f"Failed: {tif_path}\n  Reason: {exc}")

    print("\n" + "=" * 60)
    print("BATCH INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total files : {len(tif_paths)}")
    print(f"Succeeded   : {success_count}")
    print(f"Failed      : {len(failed)}")
    if failed:
        print("Failed files:")
        for p, msg in failed:
            print(f"  - {p}: {msg}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent

    parser = argparse.ArgumentParser(description='SCUNet SAR Super-Resolution Inference')
    _default_input = str(Path(__file__).resolve().parent.parent / 'data' / 'samples')
    parser.add_argument('--input',      type=str, default=_default_input,
                        help='Path to input GeoTIFF file OR directory containing GeoTIFF files')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output path. File mode: output .tif path. Directory mode: output directory path')
    parser.add_argument('--checkpoint', type=str,
                        default=str(BASE_DIR / '..' / 'checkpoints' / 'best_scunet.pth'))
    parser.add_argument('--use_raw_lr', action='store_true',
                        help='Treat input as already-degraded LR; skip degradation step')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of 64×64 LR tiles per GPU batch (tune to VRAM)')
    parser.add_argument('--smooth_sigma', type=float, default=0.0,
                        help='Post-SR Gaussian smoothing sigma in SR pixels (0 disables)')
    parser.add_argument('--very_smooth', action='store_true',
                        help='Visual preset for strong smoothing (sets --smooth_sigma 1.8 unless overridden)')
    parser.add_argument('--no_preview_png', action='store_true',
                        help='Disable writing auto-stretched PNG previews beside output GeoTIFFs')
    parser.add_argument('--no_display_tiff', action='store_true',
                        help='Disable writing display-optimized GeoTIFF beside output GeoTIFFs')

    args = parser.parse_args()
    run_inference(args)