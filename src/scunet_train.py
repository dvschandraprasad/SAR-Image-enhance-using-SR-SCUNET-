import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import rasterio

import sar_dataset as sar_data
from sar_dataset import SARDataset
from scunet_model import SCUNet

# ── Config ─────────────────────────────────────────────────────────────────────
CFG = {
    # Data
    'patch_size'        : 128,         # HR patch size; LR = patch_size / scale = 64
    'scale'             : 2,
    'patches_per_image' : 200,
    'batch_size'        : 4,           # reduce to 2 if OOM on 16GB
    'num_workers'       : 4,
    'val_split'         : 0.1,

    # Training schedule
    'num_epochs'        : 100,
    'lr'                : 2e-4,
    'weight_decay'      : 1e-5,
    # MultiStepLR: decay by 0.5 at epochs 60 and 100.
    # Much more stable than CosineAnnealingWarmRestarts — no LR spikes.
    # If still plateaued at epoch 60, the 0.5x decay will push past it.
    'milestones'        : [30, 60, 90, 120],
    'lr_gamma'          : 0.5,
    'grad_clip'         : 1.0,
    'fft_warmup_epochs' : 80,

    # Loss weights
    # v1: 0.8/0.2 — plateaued at 16.8 dB
    # v2: SSIM added — caused NaN instability, abandoned
    # v3: 0.7/0.3 + warm restart — hit 21.46 dB but collapsed from NaN cascade
    # v4 (this run): 0.8/0.2, lower LR, MultiStepLR — stable convergence target
    # If plateau reappears before epoch 60, bump fft_weight to 0.3 manually.
    'mse_weight'        : 1.0,
    'l1_weight'         : 0.0,
    'fft_weight'        : 0.0,

    # SCUNet architecture -- paper config
    'nc'                : 96,
    'depth'             : (2, 2, 2, 2),
    'num_heads'         : 6,
    'window_size'       : 8,
    'mlp_ratio'         : 4.,
    'drop_path_rate'    : 0.0,

    # Logging
    'log_every'         : 10,
    'save_every'        : 10,
    'mixed_precision'   : True,

    # Reproducibility / resume
    'seed'              : 42,
    'resume'            : False,
    'run_name'          : 'scunet-128patch-2x-v17-progressive-ease',
    'curriculum_interp' : 'linear',   # 'step' or 'linear'
}

# Progressive degradation easing (epoch_start, PSF_SIGMA, SPECKLE_LOOKS, NOISE_STD, DEGRADE_BLEND)
DEGRADE_CURRICULUM = [
    (1,  0.22, 28, 0.00030, 0.65),
    (21, 0.18, 40, 0.00020, 0.50),
    (41, 0.14, 64, 0.00010, 0.35),
    (71, 0.10, 96, 0.00005, 0.25),
]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'data')
CKPT_DIR    = os.path.join(BASE_DIR, '..', 'checkpoints')
RUNS_DIR    = os.path.join(BASE_DIR, '..', 'runs')
GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')


# ── Charbonnier Loss ──────────────────────────────────────────────────────────
# Differentiable everywhere; behaves like L2 near zero and L1 for large errors.
# Consistently outperforms plain MSE for PSNR-targeted SR training.
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((sr - hr) ** 2 + self.eps2))


# ── FFT Frequency Loss ─────────────────────────────────────────────────────────

class FFTLoss(nn.Module):
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_fft = torch.fft.rfft2(sr.float(), norm='ortho')
        hr_fft = torch.fft.rfft2(hr.float(), norm='ortho')
        sr_mag = torch.abs(sr_fft).clamp(max=100.0)
        hr_mag = torch.abs(hr_fft).clamp(max=100.0)
        return torch.mean(torch.abs(sr_mag - hr_mag))


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.6, l1_weight=0.35, fft_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight  = l1_weight
        self.fft_weight = fft_weight
        self.mse        = nn.MSELoss()
        self.l1         = nn.L1Loss()
        self.fft        = FFTLoss()

    def forward(self, sr, hr, fft_weight=None):
        mse_loss = self.mse(sr, hr)
        l1_loss  = self.l1(sr, hr)
        fft_loss = self.fft(sr, hr)
        w_fft = self.fft_weight if fft_weight is None else fft_weight
        if w_fft <= 0.0 and self.l1_weight <= 0.0:
            total = self.mse_weight * mse_loss
            return total, mse_loss, torch.zeros_like(total), torch.zeros_like(total)
        # Preserve total weight = 1.0 as FFT ramps up.
        base = max(1e-8, self.mse_weight + self.l1_weight)
        w_base = max(0.0, 1.0 - w_fft)
        w_mse = w_base * (self.mse_weight / base)
        w_l1 = w_base * (self.l1_weight / base)
        total = w_mse * mse_loss + w_l1 * l1_loss + w_fft * fft_loss
        return total, mse_loss, l1_loss, fft_loss


# ── Helpers ────────────────────────────────────────────────────────────────────

class TensorPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def patches_for_image(tif_path, patch_size, max_patches):
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
    return min(max_patches, (h // patch_size) * (w // patch_size))


def compute_metrics(sr, hr):
    """
    Returns (mean_psnr, mean_ssim) over a batch.
    Both sr and hr are float32 tensors in [0,1], shape (B, 1, H, W).

    Note: val_psnr_vals and val_ssim_vals are used as the accumulator names
    throughout to avoid any collision with the skimage function names `psnr`
    and `ssim` imported at the top of this file.
    """
    sr_np = sr.detach().cpu().float().numpy()
    hr_np = hr.detach().cpu().float().numpy()
    batch_psnr, batch_ssim = [], []
    for i in range(sr_np.shape[0]):
        s, h = sr_np[i, 0], hr_np[i, 0]
        batch_psnr.append(psnr(h, s, data_range=1.0))
        batch_ssim.append(ssim(h, s, data_range=1.0))
    return np.mean(batch_psnr), np.mean(batch_ssim)


def upsample_bicubic(lr_batch, scale):
    return F.interpolate(lr_batch, scale_factor=scale, mode='bicubic', align_corners=False)


def get_curriculum_stage(epoch):
    if CFG.get('curriculum_interp', 'step') == 'step':
        stage = DEGRADE_CURRICULUM[0]
        for item in DEGRADE_CURRICULUM:
            if epoch >= item[0]:
                stage = item
            else:
                break
        return stage

    # Linear interpolation between stage anchors avoids abrupt domain shifts.
    if epoch <= DEGRADE_CURRICULUM[0][0]:
        return DEGRADE_CURRICULUM[0]
    if epoch >= DEGRADE_CURRICULUM[-1][0]:
        return DEGRADE_CURRICULUM[-1]

    for idx in range(len(DEGRADE_CURRICULUM) - 1):
        s0 = DEGRADE_CURRICULUM[idx]
        s1 = DEGRADE_CURRICULUM[idx + 1]
        e0, psf0, looks0, noise0, blend0 = s0
        e1, psf1, looks1, noise1, blend1 = s1
        if e0 <= epoch < e1:
            t = float(epoch - e0) / float(max(1, e1 - e0))
            psf_sigma = psf0 + t * (psf1 - psf0)
            speckle_looks = int(round(looks0 + t * (looks1 - looks0)))
            noise_std = noise0 + t * (noise1 - noise0)
            degrade_blend = blend0 + t * (blend1 - blend0)
            return (epoch, psf_sigma, speckle_looks, noise_std, degrade_blend)

    stage = DEGRADE_CURRICULUM[0]
    for item in DEGRADE_CURRICULUM:
        if epoch >= item[0]:
            stage = item
        else:
            break
    return stage


def apply_degradation_stage(stage):
    _, psf_sigma, speckle_looks, noise_std, degrade_blend = stage
    sar_data.PSF_SIGMA = float(psf_sigma)
    sar_data.SPECKLE_LOOKS = int(speckle_looks)
    sar_data.NOISE_STD = float(noise_std)
    sar_data.DEGRADE_BLEND = float(degrade_blend)


if __name__ == '__main__':
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = CFG['mixed_precision'] and device.type == 'cuda'
    print(f"Device: {device}")

    # Keep validation target fixed at the final curriculum stage.
    apply_degradation_stage(DEGRADE_CURRICULUM[-1])

    # ── Data ───────────────────────────────────────────────────────────────────
    tif_paths = sorted([os.path.join(GEOTIFF_DIR, f)
                        for f in os.listdir(GEOTIFF_DIR) if f.endswith('.tif')])
    print(f"GeoTIFFs found: {len(tif_paths)}")

    per_image_patches, total_patches = [], 0
    for p in tif_paths:
        n = patches_for_image(p, CFG['patch_size'], CFG['patches_per_image'])
        per_image_patches.append(n)
        total_patches += n
        print(f"  {os.path.basename(p)}: {n} patches")
    print(f"Total patches: {total_patches}")

    train_source = ConcatDataset([
        SARDataset([p], patch_size=CFG['patch_size'], patches_per_image=n, augment=True)
        for p, n in zip(tif_paths, per_image_patches)
    ])
    val_source = ConcatDataset([
        SARDataset([p], patch_size=CFG['patch_size'], patches_per_image=n, augment=False)
        for p, n in zip(tif_paths, per_image_patches)
    ])

    val_size   = max(1, int(total_patches * CFG['val_split']))
    train_size = total_patches - val_size
    train_dataset, val_dataset = random_split(
        train_source, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Build a fixed validation set once to make PSNR progression meaningful.
    _, fixed_val_subset = random_split(
        val_source,
        [len(val_source) - val_size, val_size],
        generator=torch.Generator().manual_seed(CFG['seed'])
    )
    fixed_val_pairs = [fixed_val_subset[i] for i in range(len(fixed_val_subset))]
    val_dataset = TensorPairDataset(fixed_val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=0,
                              pin_memory=True)
    print(f"Train patches: {train_size} | Val patches: {val_size}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = SCUNet(
        in_channels   = 1,
        out_channels  = 1,
        img_size      = CFG['patch_size'] // CFG['scale'],   # 64
        nc            = CFG['nc'],
        depth         = CFG['depth'],
        num_heads     = CFG['num_heads'],
        window_size   = CFG['window_size'],
        mlp_ratio     = CFG['mlp_ratio'],
        drop_path_rate= CFG['drop_path_rate'],
        scale         = CFG['scale'],
        residual_output = True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # sanity forward pass
    lr_size = CFG['patch_size'] // CFG['scale']
    with torch.no_grad():
        dummy = torch.rand(2, 1, lr_size, lr_size).to(device)
        out   = model(dummy)
        out   = upsample_bicubic(dummy, CFG['scale']) + out
        print(f"Model check -- input ({lr_size},{lr_size}) → output ({out.shape[2]},{out.shape[3]})")
        print(f"  mean: {out.mean():.4f}  std: {out.std():.4f}  (expect mean ~0.60-0.72, std >0.05)")
        assert out.shape == (2, 1, CFG['patch_size'], CFG['patch_size']), \
            f"Shape mismatch: got {out.shape}"

    # ── Optimizer ──────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in ['bias', 'norm'])]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in ['bias', 'norm'])]
    optimizer = optim.AdamW([
        {'params': decay_params,    'weight_decay': CFG['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=CFG['lr'], betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = CFG['milestones'],
        gamma      = CFG['lr_gamma'],
    )
    criterion = CombinedLoss(
        mse_weight=CFG['mse_weight'],
        l1_weight=CFG['l1_weight'],
        fft_weight=CFG['fft_weight']
    )
    scaler    = torch.amp.GradScaler(enabled=amp_enabled)

    run_name = CFG['run_name']
    writer   = SummaryWriter(log_dir=os.path.join(RUNS_DIR, run_name))
    print(f"TensorBoard run: {run_name}")

    # ── Resume from last periodic checkpoint if available ─────────────────────
    # Scans checkpoints/ for the highest-numbered scunet_epochXXX.pth and
    # resumes from there so a crash or manual stop doesn't lose progress.
    best_val_psnr = 0.0
    best_val_gain_psnr = float('-inf')
    start_epoch   = 1

    existing = sorted([
        f for f in os.listdir(CKPT_DIR)
        if f.startswith('scunet_epoch') and f.endswith('.pth')
    ])
    if CFG['resume'] and existing:
        last_ckpt = os.path.join(CKPT_DIR, existing[-1])
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        # Override LR in loaded optimizer state — checkpoint was saved with old LR.
        # Without this, the resumed run ignores CFG['lr'] entirely.
        for pg in optimizer.param_groups:
            pg['lr'] = CFG['lr']
        start_epoch   = ckpt['epoch'] + 1
        best_val_psnr = ckpt.get('val_psnr', 0.0)
        best_val_gain_psnr = ckpt.get('val_gain_psnr', float('-inf'))
        # Replay MultiStepLR state by stepping through past epochs.
        for _ in range(ckpt['epoch']):
            scheduler.step()
        print(f"Resumed from {last_ckpt}  (epoch {ckpt['epoch']}, "
              f"val PSNR {best_val_psnr:.2f} dB)")
        print(f"Continuing from epoch {start_epoch} …")
    else:
        print("Training from scratch (resume disabled).")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG['num_epochs'] + 1):
        stage = get_curriculum_stage(epoch)
        apply_degradation_stage(stage)

        model.train()
        train_losses, train_mses, train_l1s, train_ffts = [], [], [], []
        nan_batches = 0

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            warmup_ratio = min(1.0, epoch / CFG['fft_warmup_epochs'])
            fft_weight = CFG['fft_weight'] * (warmup_ratio ** 2)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                base_batch = upsample_bicubic(lr_batch, CFG['scale'])
                residual_batch = model(lr_batch)
                sr_batch = base_batch + residual_batch
                loss, mse_loss, l1_loss, fft_loss = criterion(sr_batch.float(), hr_batch.float(), fft_weight=fft_weight)

            # NaN/inf guard -- skip bad batch without touching scaler state.
            # Do NOT call scaler.update() here -- it requires scaler.scale() to
            # have been called first in this step, which isn't true on a skip.
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad()
                if nan_batches <= 3:
                    print(f"  WARNING: NaN/inf loss at epoch {epoch} "
                          f"batch {batch_idx+1} -- skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_mses.append(mse_loss.item())
            train_l1s.append(l1_loss.item())
            train_ffts.append(fft_loss.item())

            if (batch_idx + 1) % CFG['log_every'] == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}  "
                        f"MSE: {mse_loss.item():.4f}  "
                      f"L1: {l1_loss.item():.4f}  FFT: {fft_loss.item():.4f}  "
                      f"w_fft: {fft_weight:.3f}")

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_mses, val_l1s, val_ffts = [], [], [], []
        # Use unambiguous accumulator names to avoid collision with the
        # skimage `psnr` and `ssim` function names imported at the top.
        val_psnr_vals, val_ssim_vals = [], []
        bicubic_psnr_vals, bicubic_ssim_vals = [], []

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    base_batch = upsample_bicubic(lr_batch, CFG['scale'])
                    residual_batch = model(lr_batch)
                    sr_batch = base_batch + residual_batch
                    loss, mse_loss, l1_loss, fft_loss = criterion(sr_batch.float(), hr_batch.float(), fft_weight=CFG['fft_weight'])
                val_losses.append(loss.item())
                val_mses.append(mse_loss.item())
                val_l1s.append(l1_loss.item())
                val_ffts.append(fft_loss.item())
                batch_psnr, batch_ssim = compute_metrics(sr_batch.float().clamp(0.0, 1.0), hr_batch.float())
                bicubic_psnr, bicubic_ssim = compute_metrics(base_batch.float().clamp(0.0, 1.0), hr_batch.float())
                val_psnr_vals.append(batch_psnr)
                val_ssim_vals.append(batch_ssim)
                bicubic_psnr_vals.append(bicubic_psnr)
                bicubic_ssim_vals.append(bicubic_ssim)

        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        avg_val_loss   = np.mean(val_losses)
        avg_val_mse    = np.mean(val_mses)
        avg_val_l1     = np.mean(val_l1s)
        avg_val_fft    = np.mean(val_ffts)
        finite_psnrs   = [v for v in val_psnr_vals if np.isfinite(v)]
        avg_val_psnr   = np.mean(finite_psnrs) if finite_psnrs else float('nan')
        finite_ssims   = [v for v in val_ssim_vals if np.isfinite(v)]
        avg_val_ssim   = np.mean(finite_ssims) if finite_ssims else float('nan')
        avg_bicubic_psnr = np.mean([v for v in bicubic_psnr_vals if np.isfinite(v)]) if bicubic_psnr_vals else float('nan')
        avg_bicubic_ssim = np.mean([v for v in bicubic_ssim_vals if np.isfinite(v)]) if bicubic_ssim_vals else float('nan')
        avg_gain_psnr = (avg_val_psnr - avg_bicubic_psnr) if np.isfinite(avg_val_psnr) and np.isfinite(avg_bicubic_psnr) else float('nan')
        avg_gain_ssim = (avg_val_ssim - avg_bicubic_ssim) if np.isfinite(avg_val_ssim) and np.isfinite(avg_bicubic_ssim) else float('nan')

        print(f"Epoch {epoch:03d}/{CFG['num_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} "
              f"(MSE: {avg_val_mse:.4f}  L1: {avg_val_l1:.4f}  FFT: {avg_val_fft:.4f}) | "
              f"PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f} | "
              f"Bicubic PSNR: {avg_bicubic_psnr:.2f} dB | Bicubic SSIM: {avg_bicubic_ssim:.4f}"
              f" | Gain PSNR: {avg_gain_psnr:.2f} dB | Gain SSIM: {avg_gain_ssim:.4f}"
              f" | Curriculum: sigma={stage[1]:.2f}, looks={stage[2]}, noise={stage[3]:.5f}, blend={stage[4]:.2f}"
              + (f" | NaN skipped: {nan_batches}" if nan_batches else ""))

        writer.add_scalar('train/loss',  avg_train_loss,      epoch)
        writer.add_scalar('train/mse',   np.mean(train_mses) if train_mses else float('nan'), epoch)
        writer.add_scalar('train/l1',    np.mean(train_l1s) if train_l1s else float('nan'), epoch)
        writer.add_scalar('train/fft',   np.mean(train_ffts) if train_ffts else float('nan'), epoch)
        writer.add_scalar('val/loss',    avg_val_loss,         epoch)
        writer.add_scalar('val/mse',     avg_val_mse,          epoch)
        writer.add_scalar('val/l1',      avg_val_l1,           epoch)
        writer.add_scalar('val/fft',     avg_val_fft,          epoch)
        writer.add_scalar('val/psnr',    avg_val_psnr,         epoch)
        writer.add_scalar('val/ssim',    avg_val_ssim,         epoch)
        writer.add_scalar('val/bicubic_psnr', avg_bicubic_psnr, epoch)
        writer.add_scalar('val/bicubic_ssim', avg_bicubic_ssim, epoch)
        writer.add_scalar('val/gain_psnr', avg_gain_psnr,       epoch)
        writer.add_scalar('val/gain_ssim', avg_gain_ssim,       epoch)
        writer.add_scalar('lr',          scheduler.get_last_lr()[0], epoch)

        if epoch % CFG['save_every'] == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'scunet_epoch{epoch:03d}.pth')
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'val_psnr'        : avg_val_psnr,
                'val_gain_psnr'   : avg_gain_psnr,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        is_best = False
        if np.isfinite(avg_gain_psnr):
            if avg_gain_psnr > best_val_gain_psnr + 1e-8:
                is_best = True
            elif abs(avg_gain_psnr - best_val_gain_psnr) <= 1e-8 and np.isfinite(avg_val_psnr) and avg_val_psnr > best_val_psnr:
                is_best = True
        elif best_val_gain_psnr == float('-inf') and np.isfinite(avg_val_psnr) and avg_val_psnr > best_val_psnr:
            is_best = True

        if is_best:
            best_val_psnr = avg_val_psnr
            best_val_gain_psnr = avg_gain_psnr
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'val_psnr'        : avg_val_psnr,
                'val_bicubic_psnr': avg_bicubic_psnr,
                'val_gain_psnr'   : avg_gain_psnr,
            }, os.path.join(CKPT_DIR, 'best_scunet.pth'))
            print(f"  Best model saved (Gain PSNR: {best_val_gain_psnr:.2f} dB, PSNR: {best_val_psnr:.2f} dB)")

    writer.close()
    print("Training complete.")
    