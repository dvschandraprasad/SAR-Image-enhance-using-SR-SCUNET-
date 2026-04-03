import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import rasterio

from sar_dataset import SARDataset
from model import RRDBNet

# ── Config ─────────────────────────────────────────────────────────────────────
CFG = {
    # Data
    'patch_size'       : 256,          # HR patch size; LR = 256/4 = 64
    'scale'            : 4,
    'patches_per_image': 100,
    'batch_size'       : 8,            # RRDB is pure CNN -- handles larger batch fine
    'num_workers'      : 4,
    'val_split'        : 0.1,

    # Training schedule
    'num_epochs'       : 100,
    'lr'               : 1e-4,         # conservative for 23-block depth
    'lr_decay_epochs'  : [50, 75, 90], # decay at 50/75/90 epochs, not 200/260
    'lr_decay_gamma'   : 0.5,
    'grad_clip'        : 1.0,          # CRITICAL -- was missing in old train.py

    # Loss
    'l1_weight'        : 0.8,
    'fft_weight'       : 0.2,          # FFT loss prevents mean-collapse on L1 alone

    # RRDB architecture -- only num_blocks changed vs old config
    'num_feat'         : 64,
    'num_blocks'       : 23,           # was 8 -- 23 is the ESRGAN/paper config (~16.7M params)
    'growth_rate'      : 32,

    # Logging
    'log_every'        : 10,
    'save_every'       : 10,
    'mixed_precision'  : True,
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'data')
CKPT_DIR    = os.path.join(BASE_DIR, '..', 'checkpoints')
RUNS_DIR    = os.path.join(BASE_DIR, '..', 'runs')
GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')


# ── FFT Frequency Loss ─────────────────────────────────────────────────────────
class FFTLoss(nn.Module):
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_fft = torch.fft.rfft2(sr.float(), norm='ortho')
        hr_fft = torch.fft.rfft2(hr.float(), norm='ortho')
        # Clamp magnitudes -- outlier pixels produce huge FFT coefficients that
        # spike the loss and cause gradient explosion before grad_clip can act
        sr_mag = torch.abs(sr_fft).clamp(max=100.0)
        hr_mag = torch.abs(hr_fft).clamp(max=100.0)
        return torch.mean(torch.abs(sr_mag - hr_mag))


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.8, fft_weight=0.2):
        super().__init__()
        self.l1_weight  = l1_weight
        self.fft_weight = fft_weight
        self.l1         = nn.L1Loss()
        self.fft        = FFTLoss()

    def forward(self, sr, hr):
        l1_loss  = self.l1(sr, hr)
        fft_loss = self.fft(sr, hr)
        total    = self.l1_weight * l1_loss + self.fft_weight * fft_loss
        return total, l1_loss, fft_loss


# ── Helpers ────────────────────────────────────────────────────────────────────
def patches_for_image(tif_path, patch_size, max_patches):
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
    return min(max_patches, (h // patch_size) * (w // patch_size))


def compute_metrics(sr, hr):
    sr_np = sr.detach().cpu().float().numpy()
    hr_np = hr.detach().cpu().float().numpy()
    psnr_vals, ssim_vals = [], []
    for i in range(sr_np.shape[0]):
        s = sr_np[i, 0]
        h = hr_np[i, 0]
        psnr_vals.append(psnr(h, s, data_range=1.0))
        ssim_vals.append(ssim(h, s, data_range=1.0))
    return np.mean(psnr_vals), np.mean(ssim_vals)


if __name__ == '__main__':
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    tif_paths = sorted([
        os.path.join(GEOTIFF_DIR, f)
        for f in os.listdir(GEOTIFF_DIR) if f.endswith('.tif')
    ])
    print(f"GeoTIFFs found: {len(tif_paths)}")

    per_image_patches, total_patches = [], 0
    for p in tif_paths:
        n = patches_for_image(p, CFG['patch_size'], CFG['patches_per_image'])
        per_image_patches.append(n)
        total_patches += n
        print(f"  {os.path.basename(p)}: {n} patches")
    print(f"Total patches: {total_patches}")

    datasets = [
        SARDataset([p], patch_size=CFG['patch_size'],
                   patches_per_image=n, augment=True)
        for p, n in zip(tif_paths, per_image_patches)
    ]
    full_dataset = ConcatDataset(datasets)

    val_size   = max(1, int(total_patches * CFG['val_split']))
    train_size = total_patches - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=CFG['num_workers'],
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=CFG['num_workers'],
                              pin_memory=True, persistent_workers=True)
    print(f"Train patches: {train_size} | Val patches: {val_size}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = RRDBNet(
        in_channels  = 1,
        out_channels = 1,
        num_feat     = CFG['num_feat'],
        num_blocks   = CFG['num_blocks'],
        growth_rate  = CFG['growth_rate'],
        scale        = CFG['scale'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Smoke check -- use actual LR size (64), not old 32
    lr_size = CFG['patch_size'] // CFG['scale']   # 256/4 = 64
    with torch.no_grad():
        dummy = torch.rand(2, 1, lr_size, lr_size).to(device)
        out   = model(dummy)
        print(f"Model check -- input (2,1,{lr_size},{lr_size}) → output {tuple(out.shape)}")
        print(f"  mean: {out.mean():.4f}  std: {out.std():.4f}  (expect mean ~0.62-0.72, std >0.05)")
    assert out.shape == (2, 1, CFG['patch_size'], CFG['patch_size']), \
        f"Shape mismatch: {out.shape}"

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=CFG['lr_decay_epochs'], gamma=CFG['lr_decay_gamma']
    )
    criterion = CombinedLoss(l1_weight=CFG['l1_weight'], fft_weight=CFG['fft_weight'])
    scaler    = torch.amp.GradScaler(enabled=CFG['mixed_precision'])

    run_name = 'rrdb23-256patch-v1'
    writer   = SummaryWriter(log_dir=os.path.join(RUNS_DIR, run_name))
    print(f"TensorBoard run: {run_name}")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_psnr = 0.0

    for epoch in range(1, CFG['num_epochs'] + 1):
        model.train()
        train_losses, train_l1s, train_ffts = [], [], []
        nan_batches = 0

        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=CFG['mixed_precision']):
                sr = model(lr)
                loss, l1_loss, fft_loss = criterion(sr.float(), hr.float())

            # Skip NaN/inf batches -- call scaler.update() to keep its state consistent
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad()
                scaler.update()
                if nan_batches <= 3:
                    print(f"  WARNING: NaN/inf loss at epoch {epoch} batch {batch_idx+1} -- skipping")
                continue

            scaler.scale(loss).backward()
            # unscale_ before clip_grad_norm_ -- REQUIRED, was missing in old train.py
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_l1s.append(l1_loss.item())
            train_ffts.append(fft_loss.item())

            if (batch_idx + 1) % CFG['log_every'] == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}  L1: {l1_loss.item():.4f}  FFT: {fft_loss.item():.4f}")

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_l1s, val_ffts, val_psnrs, val_ssims = [], [], [], [], []

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=CFG['mixed_precision']):
                    sr = model(lr)
                    loss, l1_loss, fft_loss = criterion(sr.float(), hr.float())
                val_losses.append(loss.item())
                val_l1s.append(l1_loss.item())
                val_ffts.append(fft_loss.item())
                p, s = compute_metrics(sr, hr)
                val_psnrs.append(p)
                val_ssims.append(s)

        # Guard empty train_losses (all batches skipped) and -inf PSNR
        # (skimage returns -inf when MSE=0 or output is constant)
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        avg_val_loss   = np.mean(val_losses)
        avg_val_l1     = np.mean(val_l1s)
        avg_val_fft    = np.mean(val_ffts)
        finite_psnrs   = [v for v in val_psnrs if np.isfinite(v)]
        avg_val_psnr   = np.mean(finite_psnrs) if finite_psnrs else float('nan')
        avg_val_ssim   = np.mean(val_ssims)

        print(f"Epoch {epoch:03d}/{CFG['num_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} (L1: {avg_val_l1:.4f}  FFT: {avg_val_fft:.4f}) | "
              f"PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f}"
              + (f" | NaN batches skipped: {nan_batches}" if nan_batches else ""))

        writer.add_scalar('train/loss', avg_train_loss,      epoch)
        writer.add_scalar('train/l1',   np.mean(train_l1s) if train_l1s else float('nan'), epoch)
        writer.add_scalar('train/fft',  np.mean(train_ffts) if train_ffts else float('nan'), epoch)
        writer.add_scalar('val/loss',   avg_val_loss,        epoch)
        writer.add_scalar('val/l1',     avg_val_l1,          epoch)
        writer.add_scalar('val/fft',    avg_val_fft,         epoch)
        writer.add_scalar('val/psnr',   avg_val_psnr,        epoch)
        writer.add_scalar('val/ssim',   avg_val_ssim,        epoch)
        writer.add_scalar('lr',         scheduler.get_last_lr()[0], epoch)

        if epoch % CFG['save_every'] == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'rrdb_epoch{epoch:03d}.pth')
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'val_psnr'        : avg_val_psnr,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Guard: don't update best if psnr is nan or -inf
        if np.isfinite(avg_val_psnr) and avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_path = os.path.join(CKPT_DIR, 'best_model.pth')
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'val_psnr'        : avg_val_psnr,
            }, best_path)
            print(f"  Best model saved (PSNR: {best_val_psnr:.2f} dB)")

    writer.close()
    print("Training complete.")