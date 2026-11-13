"""
=============================================================
STEP 3 — Fine-tune Reduced RRDB (ESRGAN generator) on Umbra SAR Data
=============================================================
Optimised for:
  HP Victus 15-fa2xxx
  GPU  : NVIDIA RTX 5050 Laptop GPU — Blackwell sm_120
  VRAM : 8.5 GB
  PyTorch : 2.12 + CUDA 12.8

Differences from SwinIR version:
  - Model    : Reduced RRDB (8 blocks, 32 channels) ~3M params vs ~11M
  - VRAM     : ~2 GB at bs=16 — much more headroom than SwinIR
  - Batch    : 16 direct (no grad accum needed, kept for compatibility)
  - Pretrain : ESRGAN x4 weights from BasicSR release (optional)
               conv_first: 3ch -> 1ch by averaging, same as before
  - AMP      : Enabled (RRDB is CNN-only, no attention instability)
  - Scheduler: CosineAnnealingLR (better than StepLR for RRDB)
  - Resume   : same logic as SwinIR version
=============================================================
"""

import os, sys, requests, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────
TRAIN_HR_DIR   = "data/train/hr"
TRAIN_LR_DIR   = "data/train/lr"
VAL_HR_DIR     = "data/val/hr"
VAL_LR_DIR     = "data/val/lr"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE       = "training_log_rrdb_v2.txt"

SCALE          = 4
BATCH_SIZE     = 8           # reduced for larger model — still efficient
GRAD_ACCUM     = 1           # no accumulation needed; kept for compatibility
NUM_EPOCHS     = 30
LR_INITIAL     = 1e-4
LR_MIN         = 1e-6
PATCH_SIZE_LR  = 32          # same as before — matches your prepared patches
PATCH_SIZE_HR  = 128
MAX_TRAIN      = 100_000     # can raise to 200k+ — RRDB trains faster per iter
VAL_SAMPLES    = 2_000

LAMBDA_L1      = 1.0
LAMBDA_PERC    = 0.0         # set to 0.1 for Phase 2 perceptual fine-tuning

# ── RESUME CONFIG ──────────────────────────────────────────
RESUME_TRAINING   = False   # fresh start with new capacity
RESUME_CHECKPOINT = "checkpoints/rrdb_sar_last_v2.pth"
# ──────────────────────────────────────────────────────────

# ESRGAN pretrained weights from BasicSR (trained on DIV2K/Flickr2K, x4, RGB)
# conv_first (3ch -> 1ch) is handled by averaging — same approach as SwinIR version
PRETRAINED_URL  = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
PRETRAINED_PATH = "checkpoints/realesrgan_x4plus.pth"
BEST_CKPT_PATH  = "checkpoints/rrdb_sar_best_v2.pth"
LAST_CKPT_PATH  = "checkpoints/rrdb_sar_last_v2.pth"


# ══════════════════════════════════════════════════════════
#  GPU CHECK  (unchanged from SwinIR version)
# ══════════════════════════════════════════════════════════

def check_gpu():
    print("=" * 62)
    print("  GPU / CUDA Check")
    print("=" * 62)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM            : {props.total_memory/1e9:.1f} GB")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  Compute cap     : sm_{props.major}{props.minor}")
        if props.major >= 12:
            print("  Blackwell GPU detected — requires PyTorch 2.7+ (you have it!)")
        print("  GPU training enabled!")
    else:
        print("\n  WARNING: CUDA not available!")
        ans = input("  Continue on CPU anyway? (y/N): ").strip().lower()
        if ans != 'y':
            sys.exit(0)
    print("=" * 62 + "\n")


# ══════════════════════════════════════════════════════════
#  RRDB MODEL
#  Reduced config: num_block=8, num_feat=32  (~3M params, ~2GB VRAM at bs=16)
#  Full ESRGAN config for reference: num_block=23, num_feat=64 (~16M params)
# ══════════════════════════════════════════════════════════

class ResidualDenseBlock(nn.Module):
    """5 conv layers with dense connections and 0.2x residual scaling."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,             num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch,   num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch, num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialise with small weights — important for stable early training
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(layer.weight)
            layer.weight.data *= 0.1

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x  # residual scaling


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block = 3× ResidualDenseBlock."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x  # residual scaling


class RRDBNet(nn.Module):
    """
    ESRGAN generator with reduced capacity for 8GB VRAM overnight training.

    Default here: num_block=8, num_feat=32, num_grow_ch=16  (~3M params)
    To match full ESRGAN: num_block=23, num_feat=64, num_grow_ch=32
    """
    def __init__(self, in_channels=1, out_channels=1,
                 num_feat=64, num_block=12, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch)
                                    for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling
        self.conv_up1  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_out = self.conv_body(self.body(feat))
        feat = feat + body_out   # global residual
        # nearest-neighbour upsample x2 twice = x4 total
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(
            F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ══════════════════════════════════════════════════════════
#  Perceptual Loss  (unchanged from SwinIR version)
# ══════════════════════════════════════════════════════════

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg      = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg = nn.Sequential(*list(vgg.features)[:18]).to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        return self.loss(self.vgg(sr), self.vgg(hr))


# ══════════════════════════════════════════════════════════
#  Dataset  (unchanged from SwinIR version)
# ══════════════════════════════════════════════════════════

class SARPatchDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, augment=False, max_samples=None):
        hr_files = sorted(f for f in os.listdir(hr_dir) if f.endswith(".png"))
        lr_files = sorted(f for f in os.listdir(lr_dir) if f.endswith(".png"))
        if max_samples:
            random.seed(42)
            indices  = random.sample(range(len(hr_files)),
                                     min(max_samples, len(hr_files)))
            hr_files = [hr_files[i] for i in sorted(indices)]
            lr_files = [lr_files[i] for i in sorted(indices)]
        valid_hr, valid_lr, skipped = [], [], 0
        for hf, lf in zip(hr_files, lr_files):
            hp = os.path.join(hr_dir, hf)
            lp = os.path.join(lr_dir, lf)
            if cv2.imread(hp, cv2.IMREAD_GRAYSCALE) is None or \
               cv2.imread(lp, cv2.IMREAD_GRAYSCALE) is None:
                skipped += 1; continue
            valid_hr.append(hp); valid_lr.append(lp)
        if skipped > 0:
            print(f"  Skipped {skipped} corrupt/missing image pairs.")
        self.hr_paths = valid_hr
        self.lr_paths = valid_lr
        self.augment  = augment

    def __len__(self): return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = cv2.imread(self.hr_paths[idx], cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(self.lr_paths[idx], cv2.IMREAD_GRAYSCALE)
        if hr is None or lr is None:
            lr = np.zeros((PATCH_SIZE_LR, PATCH_SIZE_LR), dtype=np.float32)
            hr = np.zeros((PATCH_SIZE_HR, PATCH_SIZE_HR), dtype=np.float32)
        else:
            hr = hr.astype(np.float32) / 255.
            lr = lr.astype(np.float32) / 255.
            # guard: replace any nan/inf pixels with 0 (corrupt PNG content)
            if not np.isfinite(hr).all(): hr = np.nan_to_num(hr, nan=0.0, posinf=1.0, neginf=0.0)
            if not np.isfinite(lr).all(): lr = np.nan_to_num(lr, nan=0.0, posinf=1.0, neginf=0.0)
            if hr.shape[0] != PATCH_SIZE_HR:
                hr = cv2.resize(hr, (PATCH_SIZE_HR, PATCH_SIZE_HR), cv2.INTER_CUBIC)
            if lr.shape[0] != PATCH_SIZE_LR:
                lr = cv2.resize(lr, (PATCH_SIZE_LR, PATCH_SIZE_LR), cv2.INTER_CUBIC)
        if self.augment:
            if np.random.rand() > .5: hr, lr = np.fliplr(hr).copy(), np.fliplr(lr).copy()
            if np.random.rand() > .5: hr, lr = np.flipud(hr).copy(), np.flipud(lr).copy()
        return (torch.from_numpy(lr).unsqueeze(0),
                torch.from_numpy(hr).unsqueeze(0))


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def compute_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    return 100.0 if mse == 0 else (
        20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def download_pretrained():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(PRETRAINED_PATH):
        print(f"  Pretrained weights: {PRETRAINED_PATH}  ok"); return
    print("  Downloading pretrained RealESRGAN weights (~65 MB)...")
    r = requests.get(PRETRAINED_URL, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(PRETRAINED_PATH, 'wb') as f, \
         tqdm(total=total, unit='B', unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk); bar.update(len(chunk))


def load_pretrained(model, path, device):
    """
    Load RealESRGAN pretrained weights into our reduced RRDBNet.

    Key differences handled:
      - RealESRGAN uses num_feat=64, num_block=23 — our model is smaller,
        so only conv_first + any matching layers are loaded; the rest
        initialise from scratch (still very useful warm-start).
      - conv_first: 3ch pretrained -> 1ch SAR by averaging (same as SwinIR).
      - Shape mismatches (64 vs 32 channels) are silently skipped.

    Tip: if you want a closer warm-start, temporarily set
    num_feat=64, num_block=23, load weights fully, then distil/prune.
    For overnight training the reduced random-init body is fine.
    """
    if not os.path.exists(path):
        print("  No pretrained weights — training from scratch.")
        return model

    state = torch.load(path, map_location=device, weights_only=True)
    # RealESRGAN checkpoint wraps weights under 'params_ema' or 'params'
    for key in ('params_ema', 'params', 'state_dict', 'model'):
        if key in state:
            state = state[key]; break

    ms     = model.state_dict()
    loaded = 0
    skipped_shape = 0

    for k, v in state.items():
        if k not in ms:
            continue
        target = ms[k]

        # exact match — load directly
        if target.shape == v.shape:
            ms[k] = v
            loaded += 1
            continue

        # only attempt slicing for conv weights (4-D) and bias/linear (1-D or 2-D)
        if v.dim() != target.dim():
            skipped_shape += 1
            continue

        try:
            w = v
            # conv weight: [out_ch, in_ch, kH, kW]
            if w.dim() == 4:
                # in_ch: 3ch RGB -> 1ch SAR — average
                if w.shape[1] == 3 and target.shape[1] == 1:
                    w = w.mean(dim=1, keepdim=True)
                # in_ch: any other size mismatch — slice
                elif w.shape[1] > target.shape[1]:
                    w = w[:, :target.shape[1]]
                # out_ch: pretrained larger — slice first N filters
                if w.shape[0] > target.shape[0]:
                    w = w[:target.shape[0]]

            # bias / norm weight / norm bias: 1-D, just slice
            elif w.dim() == 1:
                if w.shape[0] > target.shape[0]:
                    w = w[:target.shape[0]]

            if w.shape == target.shape:
                ms[k] = w
                loaded += 1
            else:
                skipped_shape += 1
        except Exception:
            skipped_shape += 1

    model.load_state_dict(ms, strict=False)
    print(f"  Loaded {loaded} pretrained layers out of {len(ms)} total.")
    if skipped_shape > 0:
        print(f"  Skipped {skipped_shape} layers with shape mismatch.")
    return model


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def train():
    check_gpu()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False   # disabled: Blackwell sm_120 + sliced pretrained weights causes nan loss with AMP

    print("=" * 62)
    print("  RRDBNet v2 (medium ESRGAN) Fine-tuning — RTX 5050 8GB config")
    print("=" * 62)
    print(f"  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM            : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Batch size      : {BATCH_SIZE}  (grad accum x{GRAD_ACCUM} -> effective {BATCH_SIZE*GRAD_ACCUM})")
    print(f"  Patch size LR/HR: {PATCH_SIZE_LR} / {PATCH_SIZE_HR}")
    print(f"  Train patches   : {MAX_TRAIN:,} (capped)")
    print(f"  Val patches     : {VAL_SAMPLES:,}")
    print(f"  Mixed precision : {use_amp}")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Resume training : {RESUME_TRAINING}")
    print("=" * 62)

    for d in [TRAIN_HR_DIR, TRAIN_LR_DIR, VAL_HR_DIR, VAL_LR_DIR]:
        if not os.path.isdir(d):
            print(f"\n  ERROR: {d} not found — run step2 first.")
            sys.exit(1)

    train_ds = SARPatchDataset(TRAIN_HR_DIR, TRAIN_LR_DIR,
                                augment=True, max_samples=MAX_TRAIN)
    val_ds   = SARPatchDataset(VAL_HR_DIR, VAL_LR_DIR,
                                augment=False, max_samples=VAL_SAMPLES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=8,    # match train batch size
                              shuffle=False, num_workers=4,
                              pin_memory=True, persistent_workers=True)

    print(f"\n  Train: {len(train_ds):,} patches  |  Val: {len(val_ds):,} patches")

    model = RRDBNet(
        in_channels=1,
        out_channels=1,
        num_feat=64,        # full ESRGAN width — ~6M params, still fits 8GB at bs=8
        num_block=12,       # middle ground: more capacity, trains in ~8h overnight
        num_grow_ch=32,     # full grow channels
        scale=4
    ).to(device)

    download_pretrained()
    model = load_pretrained(model, PRETRAINED_PATH, device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M\n")

    crit_l1   = nn.L1Loss()
    crit_perc = VGGPerceptualLoss(device)

    optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL, betas=(0.9, 0.99))
    # CosineAnnealingLR: smoothly decays to LR_MIN over all epochs
    # Better than StepLR for RRDB — avoids sudden LR drops stalling training
    total_iters = NUM_EPOCHS * len(train_loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=LR_MIN)
    scaler = GradScaler("cuda", enabled=use_amp)

    # ══════════════════════════════════════════════════════
    #  RESUME LOGIC  (unchanged from SwinIR version)
    # ══════════════════════════════════════════════════════
    start_epoch = 1
    best_psnr   = 0.0
    logs        = []

    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        print(f"  Resuming from: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('psnr', 0.0)
        # fast-forward cosine scheduler to the correct position
        iters_done = (ckpt['epoch']) * len(train_loader)
        for _ in range(iters_done):
            scheduler.step()
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = [l for l in f.read().splitlines() if l.strip()]
        print(f"  Resumed from epoch {ckpt['epoch']} | Best PSNR: {best_psnr:.2f} dB")
        if start_epoch > NUM_EPOCHS:
            print(f"  All {NUM_EPOCHS} epochs complete. Increase NUM_EPOCHS to train more.")
            sys.exit(0)
        print(f"  Continuing from epoch {start_epoch} to {NUM_EPOCHS}\n")
    else:
        if RESUME_TRAINING:
            print("  No checkpoint found — starting fresh.\n")
        else:
            print("  RESUME_TRAINING = False — starting fresh.\n")

    current_epoch = start_epoch
    avg_psnr      = 0.0

    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            current_epoch = epoch
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader,
                        desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}", unit="batch")

            for step, (lr_imgs, hr_imgs) in enumerate(pbar, 1):
                lr_imgs = lr_imgs.to(device, non_blocking=True)
                hr_imgs = hr_imgs.to(device, non_blocking=True)

                with autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                    sr   = model(lr_imgs)
                    loss = (LAMBDA_L1   * crit_l1(sr, hr_imgs) +
                            LAMBDA_PERC * crit_perc(sr, hr_imgs)) / GRAD_ACCUM

                scaler.scale(loss).backward()

                if step % GRAD_ACCUM == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # cosine scheduler steps every iteration, not every epoch
                scheduler.step()

                epoch_loss += loss.item() * GRAD_ACCUM
                pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            avg_loss = epoch_loss / len(train_loader)

            model.eval()
            psnr_vals = []
            with torch.no_grad():
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs = lr_imgs.to(device, non_blocking=True)
                    hr_imgs = hr_imgs.to(device, non_blocking=True)
                    with autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                        sr = torch.clamp(model(lr_imgs), 0., 1.)
                    psnr_vals.append(compute_psnr(sr, hr_imgs))

            avg_psnr = float(np.mean(psnr_vals))
            tag      = ""

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'psnr': best_psnr},
                           BEST_CKPT_PATH)
                tag = "  <- best"

            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'psnr': avg_psnr},
                       LAST_CKPT_PATH)

            line = (f"Epoch {epoch:02d} | Loss {avg_loss:.4f} | "
                    f"Val PSNR {avg_psnr:.2f} dB | Best {best_psnr:.2f} dB{tag}")
            print(f"  {line}")
            logs.append(line)
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(logs))

    except KeyboardInterrupt:
        print("\n\n  Training interrupted.")
        print(f"  Last completed epoch : {current_epoch - 1}")
        print(f"  Checkpoint saved at  : {LAST_CKPT_PATH}")
        print(f"  Best PSNR so far     : {best_psnr:.2f} dB")
        print(f"  To resume: set RESUME_TRAINING = True and run again.")
        sys.exit(0)

    print("\n" + "=" * 62)
    print(f"  Training complete!")
    print(f"  Best val PSNR : {best_psnr:.2f} dB")
    print(f"  Best model    : {os.path.abspath(BEST_CKPT_PATH)}")
    print(f"  Log           : {os.path.abspath(LOG_FILE)}")
    print("=" * 62)
    print("\n  Next: python step4_inference.py")


if __name__ == "__main__":
    train()