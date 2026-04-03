"""
SCUNet — Swin-Conv UNet for SAR Super-Resolution
Faithful implementation of:
  "Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis"
  Zhang et al., 2022  (arXiv:2203.16921)
Adapted for 4× super-resolution with single-channel SAR input.

Architecture overview:
  Encoder: 4 down-stages, each halving spatial resolution
  Bottleneck: SCB block at lowest resolution
  Decoder: 4 up-stages with skip connections from encoder
  Each stage contains `depth[i]` SCB (Swin-Conv) blocks
  SCB block = parallel Swin Transformer branch + Conv branch, summed

Key paper config reproduced here:
  nc=96, depth=[2,2,2,2], window_size=8, num_heads=6
  Conv branch kernel=3, Swin branch window=8

For SR (not denoising): output passes through PixelShuffle 4× upsampler
rather than the paper's direct residual, since we need spatial upscaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_, to_2tuple


# ── Relative position bias helper ─────────────────────────────────────────────

def get_relative_position_index(window_size):
    """Precompute relative position index for window attention."""
    ws = window_size
    coords_h = torch.arange(ws)
    coords_w = torch.arange(ws)
    coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, ws, ws)
    coords_flat = torch.flatten(coords, 1)                                      # (2, ws*ws)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]                     # (2, N, N)
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += ws - 1
    rel[:, :, 1] += ws - 1
    rel[:, :, 0] *= 2 * ws - 1
    return rel.sum(-1)   # (N, N)


# ── Window ops ─────────────────────────────────────────────────────────────────

def window_partition(x, window_size):
    """(B, H, W, C) → (num_windows*B, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, ws, ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size ** 2))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Window Multi-Head Self-Attention ───────────────────────────────────────────

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.register_buffer('relative_position_index',
                             get_relative_position_index(window_size))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # relative position bias
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── Swin Transformer Block ─────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    """One W-MSA or SW-MSA block with pre-norm and MLP."""
    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size      = window_size
        self.shift_size       = shift_size

        # clamp if resolution smaller than window
        if min(input_resolution) <= window_size:
            self.shift_size  = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, self.window_size, num_heads,
                                     qkv_bias=qkv_bias,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

        # precompute shift mask
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            for hi, hs in enumerate((slice(0, -self.window_size),
                                     slice(-self.window_size, -self.shift_size),
                                     slice(-self.shift_size, None))):
                for wi, ws_ in enumerate((slice(0, -self.window_size),
                                          slice(-self.window_size, -self.shift_size),
                                          slice(-self.shift_size, None))):
                    img_mask[:, hs, ws_, :] = hi * 3 + wi
            mw = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
            attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        xw = window_partition(x, self.window_size)
        xw = self.attn(xw.view(-1, self.window_size ** 2, C), mask=self.attn_mask)
        xw = xw.view(-1, self.window_size, self.window_size, C)
        x  = window_reverse(xw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.drop_path(x.view(B, L, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Conv Block ─────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Simple residual conv block: two 3×3 convs with GELU.
    Operates in spatial (B, C, H, W) domain.
    """
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


# ── SCB: Swin-Conv Block ───────────────────────────────────────────────────────

class SCB(nn.Module):
    """
    Swin-Conv Block from SCUNet paper.
    Splits channels into two halves:
      - Half 1: Swin Transformer path (global long-range)
      - Half 2: Conv path             (local short-range)
    Concatenate and project back to full dim.
    This is the core novelty of SCUNet — hybrid local+global processing
    in a single block rather than sequential stages.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        assert dim % 2 == 0, f"SCB dim must be even, got {dim}"
        half = dim // 2

        self.swin = SwinBlock(
            dim=half, input_resolution=input_resolution,
            num_heads=max(1, num_heads // 2),
            window_size=window_size, shift_size=shift_size,
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path,
        )
        self.conv = ConvBlock(half)

        # merge projection after concat
        self.merge = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        half = C // 2
        x_swin = self.swin(x[:, :, :half])          # (B, L, C//2)
        # conv branch needs spatial layout
        H = W = int(L ** 0.5)
        x_conv = x[:, :, half:].transpose(1, 2).view(B, half, H, W)
        x_conv = self.conv(x_conv).view(B, half, L).transpose(1, 2)  # (B, L, C//2)
        return self.merge(torch.cat([x_swin, x_conv], dim=-1))        # (B, L, C)


# ── SCUNet Stage ───────────────────────────────────────────────────────────────

class SCUNetStage(nn.Module):
    """A sequence of `depth` SCB blocks at a fixed resolution."""
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size=8, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SCB(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                # alternate regular / shifted windows across depth
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ── Downsample / Upsample ──────────────────────────────────────────────────────

class Downsample(nn.Module):
    """Strided conv 2× spatial downsampling, doubles channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, x_size):
        """x: (B, H*W, C) → (B, H//2 * W//2, 2C)"""
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)                          # (B, 2C, H//2, W//2)
        B, C2, H2, W2 = x.shape
        return x.view(B, C2, H2 * W2).transpose(1, 2), (H2, W2)


class Upsample(nn.Module):
    """Pixel-shuffle 2× spatial upsampling, halves channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.ps   = nn.PixelShuffle(2)

    def forward(self, x, x_size):
        """x: (B, H*W, C) → (B, 4H*W, C//2)"""
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.ps(self.conv(x))                 # (B, C//2, 2H, 2W)
        B, C2, H2, W2 = x.shape
        return x.view(B, C2, H2 * W2).transpose(1, 2), (H2, W2)


# ── SR Upsampler ───────────────────────────────────────────────────────────────

class SRUpsample(nn.Module):
    """
    PixelShuffle SR upsampler supporting scale=2 or scale=4.
    scale=2: single PixelShuffle stage
    scale=4: two PixelShuffle stages (2x twice)
    """
    def __init__(self, in_ch, out_ch, scale=4):
        super().__init__()
        assert scale in (2, 4), f"SRUpsample only supports scale 2 or 4, got {scale}"
        layers = []
        if scale == 4:
            layers += [
                nn.Conv2d(in_ch, in_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # final 2x stage (used for both scale=2 and scale=4)
        layers += [
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


# ── SCUNet ─────────────────────────────────────────────────────────────────────

class SCUNet(nn.Module):
    """
    SCUNet for SAR Super-Resolution (supports scale=2 or scale=4).

    Paper config reproduced:
      nc=96, depth=[2,2,2,2], window_size=8, num_heads=6
      4-level U-Net encoder-decoder with SCB blocks at each level

    Input  : (B, 1, H, W)              -- single-channel LR SAR, normalised [0,1]
        Output : (B, 1, H*scale, W*scale)
            - default mode: SR prediction in [0,1] via sigmoid
            - residual mode: signed residual prediction without sigmoid

    Channel schedule across U-Net levels (encoder):
      Level 0: nc      (96)   -- full resolution
      Level 1: nc*2   (192)   -- 2× downsampled
      Level 2: nc*4   (384)   -- 4× downsampled
      Level 3: nc*8   (768)   -- 8× downsampled  (bottleneck)

    Decoder mirrors encoder with skip connections.
    """
    def __init__(
        self,
        in_channels  = 1,
        out_channels = 1,
        img_size     = 64,        # LR input spatial size (HR patch / scale)
        nc           = 96,        # base channel count
        depth        = (2, 2, 2, 2),   # SCB blocks per encoder level
        num_heads    = 6,         # attention heads (applied to half-dim Swin branch)
        window_size  = 8,
        mlp_ratio    = 4.,
        drop_rate    = 0.,
        attn_drop    = 0.,
        drop_path_rate = 0.1,
        scale        = 4,
        residual_output = False,
    ):
        super().__init__()
        self.scale       = scale
        self.img_size    = img_size
        self.window_size = window_size
        self.residual_output = residual_output

        # channel sizes at each level
        dims = [nc, nc * 2, nc * 4, nc * 8]

        # resolutions at each encoder level
        res = [(img_size // (2 ** i), img_size // (2 ** i)) for i in range(4)]

        # stochastic depth schedule across all blocks
        total_depth = sum(depth) * 2   # encoder + decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_enc = dpr[:sum(depth)]
        dpr_dec = dpr[sum(depth):]

        # ── Input projection ────────────────────────────────────────────────
        self.conv_first = nn.Conv2d(in_channels, nc, kernel_size=3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_dpr_idx = 0
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            d = depth[i]
            stage = SCUNetStage(
                dim=dims[i], input_resolution=res[i],
                depth=d, num_heads=num_heads,
                window_size=min(window_size, res[i][0]),
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop,
                drop_path=dpr_enc[enc_dpr_idx:enc_dpr_idx + d],
            )
            self.enc_stages.append(stage)
            enc_dpr_idx += d

            if i < 3:
                self.downsamples.append(Downsample(dims[i], dims[i + 1]))

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_dpr_idx = 0
        self.dec_stages = nn.ModuleList()
        self.upsamples  = nn.ModuleList()
        self.skip_projs = nn.ModuleList()   # project concatenated skip+up → dim

        for i in range(3, 0, -1):   # levels 3→2→1→0 (3 up-steps)
            self.upsamples.append(Upsample(dims[i], dims[i - 1]))
            # after upsample we concat skip → dims[i-1]*2 → project back to dims[i-1]
            self.skip_projs.append(nn.Linear(dims[i - 1] * 2, dims[i - 1]))
            d = depth[i - 1]
            stage = SCUNetStage(
                dim=dims[i - 1], input_resolution=res[i - 1],
                depth=d, num_heads=num_heads,
                window_size=min(window_size, res[i - 1][0]),
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop,
                drop_path=dpr_dec[dec_dpr_idx:dec_dpr_idx + d],
            )
            self.dec_stages.append(stage)
            dec_dpr_idx += d

        # ── Output norm + conv ───────────────────────────────────────────────
        self.norm_out = nn.LayerNorm(nc)
        self.conv_out = nn.Conv2d(nc, nc, kernel_size=3, padding=1)

        # ── SR upsampler (scale=2 or scale=4) ───────────────────────────────
        self.sr_upsample = SRUpsample(nc, out_channels, scale=scale)

        # init all submodules first
        self.apply(self._init_weights)

        # Reinit sr_upsample convs after apply() with consistent leaky_relu kaiming.
        # apply() uses the same kaiming but with relu assumption which overestimates
        # gain -- reinit with correct a=0.2 for LeakyReLU(0.2) used in the upsampler.
        for m in self.sr_upsample.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # In residual mode start from exact bicubic behavior: zero residual.
        if self.residual_output and isinstance(self.sr_upsample.body[-1], nn.Conv2d):
            nn.init.zeros_(self.sr_upsample.body[-1].weight)
            if self.sr_upsample.body[-1].bias is not None:
                nn.init.zeros_(self.sr_upsample.body[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # fan_in: preserves forward-pass variance; consistent with sr_upsample reinit
            nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                    nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def check_image_size(self, x):
        """Pad so H,W are multiples of window_size * 8 (for 3 downsamples)."""
        _, _, h, w = x.shape
        stride = self.window_size * 8
        pad_h  = (stride - h % stride) % stride
        pad_w  = (stride - w % stride) % stride
        return F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

    def forward(self, x):
        H_orig, W_orig = x.shape[2], x.shape[3]
        x = self.check_image_size(x)
        _, _, H, W = x.shape

        # shallow feature extraction
        feat = self.conv_first(x)                          # (B, nc, H, W)

        # flatten to sequence for transformer stages
        B, C, Hf, Wf = feat.shape
        feat_seq = feat.view(B, C, Hf * Wf).transpose(1, 2)   # (B, H*W, nc)
        cur_size = (Hf, Wf)

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_skips = []
        for i in range(4):
            feat_seq = self.enc_stages[i](feat_seq)
            if i < 3:
                enc_skips.append((feat_seq, cur_size))
                feat_seq, cur_size = self.downsamples[i](feat_seq, cur_size)

        # ── Decoder ──────────────────────────────────────────────────────────
        for i in range(3):
            feat_seq, cur_size = self.upsamples[i](feat_seq, cur_size)
            skip, skip_size    = enc_skips[2 - i]
            # concatenate skip connection and project
            feat_seq = self.skip_projs[i](torch.cat([feat_seq, skip], dim=-1))
            feat_seq = self.dec_stages[i](feat_seq)

        # ── Output ───────────────────────────────────────────────────────────
        feat_seq = self.norm_out(feat_seq)
        feat_out = feat_seq.transpose(1, 2).view(B, -1, Hf, Wf)  # (B, nc, H, W)
        feat_out = self.conv_out(feat_out)

        # 2x SR upsampling; apply sigmoid only for direct image prediction.
        out = self.sr_upsample(feat_out).float()
        if not self.residual_output:
            out = torch.sigmoid(out)

        # crop to exact SR output size
        return out[:, :, :H_orig * self.scale, :W_orig * self.scale]


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SCUNet(
        in_channels=1, out_channels=1,
        img_size=64,           # LR size = 128 / 2
        nc=96,
        depth=(2, 2, 2, 2),
        num_heads=6,
        window_size=8,
        mlp_ratio=4.,
        scale=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_lr = torch.rand(2, 1, 64, 64).to(device)
    with torch.no_grad():
        dummy_hr = model(dummy_lr)

    print(f"Input  shape : {dummy_lr.shape}")
    print(f"Output shape : {dummy_hr.shape}  (expect (2, 1, 128, 128))")
    print(f"Output range : [{dummy_hr.min():.3f}, {dummy_hr.max():.3f}]  (expect [0,1])")
    print(f"Output mean  : {dummy_hr.mean():.3f}  (expect ~0.50-0.75)")
    print(f"Output std   : {dummy_hr.std():.3f}   (expect >0.05)")
    assert dummy_hr.shape == (2, 1, 128, 128), f"Shape mismatch: {dummy_hr.shape}"
    assert dummy_hr.std() > 0.05, f"Output collapsed: std={dummy_hr.std():.4f}"
    print("SCUNet smoke test passed.")