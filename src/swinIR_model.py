import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


# ── Helpers ────────────────────────────────────────────────────────────────────

def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    x : (B, H, W, C)
    Returns: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partitioning.
    windows : (num_windows*B, window_size, window_size, C)
    Returns : (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Window Multi-head Self-Attention ───────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    Supports both regular and shifted windows.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = to_2tuple(window_size)   # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # precompute relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))   # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                                    # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # add relative position bias
        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── Swin Transformer Block ─────────────────────────────────────────────────────

class SwinTransformerBlock(nn.Module):
    """
    One Swin Transformer block.
    shift_size=0  → regular window attention (W-MSA)
    shift_size>0  → shifted window attention (SW-MSA)
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio

        # clamp shift/window to resolution
        if min(self.input_resolution) <= self.window_size:
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size=self.window_size,
                                     num_heads=num_heads, qkv_bias=qkv_bias,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # precompute attention mask for shifted windows
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
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

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Residual Swin Transformer Group (RSTB) ────────────────────────────────────

class RSTB(nn.Module):
    """
    Residual Swin Transformer Block group.
    depth STBs + one 3×3 conv for local feature enhancement, then residual add.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, x_size):
        H, W = x_size
        residual = x
        for blk in self.blocks:
            x = blk(x)
        # reshape → conv → reshape for local feature enhancement
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        return x + residual


# ── Patch Embed / Unembed ──────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """3×3 conv projection: (B, C_in, H, W) → (B, H*W, embed_dim)"""
    def __init__(self, in_channels=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                          # (B, E, H, W)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, H*W, E)
        return self.norm(x), H, W


class PatchUnembed(nn.Module):
    """Reshape (B, H*W, embed_dim) → (B, embed_dim, H, W)"""
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        H, W = x_size
        return x.transpose(1, 2).view(-1, self.embed_dim, H, W)


# ── Upsampler ──────────────────────────────────────────────────────────────────

class UpsamplePixelShuffle(nn.Sequential):
    """Two stages of PixelShuffle to achieve 4× upsampling."""
    def __init__(self, scale, num_feat):
        assert scale == 4, "Only scale=4 supported (2× PixelShuffle twice)"
        super().__init__(
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )


# ── SwinIR ─────────────────────────────────────────────────────────────────────

class SwinIR(nn.Module):
    """
    SwinIR for SAR super-resolution.
    Input  : (B, 1, H, W)          -- single-channel LR SAR patch
    Output : (B, 1, H*scale, W*scale)  -- HR prediction in [0, 1]

    Architecture:
      PatchEmbed → RSTB × num_groups → LayerNorm → conv → PatchUnembed
      → UpsamplePixelShuffle → conv_last → sigmoid

    Default params match SwinIR-lightweight from the paper, tuned for
    single-channel SAR with patch_size=32 (LR) → 128 (HR).
    """
    def __init__(
        self,
        in_channels    = 1,
        out_channels   = 1,
        img_size       = 64,       # LR spatial size = HR patch_size / scale (64 for 256HR/4x)
        embed_dim      = 60,       # feature channels throughout transformer
        depths         = (6, 6, 6, 6),   # STBs per RSTB group
        num_heads      = (6, 6, 6, 6),   # attention heads per group
        window_size    = 8,        # attention window (must divide LR patch size)
        mlp_ratio      = 2.0,      # MLP hidden ratio inside STB
        scale          = 4,
        drop_rate      = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.1,
    ):
        super().__init__()
        self.scale       = scale
        self.embed_dim   = embed_dim
        self.window_size = window_size
        self.img_size    = img_size   # LR spatial size -- must match actual input

        # input_resolution drives attention mask shape -- must equal LR patch size
        self.input_resolution = (img_size, img_size)

        # ── Shallow feature extraction ──────────────────────────────────────
        self.conv_first  = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(in_channels=embed_dim, embed_dim=embed_dim)
        self.patch_unembed = PatchUnembed(embed_dim=embed_dim)

        # ── Deep feature extraction (stacked RSTBs) ─────────────────────────
        num_groups = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(num_groups):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
            )
            self.layers.append(layer)

        self.norm      = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # ── Upsampling + output ─────────────────────────────────────────────
        self.upsample  = UpsamplePixelShuffle(scale=scale, num_feat=embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

        # Init: Kaiming weights + 0.5 bias → sigmoid gives mean ~0.62 at init
        nn.init.kaiming_normal_(self.conv_last.weight, mode='fan_out',
                                nonlinearity='leaky_relu', a=0.2)
        nn.init.constant_(self.conv_last.bias, 0.5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def check_image_size(self, x):
        """Pad input so H and W are multiples of window_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # shallow features
        feat = self.conv_first(x)                      # (B, E, H, W)

        # patch embed for transformer
        feat_seq, h, w = self.patch_embed(feat)        # (B, H*W, E)
        x_size = (h, w)

        # deep feature extraction
        for layer in self.layers:
            feat_seq = layer(feat_seq, x_size)

        feat_seq = self.norm(feat_seq)
        feat_deep = self.patch_unembed(feat_seq, x_size)  # (B, E, H, W)

        # long skip + conv
        feat = feat + self.conv_after_body(feat_deep)

        # upsample + output; .float() before sigmoid fixes AMP float16 saturation
        out = self.conv_last(self.upsample(feat)).float()

        # crop back to exact output size in case of padding
        out = out[:, :, :H * self.scale, :W * self.scale]

        return torch.sigmoid(out)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SwinIR(
        in_channels=1, out_channels=1,
        img_size=64,   # LR size = 256 / 4
        embed_dim=60, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
        window_size=8, mlp_ratio=2.0, scale=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_lr = torch.rand(2, 1, 64, 64).to(device)   # LR = 256/4 = 64
    with torch.no_grad():
        dummy_hr = model(dummy_lr)

    print(f"Input  shape : {dummy_lr.shape}")
    print(f"Output shape : {dummy_hr.shape}")
    print(f"Output range : [{dummy_hr.min():.3f}, {dummy_hr.max():.3f}]")
    print(f"Output mean  : {dummy_hr.mean():.3f}  (expect ~0.62-0.72)")
    print(f"Output std   : {dummy_hr.std():.3f}   (should be >0.05)")
    print("SwinIR smoke test passed.")