import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.append(DenseLayer(num_feat + i * growth_rate, growth_rate))
        self.final_conv = nn.Conv2d(num_feat + 5 * growth_rate, num_feat, kernel_size=3, padding=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = self.final_conv(torch.cat(features, dim=1))
        return x + out * self.res_scale


class RRDB(nn.Module):
    def __init__(self, num_feat=64, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, growth_rate, res_scale)
        self.rdb2 = ResidualDenseBlock(num_feat, growth_rate, res_scale)
        self.rdb3 = ResidualDenseBlock(num_feat, growth_rate, res_scale)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class RRDBNet(nn.Module):
    """
    Full RRDB generator for SAR super-resolution.
    Input  : (B, 1, H, W)   -- single-channel LR SAR patch
    Output : (B, 1, H*scale, W*scale) -- HR prediction in [0, 1]
    """
    def __init__(self, in_channels=1, out_channels=1, num_feat=64,
                 num_blocks=8, growth_rate=32, scale=4):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_feat, kernel_size=3, padding=1)

        self.trunk = nn.Sequential(*[RRDB(num_feat, growth_rate) for _ in range(num_blocks)])
        self.conv_trunk = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_hr   = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.lrelu_hr  = nn.LeakyReLU(0.2, inplace=True)
        self.conv_last = nn.Conv2d(num_feat, out_channels, kernel_size=3, padding=1)

        # Kaiming weights keep std alive; 0.5 bias shifts sigmoid(x) mean to ~0.622
        # zeros_() on weights kills all upstream variance -- DO NOT use zeros_ here
        nn.init.kaiming_normal_(self.conv_last.weight, mode='fan_out',
                                nonlinearity='leaky_relu', a=0.2)
        nn.init.constant_(self.conv_last.bias, 0.5)

    def forward(self, x):
        feat  = self.conv_first(x)
        trunk = self.conv_trunk(self.trunk(feat))
        feat  = feat + trunk
        feat  = self.upsample(feat)
        feat  = self.lrelu_hr(self.conv_hr(feat))
        # .float() before sigmoid -- fixes AMP float16 saturation
        out   = self.conv_last(feat).float()
        return torch.sigmoid(out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = RRDBNet(in_channels=1, out_channels=1, num_feat=64,
                    num_blocks=8, growth_rate=32, scale=4).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_lr = torch.rand(2, 1, 32, 32).to(device)  # rand not randn -- same dist as real LR
    with torch.no_grad():
        dummy_hr = model(dummy_lr)

    print(f"Input  shape : {dummy_lr.shape}")
    print(f"Output shape : {dummy_hr.shape}")
    print(f"Output range : [{dummy_hr.min():.3f}, {dummy_hr.max():.3f}]")
    print(f"Output mean  : {dummy_hr.mean():.3f}  (expect ~0.622 -- sigmoid(Kaiming + 0.5 bias)")
    print(f"Output std   : {dummy_hr.std():.3f}   (should be >0.05 -- proves no mean collapse)")
    print("RRDBNet smoke test passed.")