import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv2d
from Losses.utils import FourierFeatureMap

class UFNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        return F.gelu(y)


class UFNO2dDualTask(nn.Module):
    """U-shaped FNO backbone with dual heads."""

    def __init__(
        self,
        modes1: int = 40,
        modes2: int = 48,
        width: int = 96,
        in_channels: int = 3,
        out_channels_img: int = 3,
        out_channels_scalar: int = 1,
        n_layers: int = 4,
        fourier_feats: dict | None = None,
        depth: int | None = None,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels_img = out_channels_img
        self.out_channels_scalar = out_channels_scalar
        if depth is None:
            depth = max(2, n_layers // 2) if n_layers > 0 else 2
        self.depth = max(2, depth)

    # Fourier Feature Network config (position encoding)
        self.ff_cfg = (fourier_feats or {})
        self.ff_enable = bool(self.ff_cfg.get("enable", False))
        self.ff_num_bands = int(self.ff_cfg.get("num_bands", 16))
        self.ff_std = float(self.ff_cfg.get("std", 10.0))
        self.ff_trainable = bool(self.ff_cfg.get("trainable", False))
        self.ffm = None  # will be built on first forward when H,W known
        if self.ff_enable:
            # Project 2K features to `width` channels then add residually
            self.ff_conv = nn.Conv2d(2 * self.ff_num_bands, width, kernel_size=1)


        self.fc0 = nn.Linear(in_channels + 2, width)
        def _expand_modes(m, depth, name):
            if isinstance(m, int):
                return [m] * depth
            if isinstance(m, (list, tuple)):
                assert len(m) == depth, f"{name} must have length = depth ({depth})."
                return list(m)
            raise TypeError(f"{name} must be int or list/tuple of length {depth}.")

        m1_list = _expand_modes(self.modes1, self.depth, "modes1")
        m2_list = _expand_modes(self.modes2, self.depth, "modes2")

        # 编码用前 depth-1 个；bottleneck 用最后一个；解码端对称复用
        enc_m1, enc_m2 = m1_list[:-1], m2_list[:-1]
        bot_m1, bot_m2 = m1_list[-1],  m2_list[-1]
        dec_m1, dec_m2 = enc_m1[::-1], enc_m2[::-1]

        self.encoder_blocks = nn.ModuleList([
            UFNOBlock(width, modes1=enc_m1[i], modes2=enc_m2[i])
            for i in range(self.depth - 1)
        ])
        self.bottleneck = UFNOBlock(width, modes1=bot_m1, modes2=bot_m2)
        self.decoder_blocks = nn.ModuleList([
            UFNOBlock(width, modes1=dec_m1[i], modes2=dec_m2[i])
            for i in range(self.depth - 1)
        ])

        self.merge_layers = nn.ModuleList(
            [nn.Conv2d(width * 2, width, kernel_size=1) for _ in range(self.depth - 1)]
        )
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.fc1_img = nn.Linear(width, 128)
        self.fc2_img = nn.Linear(128, out_channels_img)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1_scalar = nn.Linear(width, 64)
        self.fc2_scalar = nn.Linear(64, 32)
        self.fc3_scalar = nn.Linear(32, out_channels_scalar)

    

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=torch.float32)
        gridx = gridx.view(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=torch.float32)
        gridy = gridy.view(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x: torch.Tensor):
        bsz, _, h, w = x.shape
        grid = self.get_grid((bsz, h, w), x.device)
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if self.ff_enable:
            # Lazy build feature map for current H,W
            if (self.ffm is None) or (self.ffm.H != h) or (self.ffm.W != w):
                self.ffm = FourierFeatureMap(h, w, num_bands=self.ff_num_bands, std=self.ff_std, trainable=self.ff_trainable)
            feats = self.ffm(bsz, device=x.device, dtype=x.dtype)  # (B,2K,H,W)
            x = x + self.ff_conv(feats)
        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.downsample(x)

        x = self.bottleneck(x)

        for idx, skip in enumerate(reversed(skips)):
            x = self.upsample(x)
            # pad if needed (in case of odd dimensions)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.merge_layers[idx](x)
            x = self.decoder_blocks[idx](x)

        x_img = x.permute(0, 2, 3, 1)
        x_img = F.gelu(self.fc1_img(x_img))
        x_img = self.fc2_img(x_img)
        x_img = x_img.permute(0, 3, 1, 2)

        x_scalar = self.global_pool(x).view(bsz, -1)
        x_scalar = F.gelu(self.fc1_scalar(x_scalar))
        x_scalar = F.gelu(self.fc2_scalar(x_scalar))
        x_scalar = self.fc3_scalar(x_scalar)

        return x_img, x_scalar

    @property
    def model_size(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_layers = len(
            [m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear, SpectralConv2d))]
        )
        return n_params, n_layers