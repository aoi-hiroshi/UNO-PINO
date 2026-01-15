import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)

        self.weight_r1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_i1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_r2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weight_i2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def compl_mul2d(self, a, w):
        # a: (B, Cin, Hm, Wm) complex ; w: (Cin, Cout, Hm, Wm) complex
        # -> (B, Cout, Hm, Wm) complex
        return torch.einsum("bixy,ioxy->boxy", a, w)

    def forward(self, x):
        B, C, H, W = x.shape


        if torch.is_autocast_enabled():
            x = x.float()

        x_ft = torch.fft.rfft2(x)  # complex64
        dtype_c = x_ft.dtype
        dtype_r = torch.float32    # 实权重保持 float32

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=dtype_c, device=x.device)

        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        # 低频角
        wr1 = self.weight_r1[:, :, :m1, :m2].to(dtype=dtype_r, device=x.device)
        wi1 = self.weight_i1[:, :, :m1, :m2].to(dtype=dtype_r, device=x.device)
        w1 = torch.complex(wr1, wi1).to(dtype=dtype_c)

        # 高频“负频角”
        wr2 = self.weight_r2[:, :, :m1, :m2].to(dtype=dtype_r, device=x.device)
        wi2 = self.weight_i2[:, :, :m1, :m2].to(dtype=dtype_r, device=x.device)
        w2 = torch.complex(wr2, wi2).to(dtype=dtype_c)

        out_ft[:, :, :m1, :m2]  = self.compl_mul2d(x_ft[:, :, :m1, :m2],  w1)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], w2)

        y = torch.fft.irfft2(out_ft, s=(H, W))  # float32
        return y

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=64, in_channels=1, out_channels=3):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection
        self.fc0 = nn.Linear(in_channels + 2, self.width)  # +2 for grid coordinates

        # Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Local convolutions
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x):
        # x: (batch, channels, height, width)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # Add grid coordinates
        grid = self.get_grid([batchsize, size_x, size_y], x.device)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = torch.cat((x, grid), dim=-1)  # (batch, height, width, channels+2)

        # Input projection
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, height, width)

        # Fourier layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Output projection
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (batch, out_channels, height, width)

        return x

    @property
    def model_size(self):
        """计算模型参数数量，保持与原DenseED接口一致"""
        n_params = sum(p.numel() for p in self.parameters())
        n_layers = len([m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear, SpectralConv2d))])
        return n_params, n_layers


class FNO2d_DualTask(nn.Module):
    """
    FNO backbone (n_layers 层) + 双头：
    - 图像头：输出 (T, qx, qy)
    - 标量头：输出 keff
    """
    def __init__(self, modes1=12, modes2=12, width=64, in_channels=3,
                 out_channels_img=3, out_channels_scalar=1, n_layers=4):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels_img = out_channels_img
        self.out_channels_scalar = out_channels_scalar
        self.n_layers = n_layers

        # 输入投影（显式坐标 +2）
        self.fc0 = nn.Linear(in_channels + 2, width)

        # n 层谱卷积 + 1x1 残差
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.w1x1 = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])

        # 图像输出头
        self.fc1_img = nn.Linear(width, 128)
        self.fc2_img = nn.Linear(128, out_channels_img)

        # 标量输出头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1_scalar = nn.Linear(width, 64)
        self.fc2_scalar = nn.Linear(64, 32)
        self.fc3_scalar = nn.Linear(32, out_channels_scalar)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=torch.float32).view(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=torch.float32).view(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, _, H, W = x.shape

        # 位置编码并投影到 width
        grid = self.get_grid([B, H, W], x.device)
        x = x.permute(0, 2, 3, 1)           # (B,H,W,C)
        x = torch.cat((x, grid), dim=-1)    # (B,H,W,C+2)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)           # (B,width,H,W)

        # n 层 FNO block
        for sconv, w in zip(self.spec, self.w1x1):
            y = sconv(x) + w(x)
            x = F.gelu(y)

        # 图像输出头
        x_img = x.permute(0, 2, 3, 1)       # (B,H,W,width)
        x_img = F.gelu(self.fc1_img(x_img))
        x_img = self.fc2_img(x_img)
        x_img = x_img.permute(0, 3, 1, 2)   # (B,3,H,W)

        # 标量输出头
        x_scalar = self.global_pool(x).view(B, -1)  # (B,width)
        x_scalar = F.gelu(self.fc1_scalar(x_scalar))
        x_scalar = F.gelu(self.fc2_scalar(x_scalar))
        x_scalar = self.fc3_scalar(x_scalar)        # (B,1)

        return x_img, x_scalar




    @property
    def model_size(self):
        """计算模型参数数量"""
        n_params = sum(p.numel() for p in self.parameters())
        n_layers = len([m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear, SpectralConv2d))])
        return n_params, n_layers
def build_model(in_channels=3, width=64, modes1=16, modes2=16, n_layers=4):
    return FNO2d_DualTask(in_channels=in_channels, width=width,

                          modes1=modes1, modes2=modes2, n_layers=n_layers)
