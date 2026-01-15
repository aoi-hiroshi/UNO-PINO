import os, csv, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.dataloader import make_splits, make_splits_from_h5
from models import build_model
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
# ======================= 默认配置 =======================
DEFAULTS = dict(
    h5_path="data/test_porous_bc_384.hdf5",
    ckpt="results/cons_ufno_un_train2/checkpoints/best.pt",
    save_root="results_test",
    subname="test1",
    model="ufno",           # fno | ufno | denseed
    grad="con",                # 仅用于 tag
    tag=None,                  # None -> {grad}_{model}_test
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
    num_workers=0,
    n_vis=-1,

    in_channels=3,
    width=96,
    modes1=[48, 32, 24, 16],   # enc1, enc2, enc3, bottleneck
    modes2=[48, 32, 24, 17],   # 32x32 处 m2 ≤ 17
    n_layers=9,                # 计算depth层数

    # 可视化
    cmap="RdBu_r",
    err_cmap="RdBu_r",
    cmap_porous="gray",
    origin="lower",
    dpi=300,

    # 显示范围
    T_min=0.0,
    T_max=1.0,
    Q_abs=2.0,
    transpose_hw=True,

    # keff 散点可视范围
    keff_min=0.52,
    keff_max=0.62,

    # 指标
    enable_mape=False,
    mape_eps=1e-6,
    eps=1e-8,

    # PDE 残差
    k_solid=1.0,
    k_pore=0.2,
    bin_thresh=0.5,
    pde_dx=1.0 / (384 - 1),
    pde_dy=1.0 / (384 - 1),
    save_pde_vis=True,
    pde_clip_pct=99.0,

    # 误差频谱
    save_spectrum=True,
    spectrum_max_to_save=106,        # 单样本频谱保存数量
    spectrum_avg=True,              # 保存全测试集平均误差谱
    spectrum_radial=True,           # 保存环向平均曲线
    spectrum_radial_csv=True,       # 导出 csv

    # 校准曲线
    save_calib=True,
    calib_bins=12,

    # 孔隙率（可选）
    porosity=True,                 # 开启则保存孔隙率直方图 & 每样本 CSV
    porosity_bins=21,
)

# ======================= 基础工具 =======================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x

def _batch_to_targets(batch):
    if isinstance(batch, dict):
        y = {k: batch[k] for k in ("T", "qx", "qy", "keff") if k in batch}
        # 兼容别名
        if "keff" not in y:
            for alias in ("keffC", "k_eff", "keff_head"):
                if alias in batch:
                    y["keff"] = batch[alias]
                    break
        return batch["x"], (y if y else None)
    return batch, None

def _transpose_if_needed(t):
    # (B,1,H,W) -> (B,1,W,H)
    return t.permute(0, 1, 3, 2).contiguous()

#======================= Conservative helpers ===========================
def make_k_field_from_mask_np(porous, k_solid, k_pore, bin_thresh=0.5):
    # porous>=thr 视为“孔隙”，对应 k_pore
    m = (porous < float(bin_thresh))
    return np.where(m, float(k_pore), float(k_solid)).astype(np.float32)

def conservative_q_from_T_np(T, k, dx, dy, eps=1e-12):
    """
    给中心温度 T(H,W) 与中心 k(H,W)，在 x/y 面上做调和平均系数，
    得到中心位置的 qx, qy（通过面通量回到中心）。
    """
    H, W = T.shape

    # x-向面系数 (H, W-1) ；调和平均
    kx_face = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:] + eps)
    qx_face = - kx_face * (T[:, 1:] - T[:, :-1]) / dx  # 面通量

    qx = np.zeros_like(T, dtype=np.float32)
    if W > 1:
        qx[:, 1:-1] = 0.5 * (qx_face[:, 1:] + qx_face[:, :-1])
        qx[:, 0] = qx_face[:, 0]
        qx[:, -1] = qx_face[:, -1]

    # y-向面系数 (H-1, W)
    ky_face = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :] + eps)
    qy_face = - ky_face * (T[1:, :] - T[:-1, :]) / dy

    qy = np.zeros_like(T, dtype=np.float32)
    if H > 1:
        qy[1:-1, :] = 0.5 * (qy_face[1:, :] + qy_face[:-1, :])
        qy[0, :] = qy_face[0, :]
        qy[-1, :] = qy_face[-1, :]

    return qx.astype(np.float32), qy.astype(np.float32)

def conservative_div_from_center_np(qx_center, qy_center, dx, dy):
    """
    用中心 q近似“面通量”，做守恒散度：
      div ≈ (q_e - q_w)/dx + (q_n - q_s)/dy
    其中 q_e 为当前单元右面的通量（由相邻两中心均值近似）
    """
    H, W = qx_center.shape
    # x 方向面的通量
    qx_e = np.zeros_like(qx_center, dtype=np.float32)
    if W > 1:
        qx_e[:, :-1] = 0.5 * (qx_center[:, :-1] + qx_center[:, 1:])
        qx_e[:, -1] = qx_center[:, -1]
    else:
        qx_e[:] = qx_center

    # 左面通量
    qx_w = np.zeros_like(qx_center, dtype=np.float32)
    if W > 1:
        qx_w[:, 1:] = qx_e[:, :-1]
        qx_w[:, 0] = qx_center[:, 0]
    else:
        qx_w[:] = qx_center

    dqx = (qx_e - qx_w) / dx

    # y 方向面的通量（上面）
    qy_n = np.zeros_like(qy_center, dtype=np.float32)
    if H > 1:
        qy_n[:-1, :] = 0.5 * (qy_center[:-1, :] + qy_center[1:, :])
        qy_n[-1, :] = qy_center[-1, :]
    else:
        qy_n[:] = qy_center

    # 下面通量
    qy_s = np.zeros_like(qy_center, dtype=np.float32)
    if H > 1:
        qy_s[1:, :] = qy_n[:-1, :]
        qy_s[0, :] = qy_center[0, :]
    else:
        qy_s[:] = qy_center

    dqy = (qy_n - qy_s) / dy
    return (dqx + dqy).astype(np.float32)

# ======= 频谱工具（log|FFT| + 环向平均）=======
def _fft_mag_log(x2d: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(x2d)
    F = np.fft.fftshift(F)
    return np.log(1.0 + np.abs(F))

def _radial_profile(arr2d: np.ndarray) -> np.ndarray:
    h, w = arr2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)
    tbin = np.bincount(rr.ravel(), arr2d.ravel())
    nr = np.bincount(rr.ravel())
    return tbin / np.maximum(nr, 1)

# ======================= 指标 =======================
def mse(a, b):   return torch.mean((a - b) ** 2).item()
def mae(a, b):   return torch.mean(torch.abs(a - b)).item()
def rmse(a, b):  return math.sqrt(mse(a, b))
def rel_l2(a, b, eps=1e-8):
    num = torch.linalg.vector_norm(a - b)
    den = torch.linalg.vector_norm(b) + eps
    return (num / den).item()
def abs_l2(a, b):
    return torch.linalg.vector_norm(a - b).item()
def mape_stable(a, b, eps=1e-6):
    return torch.mean(torch.abs(a - b) / (b.abs() + eps)).item()
def smape(a, b, eps=1e-6):
    return torch.mean(torch.abs(a - b) / (0.5 * (a.abs() + b.abs()) + eps)).item()

# ==== PSNR / SSIM ====
def psnr_from_mse(mse_val: float, data_range: float, eps: float = 1e-12) -> float:
    # PSNR = 20*log10(MAX_I) - 10*log10(MSE)
    dr = max(float(data_range), eps)
    m = max(float(mse_val), eps)
    return 20.0 * math.log10(dr) - 10.0 * math.log10(m)

def _gaussian_kernel(win_size: int = 11, sigma: float = 1.5,
                     channels: int = 1, device="cpu", dtype=torch.float32):
    ax = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    g1 = g1 / (g1.sum() + 1e-12)
    g2 = (g1[:, None] @ g1[None, :])
    g2 = g2 / (g2.sum() + 1e-12)
    return g2.view(1, 1, win_size, win_size).repeat(channels, 1, 1, 1)

@torch.no_grad()
def ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range: float,
               win_size: int = 11, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03) -> float:
    """
    x, y: (B,1,H,W) 或 (1,1,H,W)，返回均值 SSIM（float）
    """
    assert x.ndim == 4 and y.ndim == 4, "expect (B,1,H,W)"
    C = x.shape[1]
    dev, dt = x.device, x.dtype
    kernel = _gaussian_kernel(win_size, sigma, channels=C, device=dev, dtype=dt)
    pad = win_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    L = float(max(data_range, 1e-12))
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean().item())

# ======================= 可视化（场 & 散点） =======================
def save_field_quad_with_porous(
    porous, gt, pred, tag, out_png, vmin, vmax,
    cmap="RdBu_r", err_cmap="RdBu_r", cmap_porous="gray", origin="lower", dpi=300,
    metrics_text: str | None = None,
):

    ensure_dir(os.path.dirname(out_png))
    err = pred - gt
    vmax_e = float(max(abs(err.min()), abs(err.max()), 1e-12))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)

    im0 = axs[0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    if str(tag).lower() == "t":
        axs[0].set_title("GT")
    axs[0].axis("off")
    _ = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    if str(tag).lower() == "t":
        axs[1].set_title("PRED")
    axs[1].axis("off")
    _ = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(err, cmap=err_cmap, vmin=-vmax_e, vmax=vmax_e, origin=origin)
    if str(tag).lower() == "t":
        axs[2].set_title("ERROR")
    axs[2].axis("off")
    _ = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    _ = metrics_text  # kept for API compatibility; intentionally not drawn

    fig.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_T_isolines(
    T_gt, T_pred, out_png, vmin, vmax,
    cmap="RdBu_r", origin="lower", dpi=300,
):

    ensure_dir(os.path.dirname(out_png))
    H, W = T_gt.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    levels = np.arange(0.1, 1.0, 0.1)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)

    im0 = axs[0].imshow(T_gt, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    CS_gt = axs[0].contour(X, Y, T_gt, levels=levels, colors="k", linewidths=0.8)
    axs[0].clabel(CS_gt, inline=True, fmt="%.1f", fontsize=8)
    axs[0].set_title("T GT + isolines")
    axs[0].axis("off")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(T_pred, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    CS_pred = axs[1].contour(X, Y, T_pred, levels=levels, colors="k", linewidths=0.8)
    axs[1].clabel(CS_pred, inline=True, fmt="%.1f", fontsize=8)
    axs[1].set_title("T Pred + isolines")
    axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close()

def scatter_keff(keff_pred, keff_gt, out_png, lo=0.52, hi=0.62):
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(keff_gt, keff_pred, s=14, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("keff GT")
    plt.ylabel("keff Pred")
    plt.title("keff scatter (per-image)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def scatter_field_pixels(pred_list, gt_list, field_name, out_png, max_points=200_000, seed=123):
    rng = np.random.default_rng(seed)
    xs = [arr.reshape(-1) for arr in pred_list]
    ys = [arr.reshape(-1) for arr in gt_list]
    if not xs:
        return
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.shape[0] > max_points:
        idx = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(y, x, s=1, alpha=0.35)
    lo = float(min(y.min(), x.min()))
    hi = float(max(y.max(), x.max()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel(f"{field_name} GT")
    plt.ylabel(f"{field_name} Pred")
    plt.title(f"{field_name} scatter ({x.size} points)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_scatter_csv(pred_list, gt_list, out_csv, max_points=200_000, seed=123):
    """
    把多个 2D 场打平成散点，最多随机采样 max_points 个，保存成两列 CSV：GT, Pred
    """
    rng = np.random.default_rng(seed)
    xs = [arr.reshape(-1) for arr in pred_list]
    ys = [arr.reshape(-1) for arr in gt_list]
    if (not xs) or (not ys):
        return
    x = np.concatenate(xs)   # Pred
    y = np.concatenate(ys)   # GT
    n = x.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT", "Pred"])
        for g, p in zip(y, x):
            w.writerow([f"{float(g):.6e}", f"{float(p):.6e}"])

def save_relL2_boxplot(stats_dict, out_png):
    data, labels = [], []
    for name in ("T_relL2", "qx_relL2", "qy_relL2", "keff_relL2"):
        if name in stats_dict and len(stats_dict[name]) > 0:
            data.append(stats_dict[name])
            labels.append(name.replace("_relL2", ""))
    if not data:
        return
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(5, 4), dpi=300)
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel("Relative L2 Error")
    plt.title("Error Distribution across Test Samples")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close()

def save_relL2_per_image(stats_dict, out_png, out_csv):

    T_vals  = stats_dict.get("T_relL2", [])
    qx_vals = stats_dict.get("qx_relL2", [])
    qy_vals = stats_dict.get("qy_relL2", [])

    N = max(len(T_vals), len(qx_vals), len(qy_vals))
    if N == 0:
        return

    idx = np.arange(N)

    # 图
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(6, 4), dpi=300)
    if len(T_vals):
        plt.scatter(idx[:len(T_vals)], T_vals, s=18, label="T")
    if len(qx_vals):
        plt.scatter(idx[:len(qx_vals)], qx_vals, s=18, label="qx")
    if len(qy_vals):
        plt.scatter(idx[:len(qy_vals)], qy_vals, s=18, label="qy")
    plt.xlabel("Image Index")
    plt.ylabel("Relative L2 Error")
    plt.title("Relative L2 Error per Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    # CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Index", "T_relL2", "qx_relL2", "qy_relL2"])
        for i in range(N):
            t  = T_vals[i]  if i < len(T_vals)  else float("nan")
            qx = qx_vals[i] if i < len(qx_vals) else float("nan")
            qy = qy_vals[i] if i < len(qy_vals) else float("nan")
            w.writerow([i, f"{t:.16g}", f"{qx:.16g}", f"{qy:.16g}"])

# ============== 热流方向箭头图 ==============
def save_q_arrow_field(
    qx, qy, background, out_png, vmin, vmax,
    cmap="RdBu_r", origin="lower", dpi=300,
    mode="both",           # "qx_only" / "qy_only" / "both"
    max_arrows=50,         # x/y 方向最多多少个箭头
    arrow_scale=0.3,       # 箭头放大系数
):

    ensure_dir(os.path.dirname(out_png))
    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(background, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)

    H, W = qx.shape
    skip = max(1, max(H, W) // max_arrows)

    X = np.arange(0, W, skip)
    Y = np.arange(0, H, skip)
    Xg, Yg = np.meshgrid(X, Y)

    qx_s = qx[::skip, ::skip]
    qy_s = qy[::skip, ::skip]

    if mode == "qx_only":
        U = qx_s
        V = np.zeros_like(qx_s)
    elif mode == "qy_only":
        U = np.zeros_like(qy_s)
        V = qy_s
    else:  # both
        U = qx_s
        V = qy_s

    mag = np.sqrt(U**2 + V**2)
    maxmag = np.percentile(mag, 95)
    if maxmag <= 1e-12:
        maxmag = 1.0

    scale = arrow_scale * float(skip) / maxmag
    U_plot = U * scale
    V_plot = V * scale

    ax.quiver(
        Xg, Yg, U_plot, V_plot,
        angles="xy", scale_units="xy", scale=1.0,
        width=0.002
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ============== PDE 残差 ==============
def _central_diff(f, dx, dy):
    fx = np.zeros_like(f, dtype=np.float32)
    fy = np.zeros_like(f, dtype=np.float32)
    fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    fy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
    fx[:, 0] = (f[:, 1] - f[:, 0]) / dx
    fx[:, -1] = (f[:, -1] - f[:, -2]) / dx
    fy[0, :] = (f[1, :] - f[0, :]) / dy
    fy[-1, :] = (f[-1, :] - f[-2, :]) / dy
    return fx, fy

def _percentile_clip(a: np.ndarray, pct: float):
    vmax = float(np.percentile(np.abs(a), pct))
    if vmax <= 0:
        vmax = float(np.abs(a).max()) + 1e-12
    return vmax

def _rms_np(a: np.ndarray) -> float:
    """Root-mean-square over all pixels."""
    a = a.astype(np.float64)
    return float(np.sqrt(np.mean(a * a)))

def save_pde_residual_maps(
    porous_np, T_pred_np, qx_pred_np, qy_pred_np,
    out_dir, k_solid=1.0, k_pore=0.2, bin_thresh=0.5, dx=1.0, dy=1.0,
    cmap="RdBu_r", origin="lower", dpi=300, clip_pct=99.0
):

    ensure_dir(out_dir)

    k_field = make_k_field_from_mask_np(porous_np, k_solid, k_pore, bin_thresh)
    qx_cons, qy_cons = conservative_q_from_T_np(
        T_pred_np.astype(np.float32), k_field, dx, dy
    )

    # 残差
    rFx = qx_pred_np.astype(np.float32) - qx_cons
    rFy = qy_pred_np.astype(np.float32) - qy_cons
    rF_mag = np.sqrt(rFx * rFx + rFy * rFy)
    rDiv = conservative_div_from_center_np(
        qx_pred_np.astype(np.float32), qy_pred_np.astype(np.float32), dx, dy
    )
    scale_sqrt = math.sqrt(0.5 * (dx * dx + dy * dy))
    rDiv_vis = rDiv * scale_sqrt

    # 动态范围
    def _clip(a):
        vmax = float(np.percentile(np.abs(a), clip_pct))
        if vmax <= 0:
            vmax = float(np.abs(a).max()) + 1e-12
        return vmax

    vmax_rFx = _clip(rFx)
    vmax_rFy = _clip(rFy)
    vmax_rDiv = _clip(rDiv_vis)
    vmax_rF = _clip(rF_mag)

    def _save(arr, name, vlim=None, symmetric=True):
        plt.figure(figsize=(4, 4), dpi=dpi)
        if symmetric:
            im = plt.imshow(arr, cmap=cmap, origin=origin, vmin=-vlim, vmax=vlim)
        else:
            im = plt.imshow(arr, cmap=cmap, origin=origin, vmin=0.0, vmax=vlim)
        plt.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight", pad_inches=0.02)
        plt.close()

    _save(rFx, "pde_rFx", vmax_rFx, symmetric=True)
    _save(rFy, "pde_rFy", vmax_rFy, symmetric=True)
    _save(rDiv_vis, "pde_rDiv", vmax_rDiv, symmetric=True)
    _save(rF_mag, "pde_abs_fourier", vmax_rF, symmetric=False)

# ============== 误差频谱（FFT） ==============
def _fft_mag(x2d: np.ndarray) -> np.ndarray:
    """归一化幅值谱，用于 log|FFT| 可视化"""
    F = np.fft.fft2(x2d)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    mag /= (mag.max() + 1e-12)
    return mag

def _fft_power(x2d: np.ndarray) -> np.ndarray:
    """功率谱 |FFT|^2，用于能量和相对误差计算"""
    F = np.fft.fft2(x2d)
    F = np.fft.fftshift(F)
    return np.abs(F) ** 2

def _radial_spectrum_1d(spec2d: np.ndarray) -> np.ndarray:
    """
    对给定 2D 频谱做环向平均，返回半径 r 上的平均值 E(r)。
    输入可以是幅值也可以是功率，这里不区分单位。
    """
    h, w = spec2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)
    tbin = np.bincount(rr.ravel(), spec2d.ravel())
    nr = np.bincount(rr.ravel())
    return tbin / np.maximum(nr, 1)

def save_error_spectra_bundle(err_all, gt_all, spec_root, field_name,
                              max_to_save=106, dpi=300, cmap="magma"):
    """
    err_all: list of (H,W) error arrays
    gt_all:  list of (H,W) ground-truth arrays（与 err_all 一一对应）
    spec_root: <...>/spectra
    field_name: "T" | "qx" | "qy"
    """
    ensure_dir(spec_root)
    out_dir = os.path.join(spec_root, "err_fft", field_name)
    ensure_dir(out_dir)

    # ---------- 逐图 log|FFT(error)| 预览 ----------
    for i, e in enumerate(err_all[:max_to_save]):
        spec = _fft_mag(e)   # 归一化幅值
        plt.figure(figsize=(4, 4), dpi=dpi)
        im = plt.imshow(np.log(spec + 1e-6), cmap=cmap)
        plt.axis("off")
        plt.title(f"{field_name}_err_fft #{i}")
        plt.colorbar(im, fraction=0.046, pad=0.04)  # 加 colorbar
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{field_name}_err_fft_{i:03d}.png"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close()

    if len(err_all) == 0:
        return

    # ---------- 平均 log|FFT(error)| 2D 图 ----------
    specs = [_fft_mag(e) for e in err_all]
    avg = np.mean(specs, axis=0)

    plt.figure(figsize=(5, 4), dpi=dpi)
    im = plt.imshow(np.log(avg + 1e-6), cmap=cmap)
    plt.axis("off")
    plt.title(f"{field_name} average log|FFT(error)|")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{field_name}_err_fft_avg.png"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    # ---------- 相对误差径向谱（ESP 风格） ----------
    rel_radials = []
    eps = 1e-12
    for e, g in zip(err_all, gt_all):
        # 误差功率谱和真值功率谱
        P_err = _fft_power(e)
        P_gt  = _fft_power(g)

        E_err = _radial_spectrum_1d(P_err)   # ErrorEnergy(r)
        E_gt  = _radial_spectrum_1d(P_gt)    # TrueEnergy(r)

        L = min(len(E_err), len(E_gt))
        rel = np.sqrt(E_err[:L] / (E_gt[:L] + eps))   # 相对误差谱
        rel_radials.append(rel)

    L = min(len(r) for r in rel_radials)
    rel_radials = [r[:L] for r in rel_radials]
    avg_rel = np.mean(rel_radials, axis=0)

    # 曲线图
    plt.figure(figsize=(5, 3), dpi=dpi)
    plt.plot(avg_rel, lw=1.8)
    plt.xlabel("Radial frequency (pixel$^{-1}$)")
    plt.ylabel("Avg relative error")
    plt.title(f"{field_name} radial error spectrum (avg)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{field_name}_err_fft_radial.png"))
    plt.close()

    # CSV 导出：k, avg_relative_error
    with open(os.path.join(out_dir, f"{field_name}_err_fft_radial.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "avg_relative_error"])
        for k_idx, v in enumerate(avg_rel):
            w.writerow([k_idx, f"{float(v):.6e}"])

# ---------- 频谱：Pred/GT 三联图（保存到 pairs/<field>/） ----------
def save_spectrum_pair_with_metrics(pred2d, gt2d, out_dir, filename_no_ext,
                                    dpi=300, cmap="magma"):
    """
    out_dir: <...>/spectra/pairs/<field>/
    filename_no_ext: "000000" -> 保存 000000_spectrum_pair.png

    Note: For clean figure export, all in-figure text is suppressed for these triplets
          EXCEPT that for T we keep the three panel titles: GT / PRED / ERROR.
    """
    ensure_dir(out_dir)
    S_pred = _fft_mag_log(pred2d)  # log|FFT|
    S_gt   = _fft_mag_log(gt2d)
    D_abs  = np.abs(S_pred - S_gt)
    delta_S_L1 = float(np.mean(D_abs))

    R_pred = _radial_profile(S_pred)
    R_gt   = _radial_profile(S_gt)
    L = min(len(R_pred), len(R_gt))
    delta_R_L1 = float(np.mean(np.abs(R_pred[:L] - R_gt[:L])))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), dpi=dpi)

    # Infer which field this 'pairs' directory corresponds to (T/qx/qy)
    field_key = os.path.basename(os.path.normpath(out_dir)).lower()

    im0 = axs[0].imshow(S_gt, cmap=cmap)
    if field_key == "t":
        axs[0].set_title("GT")
    axs[0].axis("off")
    _ = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(S_pred, cmap=cmap)
    if field_key == "t":
        axs[1].set_title("PRED")
    axs[1].axis("off")
    _ = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(D_abs, cmap=cmap)
    if field_key == "t":
        axs[2].set_title("ERROR")
    axs[2].axis("off")
    _ = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{filename_no_ext}_spectrum_pair.png"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    # 环向谱 csv
    csv_dir = os.path.join(out_dir, "csv")
    ensure_dir(csv_dir)

    def _csv(path, arr):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "value"])
            for i, v in enumerate(arr):
                w.writerow([i, f"{float(v):.6e}"])

    _csv(os.path.join(csv_dir, f"{filename_no_ext}_radial_pred.csv"), R_pred)
    _csv(os.path.join(csv_dir, f"{filename_no_ext}_radial_gt.csv"), R_gt)

    return delta_S_L1, delta_R_L1
def save_porosity_stats(porous_list, out_dir, bins=21):
    ensure_dir(out_dir)
    phis = []
    for P in porous_list:
        m = (P < 0.5).astype(np.float32)  # porous≈0 为孔隙
        phis.append(float(m.mean()))
    plt.figure(figsize=(4, 3), dpi=300)
    plt.hist(phis, bins=bins, alpha=0.85)
    plt.xlabel("Porosity")
    plt.ylabel("Count")
    plt.title("Porosity histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "porosity_hist.png"))
    plt.close()
    with open(os.path.join(out_dir, "porosity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "porosity"])
        for i, phi in enumerate(phis):
            w.writerow([i, f"{phi:.6f}"])

# ============== Loader & Model ==============
def build_test_loader(h5_path, batch_size, num_workers):
    try:
        _, _, test_ds = make_splits_from_h5(h5_path)
        print("[split] using preset test split from H5")
    except Exception:
        _, _, test_ds = make_splits(
            h5_path, n_train=800, n_val=200, n_test=106, seed=42
        )
        print("[split] fallback random split")
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

def load_model_via_factory(model_name, device, in_channels, width, modes1, modes2, n_layers, ckpt_path):
    name = model_name.strip().lower()
    common = dict(in_channels=in_channels, width=width, n_layers=n_layers)
    if name in ("fno", "ufno"):
        common.update(dict(modes1=modes1, modes2=modes2))
    model = build_model(name=name, **common).to(device)

    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("model", state)
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            from collections import OrderedDict
            new_sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
            model.load_state_dict(new_sd, strict=False)
        print(f"[{name}] loaded: {ckpt_path}")
    else:
        print(f"[{name}] WARN: checkpoint not found: {ckpt_path}, using random weights.")
    return model

# ============== 评估核心 ==============
@torch.no_grad()
def evaluate_model(model, loader, device, out_root, subname, cfg_tag,
                   cmap="RdBu_r", err_cmap="RdBu_r", cmap_porous="gray",
                   origin="lower", dpi=300, n_vis=-1,
                   T_min=0.0, T_max=1.0, Q_abs=2.0, transpose_hw=True,
                   keff_lo=0.40, keff_hi=0.65,
                   enable_mape=False, mape_eps=1e-6, eps=1e-8,
                   k_solid=1.0, k_pore=0.2, bin_thresh=0.5, pde_dx=1.0, pde_dy=1.0,
                   save_pde_vis=True, pde_clip_pct=99.0,
                   save_spectrum=True, spectrum_max_to_save=16, spectrum_avg=True,
                   spectrum_radial=True, spectrum_radial_csv=True,
                   save_calib=True, calib_bins=12,
                   porosity=True, porosity_bins=21):
    model.eval()

    out_dir = os.path.join(out_root, subname, cfg_tag)
    quad_T = os.path.join(out_dir, "quad_T")
    quad_qx = os.path.join(out_dir, "quad_qx")
    quad_qy = os.path.join(out_dir, "quad_qy")
    T_iso_dir = os.path.join(out_dir, "T_isolines")  # 新增：单独的 T 等温线目录
    qx_arrow_dir = os.path.join(out_dir, "qx_arrow")
    qy_arrow_dir = os.path.join(out_dir, "qy_arrow")

    # PDE 可视化：按类型拆分子目录，方便遍历
    pde_all_root = os.path.join(out_root, subname, cfg_tag, "vis_pde_residuals_all")
    pde_rFx_dir = os.path.join(pde_all_root, "rFx")
    pde_rFy_dir = os.path.join(pde_all_root, "rFy")
    pde_rDiv_dir = os.path.join(pde_all_root, "rDiv")
    pde_rFmag_dir = os.path.join(pde_all_root, "rF_mag")

    for p in (
        out_dir,
        quad_T,
        quad_qx,
        quad_qy,
        T_iso_dir,
        qx_arrow_dir,
        qy_arrow_dir,
        pde_all_root,
        pde_rFx_dir,
        pde_rFy_dir,
        pde_rDiv_dir,
        pde_rFmag_dir,
    ):
        ensure_dir(p)

    # 量程（供 PSNR/SSIM 使用）：固定为配置范围，保证横向可比
    dr_T = float(T_max - T_min)
    dr_Q = float(2.0 * Q_abs)

    # stats 容器
    stats = dict(
        T_mse=[],
        T_mae=[],
        T_rmse=[],
        T_relL2=[],
        T_absL2=[],
        T_smape=[],
        qx_mse=[],
        qx_mae=[],
        qx_rmse=[],
        qx_relL2=[],
        qx_absL2=[],
        qx_smape=[],
        qy_mse=[],
        qy_mae=[],
        qy_rmse=[],
        qy_relL2=[],
        qy_absL2=[],
        qy_smape=[],
        keff_mse=[],
        keff_mae=[],
        keff_rmse=[],
        keff_relL2=[],
        keff_absL2=[],
        keff_smape=[],
        # 新增：PSNR / SSIM
        T_psnr=[],
        T_ssim=[],
        qx_psnr=[],
        qx_ssim=[],
        qy_psnr=[],
        qy_ssim=[],
    )
    if enable_mape:
        for k in ("T_mape", "qx_mape", "qy_mape", "keff_mape"):
            stats[k] = []

    # R² 累计
    T_sum = T_sum_sq = T_sse = T_cnt = 0.0
    qx_sum = qx_sum_sq = qx_sse = qx_cnt = 0.0
    qy_sum = qy_sum_sq = qy_sse = qy_cnt = 0.0
    k_sum = k_sum_sq = k_sse = k_cnt = 0.0

    vis_saved = 0
    save_all = (n_vis is None) or (n_vis <= 0)
    idx_global = 0
    has_gt_any = False

    # 像素散点收集（用于散点图）
    T_pred_imgs, T_gt_imgs = [], []
    qx_pred_imgs, qx_gt_imgs = [], []
    qy_pred_imgs, qy_gt_imgs = [], []

    # 散点 CSV 用：尽量收集所有样本，最后统一随机采样
    T_pred_all, T_gt_all = [], []
    qx_pred_all, qx_gt_all = [], []
    qy_pred_all, qy_gt_all = [], []

    # keff（散点/校准）
    keff_pred_all, keff_gt_all = [], []
    keff_pred_for_csv, keff_gt_for_csv = [], []

    # 误差频谱（收集）
    err_T_all, err_qx_all, err_qy_all = [], [], []

    # 孔隙率
    porous_all = []
    pde_csv_rows = []

    for batch in loader:
        x, y_gt = _batch_to_targets(batch)
        x = _to(x, device)
        if y_gt is not None:
            y_gt = _to(y_gt, device)

        y_img, y_keff = model(x)  # (B,3,H,W), (B,1)
        B = x.size(0)

        for b in range(B):
            T_pred = y_img[b:b + 1, 0:1]
            qx_pred = y_img[b:b + 1, 1:2]
            qy_pred = y_img[b:b + 1, 2:3]
            ke_pred = y_keff[b:b + 1]

            T_gt = qx_gt = qy_gt = ke_gt = None
            if y_gt is not None:
                T_gt = y_gt.get("T", None)
                T_gt = T_gt[b:b + 1] if T_gt is not None else None
                qx_gt = y_gt.get("qx", None)
                qx_gt = qx_gt[b:b + 1] if qx_gt is not None else None
                qy_gt = y_gt.get("qy", None)
                qy_gt = qy_gt[b:b + 1] if qy_gt is not None else None
                ke_gt = y_gt.get("keff", None)
                ke_gt = ke_gt[b:b + 1] if ke_gt is not None else None
                if transpose_hw:
                    if T_gt is not None:
                        T_gt = _transpose_if_needed(T_gt)
                    if qx_gt is not None:
                        qx_gt = _transpose_if_needed(qx_gt)
                    if qy_gt is not None:
                        qy_gt = _transpose_if_needed(qy_gt)

            vmin_T, vmax_T = float(T_min), float(T_max)
            vmin_Q, vmax_Q = -float(Q_abs), float(Q_abs)
            porous_np = x[b, 0].detach().cpu().numpy()  # (H,W)

            if porosity:
                porous_all.append(porous_np.copy())

            # ===== 每样本量化 & 归一化 absL2，先算指标再画联图 =====
            metrics_text_T = metrics_text_qx = metrics_text_qy = None

            if T_gt is not None:
                has_gt_any = True
                _mse_T = mse(T_pred, T_gt)
                _rmse_T = math.sqrt(_mse_T)
                _psnr_T = psnr_from_mse(_mse_T, data_range=dr_T)
                _ssim_T = ssim_torch(T_pred, T_gt, data_range=dr_T)
                _absL2_T = abs_l2(T_pred, T_gt)
                Npix_T = T_pred.numel()
                _absL2_T_norm = float(_absL2_T / (math.sqrt(Npix_T) * dr_T + 1e-12))
                # 写入 stats
                stats["T_mse"].append(_mse_T)
                stats["T_mae"].append(mae(T_pred, T_gt))
                stats["T_rmse"].append(_rmse_T)
                stats["T_relL2"].append(rel_l2(T_pred, T_gt, eps))
                stats["T_absL2"].append(_absL2_T)
                stats.setdefault("T_absL2_norm", []).append(_absL2_T_norm)
                stats["T_smape"].append(smape(T_pred, T_gt, eps=mape_eps))
                if enable_mape:
                    stats["T_mape"].append(mape_stable(T_pred, T_gt, eps=mape_eps))
                stats["T_psnr"].append(_psnr_T)
                stats["T_ssim"].append(_ssim_T)
                # 误差场收集
                err_T_all.append((T_pred[0, 0] - T_gt[0, 0]).detach().cpu().numpy())
                # 全量保存，用于散点 CSV
                T_pred_all.append(T_pred[0, 0].detach().cpu().numpy())
                T_gt_all.append(T_gt[0, 0].detach().cpu().numpy())
                # 前 32 张用于散点可视化
                if len(T_pred_imgs) < 32:
                    T_pred_imgs.append(T_pred[0, 0].detach().cpu().numpy())
                    T_gt_imgs.append(T_gt[0, 0].detach().cpu().numpy())
                # R² 累计
                T_sum += float(T_gt.sum())
                T_sum_sq += float((T_gt ** 2).sum())
                T_sse += float(((T_pred - T_gt) ** 2).sum())
                T_cnt += T_gt.numel()
                # 四联图右侧量化条
                metrics_text_T = (
                    f"RMSE={_rmse_T:.4e}\n"
                    f"PSNR={_psnr_T:.2f}\n"
                    f"SSIM={_ssim_T:.3f}\n"
                    f"|.|2n={_absL2_T_norm:.3e}"
                )

            if qx_gt is not None:
                _mse_qx = mse(qx_pred, qx_gt)
                _rmse_qx = math.sqrt(_mse_qx)
                _psnr_qx = psnr_from_mse(_mse_qx, data_range=dr_Q)
                _ssim_qx = ssim_torch(qx_pred, qx_gt, data_range=dr_Q)
                _absL2_qx = abs_l2(qx_pred, qx_gt)
                Npix_qx = qx_pred.numel()
                _absL2_qx_norm = float(_absL2_qx / (math.sqrt(Npix_qx) * dr_Q + 1e-12))
                stats["qx_mse"].append(_mse_qx)
                stats["qx_mae"].append(mae(qx_pred, qx_gt))
                stats["qx_rmse"].append(_rmse_qx)
                stats["qx_relL2"].append(rel_l2(qx_pred, qx_gt, eps))
                stats["qx_absL2"].append(_absL2_qx)
                stats.setdefault("qx_absL2_norm", []).append(_absL2_qx_norm)
                stats["qx_smape"].append(smape(qx_pred, qx_gt, eps=mape_eps))
                if enable_mape:
                    stats["qx_mape"].append(mape_stable(qx_pred, qx_gt, eps=mape_eps))
                stats["qx_psnr"].append(_psnr_qx)
                stats["qx_ssim"].append(_ssim_qx)
                err_qx_all.append((qx_pred[0, 0] - qx_gt[0, 0]).detach().cpu().numpy())
                # 全量保存，用于散点 CSV
                qx_pred_all.append(qx_pred[0, 0].detach().cpu().numpy())
                qx_gt_all.append(qx_gt[0, 0].detach().cpu().numpy())
                # 前 32 张用于散点可视化
                if len(qx_pred_imgs) < 32:
                    qx_pred_imgs.append(qx_pred[0, 0].detach().cpu().numpy())
                    qx_gt_imgs.append(qx_gt[0, 0].detach().cpu().numpy())
                qx_sum += float(qx_gt.sum())
                qx_sum_sq += float((qx_gt ** 2).sum())
                qx_sse += float(((qx_pred - qx_gt) ** 2).sum())
                qx_cnt += qx_gt.numel()
                metrics_text_qx = (
                    f"RMSE={_rmse_qx:.4e}\n"
                    f"PSNR={_psnr_qx:.2f}\n"
                    f"SSIM={_ssim_qx:.3f}\n"
                    f"|.|2n={_absL2_qx_norm:.3e}"
                )

            if qy_gt is not None:
                _mse_qy = mse(qy_pred, qy_gt)
                _rmse_qy = math.sqrt(_mse_qy)
                _psnr_qy = psnr_from_mse(_mse_qy, data_range=dr_Q)
                _ssim_qy = ssim_torch(qy_pred, qy_gt, data_range=dr_Q)
                _absL2_qy = abs_l2(qy_pred, qy_gt)
                Npix_qy = qy_pred.numel()
                _absL2_qy_norm = float(_absL2_qy / (math.sqrt(Npix_qy) * dr_Q + 1e-12))
                stats["qy_mse"].append(_mse_qy)
                stats["qy_mae"].append(mae(qy_pred, qy_gt))
                stats["qy_rmse"].append(_rmse_qy)
                stats["qy_relL2"].append(rel_l2(qy_pred, qy_gt, eps))
                stats["qy_absL2"].append(_absL2_qy)
                stats.setdefault("qy_absL2_norm", []).append(_absL2_qy_norm)
                stats["qy_smape"].append(smape(qy_pred, qy_gt, eps=mape_eps))
                if enable_mape:
                    stats["qy_mape"].append(mape_stable(qy_pred, qy_gt, eps=mape_eps))
                stats["qy_psnr"].append(_psnr_qy)
                stats["qy_ssim"].append(_ssim_qy)
                err_qy_all.append((qy_pred[0, 0] - qy_gt[0, 0]).detach().cpu().numpy())
                # 全量保存，用于散点 CSV
                qy_pred_all.append(qy_pred[0, 0].detach().cpu().numpy())
                qy_gt_all.append(qy_gt[0, 0].detach().cpu().numpy())
                # 前 32 张用于散点可视化
                if len(qy_pred_imgs) < 32:
                    qy_pred_imgs.append(qy_pred[0, 0].detach().cpu().numpy())
                    qy_gt_imgs.append(qy_gt[0, 0].detach().cpu().numpy())
                qy_sum += float(qy_gt.sum())
                qy_sum_sq += float((qy_gt ** 2).sum())
                qy_sse += float(((qy_pred - qy_gt) ** 2).sum())
                qy_cnt += qy_gt.numel()
                metrics_text_qy = (
                    f"RMSE={_rmse_qy:.4e}\n"
                    f"PSNR={_psnr_qy:.2f}\n"
                    f"SSIM={_ssim_qy:.3f}\n"
                    f"|.|2n={_absL2_qy_norm:.3e}"
                )

            # ===== 四联图（右侧加量化条），外加单独 T 等温线图 =====
            if (T_gt is not None) and (save_all or vis_saved < n_vis):
                save_field_quad_with_porous(
                    porous_np,
                    T_gt[0, 0].cpu().numpy(),
                    T_pred[0, 0].cpu().numpy(),
                    "T",
                    os.path.join(quad_T, f"{idx_global:06d}.png"),
                    vmin=vmin_T,
                    vmax=vmax_T,
                    cmap=cmap,
                    err_cmap=err_cmap,
                    cmap_porous=cmap_porous,
                    origin=origin,
                    dpi=dpi,
                    metrics_text=metrics_text_T,
                )
                # 单独的 T 等温线
                save_T_isolines(
                    T_gt[0, 0].cpu().numpy(),
                    T_pred[0, 0].cpu().numpy(),
                    os.path.join(T_iso_dir, f"{idx_global:06d}.png"),
                    vmin=vmin_T,
                    vmax=vmax_T,
                    cmap=cmap,
                    origin=origin,
                    dpi=dpi,
                )
                if qx_gt is not None:
                    save_field_quad_with_porous(
                        porous_np,
                        qx_gt[0, 0].cpu().numpy(),
                        qx_pred[0, 0].cpu().numpy(),
                        "qx",
                        os.path.join(quad_qx, f"{idx_global:06d}.png"),
                        vmin=vmin_Q,
                        vmax=vmax_Q,
                        cmap=cmap,
                        err_cmap=err_cmap,
                        cmap_porous=cmap_porous,
                        origin=origin,
                        dpi=dpi,
                        metrics_text=metrics_text_qx,
                    )
                if qy_gt is not None:
                    save_field_quad_with_porous(
                        porous_np,
                        qy_gt[0, 0].cpu().numpy(),
                        qy_pred[0, 0].cpu().numpy(),
                        "qy",
                        os.path.join(quad_qy, f"{idx_global:06d}.png"),
                        vmin=vmin_Q,
                        vmax=vmax_Q,
                        cmap=cmap,
                        err_cmap=err_cmap,
                        cmap_porous=cmap_porous,
                        origin=origin,
                        dpi=dpi,
                        metrics_text=metrics_text_qy,
                    )
                # 热流箭头图
                qx_np = qx_pred[0, 0].detach().cpu().numpy()
                qy_np = qy_pred[0, 0].detach().cpu().numpy()

                # qx：在 qx 场上叠加 x 方向箭头
                save_q_arrow_field(
                    qx_np, qy_np,
                    background=qx_np,
                    out_png=os.path.join(qx_arrow_dir, f"{idx_global:06d}.png"),
                    vmin=vmin_Q, vmax=vmax_Q,
                    cmap=cmap, origin=origin, dpi=dpi,
                    mode="qx_only",        # 只保留 qx 分量
                    arrow_scale=1.0,       # 可以再调大/调小
                )

                # qy：在 qy 场上叠加 y 方向箭头
                save_q_arrow_field(
                    qx_np, qy_np,
                    background=qy_np,
                    out_png=os.path.join(qy_arrow_dir, f"{idx_global:06d}.png"),
                    vmin=vmin_Q, vmax=vmax_Q,
                    cmap=cmap, origin=origin, dpi=dpi,
                    mode="qy_only",        # 只保留 qy 分量（垂直箭头）
                    arrow_scale=1.0,
                )
                vis_saved += 1

            # ===== 频谱对比（Pred vs GT 的 log|FFT| 差） =====
            if save_spectrum and (save_all or vis_saved <= spectrum_max_to_save):
                spec_root = os.path.join(out_dir, "spectra")
                ensure_dir(spec_root)
                if T_gt is not None:
                    pair_dir = os.path.join(spec_root, "pairs", "T")
                    ensure_dir(pair_dir)
                    dS, dR = save_spectrum_pair_with_metrics(
                        pred2d=T_pred[0, 0].cpu().numpy(),
                        gt2d=T_gt[0, 0].cpu().numpy(),
                        out_dir=pair_dir,
                        filename_no_ext=f"{idx_global:06d}",
                        dpi=dpi,
                        cmap="magma",
                    )
                    stats.setdefault("T_deltaS_L1", []).append(float(dS))
                    stats.setdefault("T_deltaR_L1", []).append(float(dR))
                if qx_gt is not None:
                    pair_dir = os.path.join(spec_root, "pairs", "qx")
                    ensure_dir(pair_dir)
                    dS, dR = save_spectrum_pair_with_metrics(
                        pred2d=qx_pred[0, 0].cpu().numpy(),
                        gt2d=qx_gt[0, 0].cpu().numpy(),
                        out_dir=pair_dir,
                        filename_no_ext=f"{idx_global:06d}",
                        dpi=dpi,
                        cmap="magma",
                    )
                    stats.setdefault("qx_deltaS_L1", []).append(float(dS))
                    stats.setdefault("qx_deltaR_L1", []).append(float(dR))
                if qy_gt is not None:
                    pair_dir = os.path.join(spec_root, "pairs", "qy")
                    ensure_dir(pair_dir)
                    dS, dR = save_spectrum_pair_with_metrics(
                        pred2d=qy_pred[0, 0].cpu().numpy(),
                        gt2d=qy_gt[0, 0].cpu().numpy(),
                        out_dir=pair_dir,
                        filename_no_ext=f"{idx_global:06d}",
                        dpi=dpi,
                        cmap="magma",
                    )
                    stats.setdefault("qy_deltaS_L1", []).append(float(dS))
                    stats.setdefault("qy_deltaR_L1", []).append(float(dR))

            # —— 永远收集 keff 预测 ——（即使没有 GT）
            keff_pred_all.append(float(ke_pred.item()))
            if ke_gt is not None:
                keff_gt_all.append(float(ke_gt.item()))
                stats["keff_mse"].append(mse(ke_pred, ke_gt))
                stats["keff_mae"].append(mae(ke_pred, ke_gt))
                stats["keff_rmse"].append(rmse(ke_pred, ke_gt))
                stats["keff_relL2"].append(rel_l2(ke_pred, ke_gt, eps))
                stats["keff_absL2"].append(abs_l2(ke_pred, ke_gt))
                stats["keff_smape"].append(smape(ke_pred, ke_gt, eps=mape_eps))
                if enable_mape:
                    stats["keff_mape"].append(mape_stable(ke_pred, ke_gt, eps=mape_eps))
                k_sum += float(ke_gt.sum())
                k_sum_sq += float((ke_gt ** 2).sum())
                k_sse += float(((ke_pred - ke_gt) ** 2).sum())
                k_cnt += ke_gt.numel()
                # keff GT/Pred CSV
                keff_pred_for_csv.append(float(ke_pred.item()))
                keff_gt_for_csv.append(float(ke_gt.item()))

            # ===== PDE 残差可视化 + CSV 统计 =====
            if save_pde_vis:
                # Compute PDE residuals arrays for full field
                k_field = make_k_field_from_mask_np(
                    porous_np, k_solid, k_pore, bin_thresh
                )
                T_np = T_pred[0, 0].detach().cpu().numpy().astype(np.float32)
                qx_np = qx_pred[0, 0].detach().cpu().numpy().astype(np.float32)
                qy_np = qy_pred[0, 0].detach().cpu().numpy().astype(np.float32)

                # 利用 T 计算“保守”的 q，然后和网络输出做对比
                qx_cons, qy_cons = conservative_q_from_T_np(
                    T_np, k_field, pde_dx, pde_dy
                )
                rFx = qx_np - qx_cons
                rFy = qy_np - qy_cons
                rF_mag = np.sqrt(rFx * rFx + rFy * rFy)
                rDiv = conservative_div_from_center_np(
                    qx_np.astype(np.float32),
                    qy_np.astype(np.float32),
                    pde_dx,
                    pde_dy,
                )
                scale_sqrt = math.sqrt(0.5 * (pde_dx * pde_dx + pde_dy * pde_dy))
                rDiv_vis = rDiv * scale_sqrt

                # Determine color scale limits via percentile clipping
                def _clip_res(a):
                    vmax = float(np.percentile(np.abs(a), pde_clip_pct))
                    if vmax <= 0:
                        vmax = float(np.abs(a).max()) + 1e-12
                    return vmax

                vmax_rFx = _clip_res(rFx)
                vmax_rFy = _clip_res(rFy)
                vmax_rDiv = _clip_res(rDiv_vis)
                vmax_rF = _clip_res(rF_mag)

                # Save residual maps for this sample
                plt.figure(figsize=(4, 4), dpi=dpi)
                imA = plt.imshow(
                    rFx,
                    cmap=cmap,
                    origin=origin,
                    vmin=-vmax_rFx,
                    vmax=vmax_rFx,
                )
                plt.axis("off")
                plt.colorbar(imA, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(pde_rFx_dir, f"{idx_global:06d}.png"),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.close()

                plt.figure(figsize=(4, 4), dpi=dpi)
                imB = plt.imshow(
                    rFy,
                    cmap=cmap,
                    origin=origin,
                    vmin=-vmax_rFy,
                    vmax=vmax_rFy,
                )
                plt.axis("off")
                plt.colorbar(imB, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(pde_rFy_dir, f"{idx_global:06d}.png"),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.close()

                plt.figure(figsize=(4, 4), dpi=dpi)
                imC = plt.imshow(
                    rDiv_vis,
                    cmap=cmap,
                    origin=origin,
                    vmin=-vmax_rDiv,
                    vmax=vmax_rDiv,
                )
                plt.axis("off")
                plt.colorbar(imC, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(pde_rDiv_dir, f"{idx_global:06d}.png"),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.close()

                plt.figure(figsize=(4, 4), dpi=dpi)
                imD = plt.imshow(
                    rF_mag,
                    cmap=cmap,
                    origin=origin,
                    vmin=0.0,
                    vmax=vmax_rF,
                )
                plt.axis("off")
                plt.colorbar(imD, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(pde_rFmag_dir, f"{idx_global:06d}.png"),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.close()

                # ===== 这里是写 CSV 的部分：用 RMS（均方根），不是简单 L2 累加 =====
                rms_rFx   = _rms_np(rFx)      # x 列：rFx 的 RMS
                rms_rFy   = _rms_np(rFy)      # y 列：rFy 的 RMS
                rms_rFmag = _rms_np(rF_mag)   # sqrt 列：|rF| 的 RMS
                rms_rDiv  = _rms_np(rDiv_vis)     # div 列：div(q) 的 RMS（未乘 scale_sqrt）

                pde_csv_rows.append(
                    (
                        f"{idx_global:06d}",
                        f"{rms_rFx:.6e}",
                        f"{rms_rFy:.6e}",
                        f"{rms_rFmag:.6e}",
                        f"{rms_rDiv:.6e}",
                    )
                )


            idx_global += 1

    # ===== 汇总输出 =====
    metrics = {}
    if has_gt_any:
        avg = {
            k: (float(np.mean(v)) if len(v) > 0 else float("nan"))
            for k, v in stats.items()
        }
        metrics.update(avg)

        # R²
        if T_cnt > 0:
            T_mean = T_sum / T_cnt
            T_sst = T_sum_sq - T_cnt * (T_mean ** 2)
            metrics["T_R2"] = 1.0 - (T_sse / (T_sst + eps))
        if qx_cnt > 0:
            qx_mean = qx_sum / qx_cnt
            qx_sst = qx_sum_sq - qx_cnt * (qx_mean ** 2)
            metrics["qx_R2"] = 1.0 - (qx_sse / (qx_sst + eps))
        if qy_cnt > 0:
            qy_mean = qy_sum / qy_cnt
            qy_sst = qy_sum_sq - qy_cnt * (qy_mean ** 2)
            metrics["qy_R2"] = 1.0 - (qy_sse / (qy_sst + eps))
        if k_cnt > 0:
            k_mean = k_sum / k_cnt
            k_sst = k_sum_sq - k_cnt * (k_mean ** 2)
            metrics["keff_R2"] = 1.0 - (k_sse / (k_sst + eps))

        # 散点图
        if len(keff_gt_all) > 0:
            scatter_keff(
                keff_pred_all,
                keff_gt_all,
                os.path.join(out_dir, "scatter_keff.png"),
                lo=keff_lo,
                hi=keff_hi,
            )
            # keff 测试集 GT/Pred CSV
            keff_csv = os.path.join(out_dir, "keff_gt_pred.csv")
            with open(keff_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["GT", "Pred"])
                for g, p in zip(keff_gt_for_csv, keff_pred_for_csv):
                    w.writerow([f"{g:.6e}", f"{p:.6e}"])

        if len(T_pred_imgs) > 0:
            scatter_field_pixels(
                T_pred_imgs,
                T_gt_imgs,
                "T",
                os.path.join(out_dir, "scatter_T_pixels.png"),
            )
        if len(qx_pred_imgs) > 0 and len(qx_gt_imgs) > 0:
            scatter_field_pixels(
                qx_pred_imgs,
                qx_gt_imgs,
                "qx",
                os.path.join(out_dir, "scatter_qx_pixels.png"),
            )
        if len(qy_pred_imgs) > 0 and len(qy_gt_imgs) > 0:
            scatter_field_pixels(
                qy_pred_imgs,
                qy_gt_imgs,
                "qy",
                os.path.join(out_dir, "scatter_qy_pixels.png"),
            )

        # 像素散点 CSV（最多 20 万点）
        if len(T_pred_all) > 0:
            save_scatter_csv(
                T_pred_all,
                T_gt_all,
                os.path.join(out_dir, "scatter_T_pixels.csv"),
                max_points=200_000,
            )
        if len(qx_pred_all) > 0:
            save_scatter_csv(
                qx_pred_all,
                qx_gt_all,
                os.path.join(out_dir, "scatter_qx_pixels.csv"),
                max_points=200_000,
            )
        if len(qy_pred_all) > 0:
            save_scatter_csv(
                qy_pred_all,
                qy_gt_all,
                os.path.join(out_dir, "scatter_qy_pixels.csv"),
                max_points=200_000,
            )

        # 相对L2分布箱型图
        save_relL2_boxplot(
            stats,
            os.path.join(out_dir, "error_distribution_boxplot.png"),
        )
            # 每张图的相对 L2（散点 + CSV）
        save_relL2_per_image(
            stats,
            out_png=os.path.join(out_dir, "relL2_per_image.png"),
            out_csv=os.path.join(out_dir, "relL2_per_image.csv"),
        )

        # 误差频谱
        if save_spectrum:
            spec_dir = os.path.join(out_dir, "spectra")
            ensure_dir(spec_dir)
            if len(err_T_all) > 0:
                save_error_spectra_bundle(
                    err_T_all,
                    T_gt_all,
                    spec_root=spec_dir,
                    field_name="T",
                    max_to_save=spectrum_max_to_save,
                    dpi=dpi,
                    cmap="magma",
                )
            if len(err_qx_all) > 0:
                save_error_spectra_bundle(
                    err_qx_all,
                    qx_gt_all,
                    spec_root=spec_dir,
                    field_name="qx",
                    max_to_save=spectrum_max_to_save,
                    dpi=dpi,
                    cmap="magma",
                )
            if len(err_qy_all) > 0:
                save_error_spectra_bundle(
                    err_qy_all,
                    qy_gt_all,
                    spec_root=spec_dir,
                    field_name="qy",
                    max_to_save=spectrum_max_to_save,
                    dpi=dpi,
                    cmap="magma",
                )

        # 校准曲线
        if save_calib and len(keff_gt_all) > 0:
            ensure_dir(os.path.join(out_dir, "calib"))
            save_keff_calibration(
                keff_pred_all,
                keff_gt_all,
                os.path.join(out_dir, "calib", "keff_calibration.png"),
                bins=calib_bins,
            )

        # 指标 CSV（包含 *_absL2_norm、*_deltaS_L1、*_deltaR_L1）
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k in sorted(metrics.keys()):
                w.writerow([k, f"{metrics[k]:.6e}"])
        print(f"[{cfg_tag}] metrics saved to {os.path.join(out_dir, 'metrics.csv')}")

    else:
        # 无 GT：至少把 keff 预测落盘
        pred_path = os.path.join(out_dir, "keff_pred_only.csv")
        with open(pred_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "keff_pred"])
            for i, v in enumerate(keff_pred_all):
                w.writerow([i, f"{v:.6f}"])
        plt.figure(figsize=(4, 3), dpi=300)
        plt.hist(keff_pred_all, bins=20, alpha=0.85)
        plt.xlabel("Predicted keff")
        plt.ylabel("Count")
        plt.title("keff histogram (no GT)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "keff_pred_hist.png"))
        plt.close()
        print(f"[{cfg_tag}] no GT — predictions saved to {out_dir}")

    if porosity and len(porous_all) > 0:
        save_porosity_stats(
            porous_all,
            os.path.join(out_dir, "porosity"),
            bins=porosity_bins,
        )

    # 写出 PDE 残差汇总 CSV：index, x, y, sqrt, div（全部是 RMS）
    with open(os.path.join(out_dir, "pde_residuals_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "sqrt", "div"])
        for row in pde_csv_rows:
            w.writerow(row)

    print(
        f"[{cfg_tag}] PDE residuals summary saved to "
        f"{os.path.join(out_dir, 'pde_residuals_summary.csv')}"
    )

    return metrics, out_dir

# ============== 校准曲线（keff） ==============
def save_keff_calibration(keff_pred, keff_gt, out_png, bins=12):
    ensure_dir(os.path.dirname(out_png))
    keff_pred = np.asarray(keff_pred, float)
    keff_gt = np.asarray(keff_gt, float)
    if keff_pred.size == 0:
        return
    lo, hi = keff_pred.min(), keff_pred.max()
    if lo == hi:
        lo, hi = lo - 1e-3, hi + 1e-3
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    avg_gt, avg_pred = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (keff_pred >= a) & (keff_pred < b)
        if not np.any(mask):
            avg_gt.append(np.nan)
            avg_pred.append(np.nan)
        else:
            avg_gt.append(float(np.mean(keff_gt[mask])))
            avg_pred.append(float(np.mean(keff_pred[mask])))
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    plt.plot(centers, avg_gt, "o-", label="E[gt | pred-bin]")
    plt.xlabel("Predicted keff (bin center)")
    plt.ylabel("Average GT keff")
    plt.title("Calibration curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

@torch.no_grad()
def time_model(model, loader, device, tag, max_batches=20, warmup=5):
    model.eval()
    it = iter(loader)
    for _ in range(max(warmup, 0)):
        try:
            x, _ = _batch_to_targets(next(it))
        except StopIteration:
            it = iter(loader)
            x, _ = _batch_to_targets(next(it))
        x = _to(x, device)
        _ = model(x)
    n_imgs, t_sum = 0, 0.0
    it = iter(loader)
    for _ in range(max_batches):
        try:
            x, _ = _batch_to_targets(next(it))
        except StopIteration:
            break
        x = _to(x, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_sum += (time.perf_counter() - t0)
        n_imgs += x.size(0)
    ms_per_img = (t_sum / max(1, n_imgs)) * 1000.0
    print(
        f"[{tag}] Inference time: {ms_per_img:.2f} ms / image  "
        f"(batch={loader.batch_size})"
    )
    return ms_per_img

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser("Unified test & visualization")
    p.add_argument("--h5", type=str, default=DEFAULTS["h5_path"])
    p.add_argument("--ckpt", type=str, default=DEFAULTS["ckpt"])
    p.add_argument("--out", type=str, default=DEFAULTS["save_root"])
    p.add_argument("--subname", type=str, default=DEFAULTS["subname"])

    p.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["model"],
        choices=["fno", "ufno", "denseed"],
    )
    p.add_argument(
        "--grad",
        type=str,
        default=DEFAULTS["grad"],
        choices=["fd", "spec"],
    )   # 仅用于 tag
    p.add_argument("--tag", type=str, default=DEFAULTS["tag"])

    p.add_argument("--device", type=str, default=DEFAULTS["device"])
    p.add_argument("--bs", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--n_vis", type=int, default=DEFAULTS["n_vis"])

    # 模型
    p.add_argument("--in_ch", type=int, default=DEFAULTS["in_channels"])
    p.add_argument("--width", type=int, default=DEFAULTS["width"])
    p.add_argument("--modes1", type=int, default=DEFAULTS["modes1"])
    p.add_argument("--modes2", type=int, default=DEFAULTS["modes2"])
    p.add_argument("--layers", type=int, default=DEFAULTS["n_layers"])

    # viz
    p.add_argument("--cmap", type=str, default=DEFAULTS["cmap"])
    p.add_argument("--err_cmap", type=str, default=DEFAULTS["err_cmap"])
    p.add_argument("--cmap_porous", type=str, default=DEFAULTS["cmap_porous"])
    p.add_argument("--origin", type=str, default=DEFAULTS["origin"])
    p.add_argument("--dpi", type=int, default=DEFAULTS["dpi"])

    # ranges
    p.add_argument("--T_min", type=float, default=DEFAULTS["T_min"])
    p.add_argument("--T_max", type=float, default=DEFAULTS["T_max"])
    p.add_argument("--Q_abs", type=float, default=DEFAULTS["Q_abs"])

    # transpose for GT
    p.add_argument("--no-transpose_hw", action="store_true")

    # keff axis
    p.add_argument("--keff_min", type=float, default=DEFAULTS["keff_min"])
    p.add_argument("--keff_max", type=float, default=DEFAULTS["keff_max"])

    # metrics
    p.add_argument("--enable_mape", action="store_true")
    p.add_argument("--mape_eps", type=float, default=DEFAULTS["mape_eps"])

    # PDE residual
    p.add_argument("--k_solid", type=float, default=DEFAULTS["k_solid"])
    p.add_argument("--k_pore", type=float, default=DEFAULTS["k_pore"])
    p.add_argument("--bin_thresh", type=float, default=DEFAULTS["bin_thresh"])
    p.add_argument("--pde_dx", type=float, default=DEFAULTS["pde_dx"])
    p.add_argument("--pde_dy", type=float, default=DEFAULTS["pde_dy"])
    p.add_argument("--pde_clip_pct", type=float, default=DEFAULTS["pde_clip_pct"])
    p.add_argument("--no_pde_vis", action="store_true")

    # spectra & calibration
    p.add_argument("--no_spectrum", action="store_true")
    p.add_argument("--spectrum_max", type=int, default=DEFAULTS["spectrum_max_to_save"])
    p.add_argument("--no_spectrum_avg", action="store_true")
    p.add_argument("--no_spectrum_radial", action="store_true")
    p.add_argument("--no_spectrum_radial_csv", action="store_true")
    p.add_argument("--no_calib", action="store_true")
    p.add_argument("--calib_bins", type=int, default=DEFAULTS["calib_bins"])

    # porosity
    p.add_argument("--porosity", action="store_true", default=True)
    p.add_argument("--porosity_bins", type=int, default=DEFAULTS["porosity_bins"])
    return p.parse_args()

def main():
    args = parse_args()
    # 默认 tag
    tag = args.tag or f"{args.grad}_{args.model}_test"

    test_loader = build_test_loader(args.h5, args.bs, args.workers)
    model = load_model_via_factory(
        model_name=args.model,
        device=args.device,
        in_channels=args.in_ch,
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        n_layers=args.layers,
        ckpt_path=args.ckpt,
    )

    metrics, out_dir = evaluate_model(
        model,
        test_loader,
        args.device,
        out_root=args.out,
        subname=args.subname,
        cfg_tag=tag,
        cmap=args.cmap,
        err_cmap=args.err_cmap,
        cmap_porous=args.cmap_porous,
        origin=args.origin,
        dpi=args.dpi,
        n_vis=args.n_vis,
        T_min=args.T_min,
        T_max=args.T_max,
        Q_abs=args.Q_abs,
        transpose_hw=(not args.no_transpose_hw),
        keff_lo=args.keff_min,
        keff_hi=args.keff_max,
        enable_mape=args.enable_mape,
        mape_eps=args.mape_eps,
        eps=DEFAULTS["eps"],
        k_solid=args.k_solid,
        k_pore=args.k_pore,
        bin_thresh=args.bin_thresh,
        pde_dx=args.pde_dx,
        pde_dy=args.pde_dy,
        save_pde_vis=(not args.no_pde_vis),
        pde_clip_pct=args.pde_clip_pct,
        save_spectrum=(not args.no_spectrum),
        spectrum_max_to_save=args.spectrum_max,
        spectrum_avg=(not args.no_spectrum_avg),
        spectrum_radial=(not args.no_spectrum_radial),
        spectrum_radial_csv=(not args.no_spectrum_radial_csv),
        save_calib=(not args.no_calib),
        calib_bins=args.calib_bins,
        # 关键：这里直接强制开启孔隙度统计
        porosity=True,
        porosity_bins=args.porosity_bins,
    )

    _ = time_model(
        model,
        test_loader,
        args.device,
        tag=tag,
        max_batches=20,
        warmup=5,
    )

    if metrics:
        root_csv_dir = os.path.join(args.out, args.subname)
        ensure_dir(root_csv_dir)
        root_csv = os.path.join(root_csv_dir, f"{tag}_metrics_root.csv")
        with open(root_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k in sorted(metrics.keys()):
                w.writerow([k, f"{metrics[k]:.6e}"])
        print(f"[done] metrics root CSV saved to {root_csv}")

if __name__ == "__main__":
    main()
