# pino_norm.py — stable unsupervised edition (fd / conservative / dct + baseline/divT/Tonly)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= Grad backends =======================
class FDGrad2D(nn.Module):
    def __init__(self, dx: float = 1.0, dy: float = 1.0):
        super().__init__()
        kx = torch.tensor([[0., 0., 0.],
                           [-0.5/dx, 0., 0.5/dx],
                           [0., 0.,  0.]], dtype=torch.float32)
        ky = torch.tensor([[0., -0.5/dy, 0.],
                           [0.,   0.,   0.],
                           [0.,  0.5/dy, 0.]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)

    def _kern(self, proto: torch.Tensor):
        return (self.kx.to(proto), self.ky.to(proto))

    def grad(self, U):
        C = U.shape[1]
        kx, ky = self._kern(U)
        U_pad = F.pad(U, (1, 1, 1, 1), mode="replicate")
        Ux = F.conv2d(U_pad, kx.repeat(C, 1, 1, 1), groups=C)
        Uy = F.conv2d(U_pad, ky.repeat(C, 1, 1, 1), groups=C)
        return Ux, Uy

    def div_parts(self, Qx, Qy):
        Cx, Cy = Qx.shape[1], Qy.shape[1]
        kx, ky = self._kern(Qx)
        Qx_pad = F.pad(Qx, (1, 1, 1, 1), mode="replicate")
        Qy_pad = F.pad(Qy, (1, 1, 1, 1), mode="replicate")
        dQx = F.conv2d(Qx_pad, kx.repeat(Cx, 1, 1, 1), groups=Cx)
        dQy = F.conv2d(Qy_pad, ky.repeat(Cy, 1, 1, 1), groups=Cy)
        return dQx, dQy


class DCTLikeGrad2D(nn.Module):
    """
    频域镜像 + DCT-like：严格匹配左右 Dirichlet（odd）与上下 Neumann
    """
    def __init__(self, bc_x: str = "dirichlet", bc_y: str = "neumann", antialias: bool = False):
        super().__init__()
        assert bc_x in ("dirichlet","neumann")
        assert bc_y in ("dirichlet","neumann")
        self.bc_x, self.bc_y = bc_x, bc_y
        self.antialias = antialias

    @staticmethod
    def _mirror_1d(U, axis: int, kind: str):
        tail = U[..., :-1].flip(-1 if axis == -1 else -2)
        if kind == "odd": tail = -tail
        return torch.cat([U, tail], dim=axis)

    @staticmethod
    def _freqs(Hext, Wext, dx, dy, device, dtype):
        fx = torch.fft.rfftfreq(Wext, d=dx, device=device).to(dtype)
        fy = torch.fft.fftfreq(Hext,  d=dy, device=device).to(dtype)
        kx = (2.0*math.pi) * fx.view(1,1,1,-1)
        ky = (2.0*math.pi) * fy.view(1,1,-1,1)
        return kx, ky

    def _maybe_antialias(self, Fhat):
        if not self.antialias: return Fhat
        Ny, Nx_r = Fhat.shape[-2], Fhat.shape[-1]
        cut_x = (Nx_r-1)//3
        cut_y = Ny//3
        mask_x = torch.ones(Nx_r, dtype=torch.bool, device=Fhat.device)
        if cut_x+1 < Nx_r: mask_x[cut_x+1:] = False
        arange_y = torch.arange(Ny, device=Fhat.device)
        half_y = Ny//2
        freq_y = torch.where(arange_y <= half_y, arange_y, Ny - arange_y)
        mask_y = (freq_y <= cut_y)
        return Fhat * (mask_y.view(Ny,1) & mask_x.view(1,Nx_r)).to(Fhat.dtype)

    def grad_phys(self, T, dx, dy):
        B,C,H,W = T.shape
        dtype, dev = T.dtype, T.device
        T_extx = self._mirror_1d(T, axis=-1, kind=("odd" if self.bc_x=="dirichlet" else "even"))
        T_ext  = self._mirror_1d(T_extx, axis=-2, kind=("odd" if self.bc_y=="dirichlet" else "even"))
        Hext, Wext = T_ext.shape[-2], T_ext.shape[-1]
        That = torch.fft.rfft2(T_ext, dim=(-2,-1))
        That = self._maybe_antialias(That)
        kx, ky = self._freqs(Hext, Wext, dx, dy, dev, dtype)
        i = torch.complex(torch.zeros((), device=dev, dtype=dtype), torch.ones((), device=dev, dtype=dtype))
        Tx_hat = i * kx * That;  Ty_hat = i * ky * That
        Tx_ext = torch.fft.irfft2(Tx_hat, s=(Hext,Wext), dim=(-2,-1))
        Ty_ext = torch.fft.irfft2(Ty_hat, s=(Hext,Wext), dim=(-2,-1))
        return Tx_ext[..., :H, :W], Ty_ext[..., :H, :W]

    def div_phys(self, Qx, Qy, dx, dy):
        B,C,H,W = Qx.shape
        dtype, dev = Qx.dtype, Qx.device
        # Qx: even-x, (y: even if Neumann else odd) ; Qy: odd-x if Dirichlet else even-x, (y: odd if Neumann else even)
        kind_x_qx = "even"
        kind_y_qx = "even" if self.bc_y=="neumann" else "odd"
        kind_x_qy = "odd"  if self.bc_x=="dirichlet" else "even"
        kind_y_qy = "odd"  if self.bc_y=="neumann"  else "even"
        Qx_ex = self._mirror_1d(Qx, axis=-1, kind=kind_x_qx); Qx_ex = self._mirror_1d(Qx_ex, axis=-2, kind=kind_y_qx)
        Qy_ex = self._mirror_1d(Qy, axis=-1, kind=kind_x_qy); Qy_ex = self._mirror_1d(Qy_ex, axis=-2, kind=kind_y_qy)
        Hext, Wext = Qx_ex.shape[-2], Qx_ex.shape[-1]
        Qx_hat = torch.fft.rfft2(Qx_ex, dim=(-2,-1))
        Qy_hat = torch.fft.rfft2(Qy_ex, dim=(-2,-1))
        Qx_hat = self._maybe_antialias(Qx_hat)
        Qy_hat = self._maybe_antialias(Qy_hat)
        kx, ky = self._freqs(Hext, Wext, dx, dy, dev, dtype)
        i = torch.complex(torch.zeros((), device=dev, dtype=dtype), torch.ones((), device=dev, dtype=dtype))
        Div_hat = i * kx * Qx_hat + i * ky * Qy_hat
        Div_ex = torch.fft.irfft2(Div_hat, s=(Hext,Wext), dim=(-2,-1))
        return Div_ex[..., :H, :W]


# ======================= Phys helpers =======================
def phys_from_cfg(CFG_PDE: dict, H: int, W: int):
    # 固定步长/尺度，不再用相对归一化
    dx = float(CFG_PDE.get("dx", 1.0))
    dy = float(CFG_PDE.get("dy", 1.0))
    k_solid = float(CFG_PDE.get("k_solid", 1.0))
    k_pore  = float(CFG_PDE.get("k_pore",  0.2))
    T_left  = float(CFG_PDE.get("T_left",  1.0))
    T_right = float(CFG_PDE.get("T_right", 0.0))
    T_scale = float(CFG_PDE.get("T_scale", 1.0))
    q_scale = float(CFG_PDE.get("q_scale", 1.0))
    return dx, dy, k_solid, k_pore, T_left, T_right, T_scale, q_scale


def make_k_field_from_mask(porous_mask: torch.Tensor, k_solid: float, k_pore: float):
    """
    固定约定：黑=孔隙=0 → k_pore；白=基体=1 → k_solid
    """
    if porous_mask.dim() == 2:
        porous_mask = porous_mask.unsqueeze(0).unsqueeze(0)
    elif porous_mask.dim() == 3:
        porous_mask = porous_mask.unsqueeze(1)
    v = porous_mask.float()
    is_pore = (v < 0.5)              # 黑色<0.5 → 孔隙
    k_field = is_pore.to(v.dtype) * k_pore + (~is_pore).to(v.dtype) * k_solid
    return k_field


def apply_hard_bc_T(T: torch.Tensor, T_left: float, T_right: float):
    T_bc = T.clone()
    T_bc[...,  :,  0] = T_left
    T_bc[...,  :, -1] = T_right
    return T_bc


# ======================= Conservative ops =======================
def conservative_grad_div(T: torch.Tensor, k_field: torch.Tensor, dx: float, dy: float):
    """
    面通量保守型：中心差分在面上做调和平均 k → 面通量 → 中心散度
    """
    eps_k = 1e-12
    # x faces
    T_xf = (T[...,  :, 1:] - T[...,  :, :-1]) / dx
    k_l, k_r = k_field[..., :, :-1], k_field[..., :, 1:]
    den_x = (k_l + k_r).clamp_min(eps_k)
    k_xf = (2.0 * k_l * k_r / den_x).clamp_min(eps_k)
    qx_face = -k_xf * T_xf
    # y faces
    T_yf = (T[..., 1:, :] - T[..., :-1, :]) / dy
    k_b, k_t = k_field[..., :-1, :], k_field[..., 1:, :]
    den_y = (k_b + k_t).clamp_min(eps_k)
    k_yf = (2.0 * k_b * k_t / den_y).clamp_min(eps_k)
    qy_face = -k_yf * T_yf
    # face -> center
    qx_center = 0.5 * (F.pad(qx_face, (1,0,0,0), mode="replicate") +
                       F.pad(qx_face, (0,1,0,0), mode="replicate"))
    qy_center = 0.5 * (F.pad(qy_face, (0,0,1,0), mode="replicate") +
                       F.pad(qy_face, (0,0,0,1), mode="replicate"))
    # conservative divergence
    div_x_core = (qx_face[...,  :, 1:] - qx_face[...,  :, :-1]) / dx
    div_x = F.pad(div_x_core, (1,1,0,0), mode="replicate")
    div_y_core = (qy_face[..., 1:, :] - qy_face[..., :-1, :]) / dy
    div_y = F.pad(div_y_core, (0,0,1,1), mode="replicate")
    divergence = div_x + div_y
    return qx_center, qy_center, divergence


def conservative_div_from_center(qx_center: torch.Tensor, qy_center: torch.Tensor,
                                 dx: float, dy: float) -> torch.Tensor:
    qx_face = 0.5 * (qx_center[..., :, 1:] + qx_center[..., :, :-1])
    qy_face = 0.5 * (qy_center[..., 1:, :] + qy_center[..., :-1, :])
    div_x_core = (qx_face[..., :, 1:] - qx_face[..., :, :-1]) / dx
    div_x = F.pad(div_x_core, (1,1,0,0), mode="replicate")
    div_y_core = (qy_face[..., 1:, :] - qy_face[..., :-1, :]) / dy
    div_y = F.pad(div_y_core, (0,0,1,1), mode="replicate")
    return div_x + div_y


def conservative_Ty_center(T: torch.Tensor, dy: float) -> torch.Tensor:
    T_yf = (T[..., 1:, :] - T[..., :-1, :]) / dy
    return 0.5 * (F.pad(T_yf, (0,0,1,0), mode="replicate") +
                  F.pad(T_yf, (0,0,0,1), mode="replicate"))

@torch.no_grad()
def _edge_mse_tb(U_head: torch.Tensor, U_cons: torch.Tensor, n: int = 1) -> torch.Tensor:
    """top/bottom 各 n 像素带上的 MSE（不回传到 T）"""
    if U_head is None or U_cons is None or n <= 0:
        return U_head.new_zeros(()) if isinstance(U_head, torch.Tensor) else torch.tensor(0.0)
    cs_top    = (slice(None), slice(None), slice(0, n),      slice(None))
    cs_bottom = (slice(None), slice(None), slice(-n, None),  slice(None))
    return 0.5 * (F.mse_loss(U_head[cs_top], U_cons[cs_top]) +
                  F.mse_loss(U_head[cs_bottom], U_cons[cs_bottom]))

@torch.no_grad()
def _edge_mse_lr(U_head: torch.Tensor, U_cons: torch.Tensor, n: int = 1) -> torch.Tensor:
    """left/right 各 n 像素带上的 MSE（不回传到 T）"""
    if U_head is None or U_cons is None or n <= 0:
        return U_head.new_zeros(()) if isinstance(U_head, torch.Tensor) else torch.tensor(0.0)
    cs_left  = (slice(None), slice(None), slice(None), slice(0, n))
    cs_right = (slice(None), slice(None), slice(None), slice(-n, None))
    return 0.5 * (F.mse_loss(U_head[cs_left],  U_cons[cs_left]) +
                  F.mse_loss(U_head[cs_right], U_cons[cs_right]))


# ======================= Core loss =======================
def _heat_loss_core(
    porous_mask, T_pred, qx_pred, qy_pred,
    *, dx_phys, dy_phys, T_scale, q_scale,
    k_solid, k_pore, T_left, T_right,
    adiabatic_mode, grad_backend, q_mode,
    y_true, keff_head, keff_label,
    w_four, w_div, w_bc, w_T, w_qx, w_qy,
    w_keff_sup, w_keff_self,
    hard_bc: bool,
    antialias: bool = False,
    interior_band: int = 2,
):
    """
    模式：
      - baseline:  L_div 用 div(q_head)=0；Fourier 用 (q_head - q_cons(T))=0
      - divT:      L_div 用 div(q_cons(T))=0；Fourier 仍 (q_head - q_cons(T))=0
      - Tonly:     只输出 T；L_div 用 div(q_cons(T))=0；无 Fourier
    """
    B, _, H, W = T_pred.shape
    device, dtype = T_pred.device, T_pred.dtype
    q_mode = str(q_mode).lower()
    has_q_head = (qx_pred is not None) and (qy_pred is not None)
    if q_mode == "tonly":
        has_q_head = False

    # 物理量域
    T_phys = T_pred * T_scale
    if has_q_head:
        qx_phys = qx_pred * q_scale
        qy_phys = qy_pred * q_scale
    else:
        qx_phys = qy_phys = None

    # --- pino1 风格的“硬边界”：左右 Dirichlet；上下绝热按 qy0 或 Ty0 二选一 ---
    if hard_bc:
        T_phys_bc  = T_phys.clone()
        qx_phys_bc = qx_phys if has_q_head else None
        qy_phys_bc = qy_phys.clone() if has_q_head else None

        # 左右 Dirichlet（直接钉值）
        T_phys_bc[:, :, :, 0]  = T_left
        T_phys_bc[:, :, :, -1] = T_right

        # 上下绝热
        if str(adiabatic_mode).lower() == "qy0":
            if has_q_head:
                qy_phys_bc[:, :, 0, :]  = 0.0
                qy_phys_bc[:, :, -1, :] = 0.0
            # Ty 不强钉，后面用差分/一致化在损失里约束
        else:  # "Ty0"
            # 复制内层一行到边界，等价零法向导数
            T_phys_bc[:, :, 0, :]  = T_phys_bc[:, :, 1, :]
            T_phys_bc[:, :, -1, :] = T_phys_bc[:, :, -2, :]
    else:
        T_phys_bc  = T_phys
        qx_phys_bc = qx_phys
        qy_phys_bc = qy_phys

    # k(x,y)
    k_field = make_k_field_from_mask(porous_mask, k_solid, k_pore).to(dtype=T_pred.dtype, device=T_pred.device)

    # ===== 梯度/散度后端 =====
    gb = str(grad_backend).lower()
    if gb in ("cons", "conservative"):
        # 从 T_phys_bc 计算一致化通量和散度；并给边界损失准备 Ty 的一致化版本
        q_cons_x, q_cons_y, div_cons = conservative_grad_div(T_phys_bc, k_field, dx_phys, dy_phys)
        Ty_center = conservative_Ty_center(T_phys_bc, dy_phys)
        div_q_head = conservative_div_from_center(qx_phys_bc, qy_phys_bc, dx_phys, dy_phys) if has_q_head else None
    elif gb in ("spec_dct", "dct"):
        sg = DCTLikeGrad2D(
            bc_x="dirichlet",
            bc_y=("neumann" if str(adiabatic_mode).lower() in ("ty0","qy0") else "dirichlet"),
            antialias=antialias
        )
        Tx, Ty = sg.grad_phys(T_phys_bc, dx_phys, dy_phys)
        q_cons_x, q_cons_y = -k_field * Tx, -k_field * Ty
        div_cons  = sg.div_phys(q_cons_x, q_cons_y, dx_phys, dy_phys)
        div_q_head = sg.div_phys(qx_phys_bc, qy_phys_bc, dx_phys, dy_phys) if has_q_head else None
        Ty_center = Ty
    else:  # 'fd'
        fd = FDGrad2D(dx_phys, dy_phys)
        Tx, Ty = fd.grad(T_phys_bc)
        q_cons_x, q_cons_y = -k_field * Tx, -k_field * Ty
        dQx_c, dQy_c = fd.div_parts(q_cons_x, q_cons_y)
        div_cons = dQx_c + dQy_c
        if has_q_head:
            dQx_h, dQy_h = fd.div_parts(qx_phys_bc, qy_phys_bc)
            div_q_head = dQx_h + dQy_h
        else:
            div_q_head = None
        Ty_center = Ty

    # ===== 不裁边（pino1 风格，全域参与损失）=====
    T_core       = T_phys_bc
    q_cons_x_c   = q_cons_x
    q_cons_y_c   = q_cons_y
    div_cons_c   = div_cons
    if has_q_head:
        qx_head_c = qx_phys_bc
        qy_head_c = qy_phys_bc
        div_head_c = div_q_head
    else:
        qx_head_c = qy_head_c = div_head_c = None
    edge_trim = 0

    # ===== Fourier: (q_head - q_cons(T)) → 0（只有有 head 时参与）=====
    if has_q_head:
        L_four = F.mse_loss(qx_head_c - q_cons_x_c, torch.zeros_like(q_cons_x_c)) + \
                 F.mse_loss(qy_head_c - q_cons_y_c, torch.zeros_like(q_cons_y_c))
    else:
        L_four = T_core.new_zeros(())

    # ===== Divergence（保留 divT 你的创新；否则 baseline）=====
    scale_div = 0.5 * (dx_phys * dx_phys + dy_phys * dy_phys)
    if (q_mode == "divT") or (q_mode == "tonly") or (not has_q_head):
        L_div = scale_div * F.mse_loss(div_cons_c, torch.zeros_like(div_cons_c))
        res_div_raw = div_cons_c
    else:
        L_div = scale_div * (F.mse_loss(div_head_c, torch.zeros_like(div_head_c))
                             if (div_head_c is not None) else T_core.new_zeros(()))
        res_div_raw = div_head_c if (div_head_c is not None) else div_cons_c
    # if (q_mode == "divT") or (q_mode == "tonly") or (not has_q_head):
    #     L_div = F.mse_loss(div_cons_c, torch.zeros_like(div_cons_c))
    #     res_div_raw = div_cons_c
    # else:
    #     L_div = (F.mse_loss(div_head_c, torch.zeros_like(div_head_c))
    #              if (div_head_c is not None) else T_core.new_zeros(()))
    #     res_div_raw = div_head_c if (div_head_c is not None) else div_cons_c


    # ===== 边界项（与 hard_bc 相容：hard_bc=True 时二者应 ~0）=====
    # 左右 Dirichlet
    L_bc_T = F.mse_loss(T_phys[:, :, :, 0],  torch.full_like(T_phys[:, :, :, 0],  T_left)) + \
             F.mse_loss(T_phys[:, :, :, -1], torch.full_like(T_phys[:, :, :, -1], T_right))

    # 上下绝热：qy0 用 qy@边界；否则 Ty=0 用 Ty_center@边界
    if str(adiabatic_mode).lower() == "qy0":
        if has_q_head:
            top = qy_phys_bc[:, :, 0, :];  bot = qy_phys_bc[:, :, -1, :]
        else:
            top = q_cons_y[:, :, 0, :];    bot = q_cons_y[:, :, -1, :]
    else:
        top = Ty_center[:, :, 0, :];      bot = Ty_center[:, :, -1, :]
    L_bc_N = F.mse_loss(top, torch.zeros_like(top)) + F.mse_loss(bot, torch.zeros_like(bot))
    L_bc   = L_bc_T + L_bc_N

    # ===== keff_from_flux：用 q_head（避免与 divT 拉扯），内域均值抑制边界层 =====
    eps = 1e-12
    if has_q_head:
        qx_for_flux = qx_phys  # 用 head
    else:
        qx_for_flux = q_cons_x # Tonly 兜底
    qx_mean = qx_for_flux[:, :, 1:-1, 1:-1].mean(dim=(2, 3))  # (B,1)
    L_phys = (W - 1) * dx_phys
    dT = (T_left - T_right)
    keff_from_flux = (qx_mean * L_phys) / (dT + eps)

    L_keff_sup = L_keff_self = T_core.new_zeros(())
    if keff_head is not None:
        kh = keff_head.view(B)
        if keff_label is not None:
            lbl  = keff_label.view(B)
            mask = torch.isfinite(lbl).float()
            if mask.sum() > 0:
                L_keff_sup  = ((kh - lbl).pow(2) * mask).sum() / (mask.sum() + 1e-12)
            no = 1.0 - mask
            if no.sum() > 0:
                L_keff_self = ((kh - keff_from_flux.view(B))**2 * no).sum() / (no.sum() + 1e-12)
        else:
            L_keff_self = F.mse_loss(kh, keff_from_flux.view(B))

    # ===== 监督（如给）=====
    L_T = L_qx = L_qy = T_core.new_zeros(())
    if y_true is not None:
        if y_true.get("T",  None) is not None:  L_T  = F.mse_loss(T_phys,              y_true["T"])
        if has_q_head and y_true.get("qx", None) is not None:  L_qx = F.mse_loss(qx_pred * q_scale, y_true["qx"])
        if has_q_head and y_true.get("qy", None) is not None:  L_qy = F.mse_loss(qy_pred * q_scale, y_true["qy"])
    L_mse = L_T + L_qx + L_qy

    # ===== 总损失 =====
    L_keffC = w_keff_sup * L_keff_sup + w_keff_self * L_keff_self
    total = (w_four * L_four + w_div * L_div + w_bc * L_bc +
             w_T * L_T + w_qx * L_qx + w_qy * L_qy + L_keffC)

    # ===== 可视化残差（无量纲）=====
    scale_sqrt = math.sqrt(scale_div)
    res_div_vis = res_div_raw * scale_sqrt
    # res_div_vis = res_div_raw

    comps = dict(
        L_total=total.detach(),
        L_mse=L_mse.detach(),
        L_four=L_four.detach(),
        L_div=L_div.detach(),
        L_bc_T=L_bc_T.detach(), L_bc_N=L_bc_N.detach(),
        L_keffC=L_keffC.detach(),
        keff_from_flux=keff_from_flux.detach().view(B, 1),
        edge_trim=edge_trim,  # 0：不裁边
        res_div=res_div_vis.detach(),
        div_rms=res_div_vis.pow(2).mean().sqrt().detach(),
        div_meanabs=res_div_vis.abs().mean().detach(),
    )
    if has_q_head:
        comps["res_fourier_x"] = (qx_head_c - q_cons_x_c).detach()
        comps["res_fourier_y"] = (qy_head_c - q_cons_y_c).detach()
    if keff_head is not None:
        comps["keff_head"] = keff_head.detach().view(B, 1)

    return total, comps



# ======================= Public API =======================
def total_loss(x, y_img, y_keff, CFG_LOSS: dict, CFG_PDE: dict, y_gt: dict | None = None):
    """
    x:     (B,*,H,W) ; x[:,0:1] 为孔隙掩膜（黑=孔隙=0，白=基体=1）
    y_img: baseline/divT: (B,3,H,W) -> [T, qx, qy] ; Tonly: (B,1,H,W) -> [T]
    y_keff: (B,1) or None
    """
    porous  = x[:, 0:1]
    C = y_img.shape[1]
    T_pred  = y_img[:, 0:1]
    qx_pred = y_img[:, 1:2] if C >= 2 else None
    qy_pred = y_img[:, 2:3] if C >= 3 else None

    _, _, H, W = y_img.shape
    dx, dy, k_solid, k_pore, T_left, T_right, T_scale, q_scale = phys_from_cfg(CFG_PDE, H, W)

    adiabatic_mode = str(CFG_PDE.get("adiabatic_mode", "qy0"))
    grad_backend   = str(CFG_PDE.get("grad_backend", "conservative"))   # 'fd' | 'conservative' | 'spec_dct'
    q_mode         = str(CFG_PDE.get("q_mode", "baseline"))             # 'baseline' | 'divT' | 'Tonly'
    antialias      = bool(CFG_PDE.get("antialias", False))
    interior_band  = int(CFG_PDE.get("interior_band", 2))
    hard_bc        = bool(CFG_PDE.get("hard_bc", True))

    w_mse  = float(CFG_LOSS.get("w_mse", 0.0))
    w_bc   = float(CFG_LOSS.get("w_bc", 10.0))
    w_four = float(CFG_LOSS.get("w_four", 3.0))
    w_div  = float(CFG_LOSS.get("w_div", 3.0))
    w_keff = float(CFG_LOSS.get("w_keff", 0.0))
    w_T = w_qx = w_qy = w_mse
    w_keff_sup = w_keff_self = w_keff

    keff_label = None
    if y_gt is not None and ("keff" in y_gt) and (y_gt["keff"] is not None):
        keff_label = y_gt["keff"]

    return _heat_loss_core(
        porous, T_pred, qx_pred, qy_pred,
        dx_phys=dx, dy_phys=dy, T_scale=T_scale, q_scale=q_scale,
        k_solid=k_solid, k_pore=k_pore, T_left=T_left, T_right=T_right,
        adiabatic_mode=adiabatic_mode, grad_backend=grad_backend, q_mode=q_mode,
        y_true=y_gt, keff_head=y_keff, keff_label=keff_label,
        w_four=w_four, w_div=w_div, w_bc=w_bc,
        w_T=w_T, w_qx=w_qx, w_qy=w_qy,
        w_keff_sup=w_keff_sup, w_keff_self=w_keff_self,
        hard_bc=hard_bc, antialias=antialias, interior_band=interior_band
    )
