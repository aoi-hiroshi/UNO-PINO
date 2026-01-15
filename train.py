import os, time, math, csv, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataloader import make_splits, make_splits_from_h5
from models import build_model
from Losses.pino_norm import (
    total_loss,
    phys_from_cfg,
    make_k_field_from_mask,
    apply_hard_bc_T,
    conservative_grad_div,
)

# -------- utils --------
def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _next_run_dir(base_root: str, tag_prefix: str) -> str:
    _ensure_dir(base_root); existing=[]
    for name in os.listdir(base_root):
        if name.startswith(tag_prefix):
            suf = name[len(tag_prefix):]
            if suf.isdigit(): existing.append(int(suf))
    nxt = (max(existing)+1) if existing else 1
    return os.path.join(base_root, f"{tag_prefix}{nxt}")

def _to_dev(x, device):
    if isinstance(x, torch.Tensor): return x.to(device, non_blocking=True)
    if isinstance(x, dict):         return {k: _to_dev(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):return type(x)(_to_dev(v, device) for v in x)
    if isinstance(x, np.ndarray):   return torch.from_numpy(x).to(device, non_blocking=True)
    return x

def _tHW(t: torch.Tensor): return t.permute(0,1,3,2).contiguous()

# -------- settings --------
DATASET_TYPE = "supervised"         # "unsupervised" | "half_supervised" | "supervised" | "data_driven"
DATASET_PATHS = {
    "unsupervised":      "data/porous_bc_256.hdf5",
    "half_supervised":   "data/half_supervised_porous_bc_256.hdf5",
    "supervised":        "data/supervised_porous_bc_256.hdf5",
    "data_driven":       "data/supervised_porous_bc_256.hdf5",
}
H5_PATH = DATASET_PATHS[DATASET_TYPE]

MODEL_TYPE = "ufno"  #'denseed' 'fno' 'ufno'
GRAD_TYPE  = "cons" #'cons' 'dct' 'fd'

SEED = 42
EPOCHS = 600
BS_TRAIN = 16
BS_EVAL  = 16
USE_AMP = False
CLIP_GRAD_NORM = 1.0

IN_CHANNELS = 3
WIDTH  = 96
N_LAYERS = 9   # depth = 4 -> 3 downsamplings: 256->128->64->32
MODES1 = [48, 32, 24, 16]   # enc1, enc2, enc3, bottleneck
MODES2 = [48, 32, 24, 17]   # 32x32 处 m2 ≤ 17

# ---- PDE config (explicitly black=porous, white=matrix) ----
CFG_PDE = dict(
    k_solid=1.0, k_pore=0.2,
    dx=1.0/(256-1), dy=1.0/(256-1),
    grad_backend=GRAD_TYPE,
    adiabatic_mode="qy0",
    core_crop_px=1,
    interior_band=4,
    q_mode = "divT"
)

# ---- loss weights ----
if DATASET_TYPE == "unsupervised":
    CFG_LOSS = dict(w_mse=0.0, w_bc=10.0, w_four=3.0, w_div=3.0, w_keff=0.05)
elif DATASET_TYPE == "half_supervised":
    CFG_LOSS = dict(w_mse=1.0, w_bc=0.0, w_four=1.0, w_div=1.0, w_keff=0.1)
elif DATASET_TYPE == "data_driven":
    CFG_LOSS = dict(w_mse=1.0, w_bc=0.0, w_four=0.0, w_div=0.0, w_keff=0.2)
else:  # supervised
    CFG_LOSS = dict(w_mse=65536.0, w_bc=0.0, w_four=0.0, w_div=0.0, w_keff=0.1)

_SUPV_ABBR = {"unsupervised": "un", "half_supervised": "half", "supervised": "sup","data_driven": "dd"}[DATASET_TYPE]
TAG_PREFIX = f"{GRAD_TYPE.lower()}_{MODEL_TYPE.lower()}_{_SUPV_ABBR}_train"
RUN_DIR = _next_run_dir("results", TAG_PREFIX)
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
RESULT_IMG_DIR = os.path.join(RUN_DIR, "viz")
LOSS_DIR       = os.path.join(RUN_DIR, "loss")
LOG_CSV_PATH   = os.path.join(RUN_DIR, "train_log.csv")
for p in [RUN_DIR, CHECKPOINT_DIR, RESULT_IMG_DIR, LOSS_DIR]: _ensure_dir(p)

def _dump_run_config(run_dir: str, extra: dict):
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)

# ---- schedules (optional) ----
def _lerp(a: float, b: float, t: float) -> float: return a + (b - a) * t

def _interp_dict(d0: dict, d1: dict, t: float) -> dict:
    keys = set(d0.keys()) | set(d1.keys()); out={}
    for k in keys:
        a = float(d0.get(k, 0.0)); b = float(d1.get(k, 0.0))
        out[k] = _lerp(a, b, t)
    return out


LOSS_SCHEDULES = { #默认不调用
    "supervised":
    [
        (0,   dict(w_mse=65536.0, w_bc=1.0,  w_four=0.1, w_div=0.1, w_keff=0.1)),
        (150, dict(w_mse=65536,  w_bc=6.0,  w_four=2.0, w_div=2.0, w_keff=0.3)),
        (300, dict(w_mse=65536,  w_bc=10.0, w_four=4.0, w_div=4.0, w_keff=0.5)),
    ],
    "half_supervised": [
        (0,   dict(w_mse=10.0, w_bc=1.0,  w_four=0.1, w_div=0.1, w_keff=0.1)),
        (150, dict(w_mse=0.3,  w_bc=6.0,  w_four=2.0, w_div=2.0, w_keff=0.3)),
        (300, dict(w_mse=0.1,  w_bc=10.0, w_four=4.0, w_div=4.0, w_keff=0.5)),
    ],
    "unsupervised": [
        (0,   dict(w_mse=0.0, w_keff=0.05, w_bc=10.0, w_four=3.0, w_div=3.0)),
        (200, dict(w_mse=0.0, w_keff=0.20, w_bc=4.0,  w_four=6.0, w_div=2.0)),
        (300, dict(w_mse=0.0, w_keff=0.50, w_bc=8.0,  w_four=8.0, w_div=4.0)),
    ],
}

def _weights_by_epoch(epoch: int) -> dict:
    sched = LOSS_SCHEDULES[DATASET_TYPE]
    for i in range(len(sched) - 1):
        e0, d0 = sched[i]; e1, d1 = sched[i + 1]
        if e0 <= epoch < e1:
            t = (epoch - e0) / max(1, (e1 - e0))
            return _interp_dict(d0, d1, t)
    return dict(LOSS_SCHEDULES[DATASET_TYPE][-1][1])

def _apply_loss_schedule(CFG_LOSS: dict, epoch: int):
    CFG_LOSS.update({k: float(v) for k, v in _weights_by_epoch(epoch).items()})

# ---- residual maps (use comps directly) ----

def save_residual_maps(comps: dict, epoch: int, out_root: str):
    os.makedirs(out_root, exist_ok=True)

    def _as_tensor(x):
        return x if isinstance(x, torch.Tensor) else None

    res_qx  = _as_tensor(comps.get("res_fourier_x", None))
    res_qy  = _as_tensor(comps.get("res_fourier_y", None))
    res_div = _as_tensor(comps.get("res_div", None))

    def _save_heatmap(tensor, name):
        if tensor is None: return
        arr = tensor[0, 0].detach().cpu().numpy()
        vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4), dpi=150)
        plt.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin="upper")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"{name} (max|·|={vmax:.2e})")
        plt.axis("off")
        plt.tight_layout()
        fp = os.path.join(out_root, f"epoch_{epoch}_{name}.png")
        plt.savefig(fp, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    _save_heatmap(res_qx,  "res_qx")
    _save_heatmap(res_qy,  "res_qy")
    _save_heatmap(res_div, "res_div")

# ---- val visuals ----

def save_val_visuals(model, val_loader, epoch, device, out_root):
    import os, numpy as np, matplotlib.pyplot as plt

    # 目录
    for sub in ["T","qx","qy","keff","qx_consT","qy_consT","residual_maps"]:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    # 取一批验证样本
    batch = next(iter(val_loader))
    x, y_gt = _batch_to_targets(batch)
    x = x.to(device); y_gt = _to_dev(y_gt, device) if y_gt is not None else None

    # 前向 & 取与训练一致的裁边残差（comps）
    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast if USE_AMP else nullcontext
    with torch.no_grad(), amp_ctx("cuda" if device.startswith("cuda") else "cpu"):
        y_img, y_keff = model(x)
        _, comps = total_loss(x, y_img, y_keff, CFG_LOSS, CFG_PDE, y_gt)

    # —— 可视化头输出 —— #
    T  = y_img[0,0].detach().cpu().numpy()
    qx = y_img[0,1].detach().cpu().numpy()
    qy = y_img[0,2].detach().cpu().numpy()

    def _save_img(arr, path, cmap="RdBu_r", vmin=None, vmax=None):
        plt.figure(figsize=(4,4), dpi=300)
        im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        plt.axis("off"); plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight", pad_inches=0.02); plt.close()

    _save_img(T,  os.path.join(out_root,"T",  f"epoch_{epoch}_T.png"),  vmin=0.0, vmax=1.0)
    vmax_q = 2.0
    _save_img(qx, os.path.join(out_root,"qx", f"epoch_{epoch}_qx.png"), vmin=-vmax_q, vmax=vmax_q)
    _save_img(qy, os.path.join(out_root,"qy", f"epoch_{epoch}_qy.png"), vmin=-vmax_q, vmax=vmax_q)

    # —— keff CSV —— #
    keff_head = float(y_keff[0, 0].detach().cpu().item())
    kflux_t   = comps.get("keff_from_flux", None)
    keff_flux = float(kflux_t[0].view(-1)[0].detach().cpu().item()) if isinstance(kflux_t, torch.Tensor) else None
    keff_gt   = None
    if y_gt is not None and isinstance(y_gt.get("keff", None), torch.Tensor):
        keff_gt = float(y_gt["keff"][0].view(-1)[0].detach().cpu().item())

    import csv as _csv
    with open(os.path.join(out_root,"keff",f"epoch_{epoch}_keff.csv"),"w",newline="") as f:
        w = _csv.writer(f); hdr = ["keff_head","keff_from_flux"]
        if keff_gt is not None: hdr += ["keff_gt"]
        w.writerow(hdr)
        row = [f"{keff_head:.8f}", f"{keff_flux:.8f}" if keff_flux is not None else ""]
        if keff_gt is not None: row += [f"{keff_gt:.8f}"]
        w.writerow(row)

    # —— 由 T 推导的一致化 q_cons(T) —— #
    B, _, H, W = y_img.shape
    dx, dy, k_solid, k_pore, T_left, T_right, T_scale, q_scale = phys_from_cfg(CFG_PDE, H, W)
    porous = x[:,0:1].float().detach()
    k_field = make_k_field_from_mask(porous, k_solid, k_pore).to(device=y_img.device, dtype=y_img.dtype)

    T_phys    = y_img[:,0:1].detach() * T_scale
    T_phys_bc = apply_hard_bc_T(T_phys, T_left, T_right) if bool(CFG_PDE.get("hard_bc",True)) else T_phys
    qx_cons_c, qy_cons_c, _ = conservative_grad_div(T_phys_bc, k_field, dx, dy)
    _save_img(qx_cons_c[0,0].detach().cpu().numpy(), os.path.join(out_root,"qx_consT", f"epoch_{epoch}_qx_consT.png"),
              vmin=-vmax_q, vmax=vmax_q)
    _save_img(qy_cons_c[0,0].detach().cpu().numpy(), os.path.join(out_root,"qy_consT", f"epoch_{epoch}_qy_consT.png"),
              vmin=-vmax_q, vmax=vmax_q)

    # —— PDE 残差热图 —— #
    def _save_residual(tensor, name):
        if not isinstance(tensor, torch.Tensor): 
            return
        arr  = tensor[0,0].detach().cpu().numpy()
        mabs = float(np.mean(np.abs(arr))) if arr.size else 0.0   
        vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
        plt.figure(figsize=(4,4), dpi=300)
        plt.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin="upper")
        plt.axis("off"); plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"{name} (mean|·|={mabs:.2e})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root,"residual_maps", f"epoch_{epoch}_{name}.png"),
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()

    _save_residual(comps.get("res_fourier_x", None), "res_qx")
    _save_residual(comps.get("res_fourier_y", None), "res_qy")
    _save_residual(comps.get("res_div",       None), "res_div")

# ---- curves ----

def save_loss_curves(hist, out_root):
    import matplotlib.pyplot as plt
    os.makedirs(out_root, exist_ok=True)
    METRICS = {
        "total": "L_total", "mse": "L_mse", "fourier": "L_four", "div": "L_div",
        "bc_T": "L_bc_T", "bc_N": "L_bc_N", "keff": "L_keffC", "keff_flux": "keff_from_flux",
    }
    for nice, key in METRICS.items():
        tr = hist["train"].get(key, []); va = hist["val"].get(key, [])
        if not tr and not va: continue
        plt.figure(figsize=(6,4), dpi=160)
        if tr: plt.plot(np.arange(1, len(tr)+1), tr, label="train")
        if va: plt.plot(np.arange(1, len(va)+1), va, label="val")
        plt.xlabel("epoch"); plt.ylabel(nice); plt.title(f"{nice} curve")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        subdir = os.path.join(out_root, nice); _ensure_dir(subdir)
        plt.savefig(os.path.join(subdir, "curve.png")); plt.close()
        with open(os.path.join(subdir, "curve.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch", f"{nice}_train", f"{nice}_val"]) 
            L = max(len(tr), len(va))
            for i in range(L): w.writerow([i+1, tr[i] if i < len(tr) else "", va[i] if i < len(va) else ""]) 

# ---- batch helper ----
TRANSPOSE_GT_HW = True

def _batch_to_targets(batch):
    if not isinstance(batch, dict): return batch, None
    x = batch["x"]
    if DATASET_TYPE == "unsupervised": return x, None
    ygt = {"T": batch.get("T", None), "qx": batch.get("qx", None), "qy": batch.get("qy", None), "keff": batch.get("keff", None)}
    if TRANSPOSE_GT_HW:
        for k in ("T","qx","qy"):
            if isinstance(ygt.get(k, None), torch.Tensor) and ygt[k].ndim == 4:
                ygt[k] = _tHW(ygt[k])
    return x, ygt

# ---- main ----

def main():

    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== {DATASET_TYPE.upper()} | MODEL={MODEL_TYPE} | GRAD={GRAD_TYPE} ===")
    print(f"Data: {H5_PATH}"); print(f"RUN_DIR: {RUN_DIR}")
    _dump_run_config(RUN_DIR, dict(
        DATASET_TYPE=DATASET_TYPE, H5_PATH=H5_PATH,
        MODEL_TYPE=MODEL_TYPE, GRAD_TYPE=GRAD_TYPE,
        IN_CHANNELS=IN_CHANNELS, WIDTH=WIDTH, MODES1=MODES1, MODES2=MODES2, N_LAYERS=N_LAYERS,
        CFG_PDE=CFG_PDE, CFG_LOSS=CFG_LOSS, USE_AMP=USE_AMP,
    ))

    try:
        train_ds, val_ds, test_ds = make_splits_from_h5(H5_PATH)
        print("[data] use preset splits")
    except Exception:
        train_ds, val_ds, test_ds = make_splits(H5_PATH, n_train=800, n_val=200, n_test=106, seed=SEED)
        print("[data] random splits")
    print("val_ds[0] 原始索引 =", val_ds.indices[0])
    train_loader = DataLoader(train_ds, batch_size=BS_TRAIN, shuffle=True,  num_workers=20, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS_EVAL,  shuffle=False, num_workers=20, pin_memory=True)

    model = build_model(name=MODEL_TYPE, in_channels=IN_CHANNELS, width=WIDTH,
                        modes1=MODES1, modes2=MODES2, n_layers=N_LAYERS).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    ADAMW_LR = 2e-3; ADAMW_MIN_LR = 2e-4; ADAMW_WD = 1e-4; ADAMW_BETAS = (0.9, 0.98)
    opt = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=ADAMW_WD, betas=ADAMW_BETAS)
    steps_per_epoch = max(1, len(train_loader)); WARMUP_EPOCHS = 3

    def _cosine_with_warmup(optimizer, base_lr, min_lr, warmup_epochs, total_epochs):
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        total_steps  = int(total_epochs * steps_per_epoch)
        min_ratio = float(min_lr) / float(base_lr)
        def lr_lambda(gs):
            if gs < warmup_steps: return float(gs) / max(1, warmup_steps)
            prog = (gs - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return min_ratio + (1.0 - min_ratio) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    sched = _cosine_with_warmup(opt, ADAMW_LR, ADAMW_MIN_LR, WARMUP_EPOCHS, EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    with open(LOG_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","phase","time_sec","L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC","keff_from_flux"])

    hist = {
        "train": {k: [] for k in ["L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC","keff_from_flux"]},
        "val":   {k: [] for k in ["L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC","keff_from_flux"]},
    }

    def _log_row(ep, phase, dt, comps_avg):
        row = [ep, phase, f"{dt:.3f}"] + [float(comps_avg.get(k, 0.0)) for k in
            ["L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC","keff_from_flux"]]
        with open(LOG_CSV_PATH, "a", newline="") as f: csv.writer(f).writerow(row)

    def _accum(dsum, comps):
        for k in ["L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC"]:
            dsum[k] += float(comps.get(k, 0.0))
        if "keff_from_flux" in comps and isinstance(comps["keff_from_flux"], torch.Tensor):
            dsum["keff_from_flux"] += float(comps["keff_from_flux"].mean().item())
        else:
            dsum["keff_from_flux"] += float(comps.get("keff_from_flux", 0.0))

    def _avg(dsum, n): return {k: (dsum[k] / max(1, n)) for k in dsum.keys()}

    best_path = os.path.join(CHECKPOINT_DIR, "best.pt"); best_val = float("inf"); global_step = 0

    for ep in range(1, EPOCHS+1):
        # Optional schedule
        # _apply_loss_schedule(CFG_LOSS, ep)

        # Train
        model.train(); t0 = time.perf_counter()
        tsum = {k: 0.0 for k in ["L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC","keff_from_flux"]}; n_tb = 0
        for batch in DataLoader(train_ds, batch_size=BS_TRAIN, shuffle=True,  num_workers=20, pin_memory=True):
            x, y_gt = _batch_to_targets(batch)
            x = _to_dev(x, device); y_gt = _to_dev(y_gt, device) if y_gt is not None else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                y_img, y_keff = model(x)
                loss, comps = total_loss(x, y_img, y_keff, CFG_LOSS, CFG_PDE, y_gt)
            scaler.scale(loss).backward()
            if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            scaler.step(opt); scaler.update(); sched.step()
            _accum(tsum, comps); n_tb += 1; global_step += 1
        tavg = _avg(tsum, n_tb); _log_row(ep, "train", time.perf_counter()-t0, tavg)
        for k in hist["train"].keys(): hist["train"][k].append(tavg[k])

        # Val
        model.eval(); v0 = time.perf_counter()
        vsum = {k: 0.0 for k in ["L_total","L_mse","L_four","L_div","L_bc_T","L_bc_N","L_keffC","keff_from_flux"]}; n_vb = 0
        with torch.no_grad():
            for batch in DataLoader(val_ds, batch_size=BS_EVAL, shuffle=False, num_workers=20, pin_memory=True):
                x, y_gt = _batch_to_targets(batch)
                x = _to_dev(x, device); y_gt = _to_dev(y_gt, device) if y_gt is not None else None
                y_img, y_keff = model(x)
                loss, comps = total_loss(x, y_img, y_keff, CFG_LOSS, CFG_PDE, y_gt)
                _accum(vsum, comps); n_vb += 1
        vavg = _avg(vsum, n_vb); _log_row(ep, "val", time.perf_counter()-v0, vavg)
        for k in hist["val"].keys(): hist["val"][k].append(vavg[k])

        save_val_visuals(model, DataLoader(val_ds, batch_size=BS_EVAL, shuffle=False, num_workers=2), ep, device, RESULT_IMG_DIR)
        save_loss_curves(hist, LOSS_DIR)

        if vavg["L_total"] < best_val:
            best_val = vavg["L_total"]
            to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({"epoch": ep, "best_val": best_val, "model": to_save.state_dict()}, best_path)

    print(f"[done] best val L_total = {best_val:.6g} | ckpt: {best_path}")

if __name__ == "__main__":
    main()
