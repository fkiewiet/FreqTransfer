"""
ml.py
-----
Optional ML utilities for FreqTransfer: datasets, models, training, and metrics.

If PyTorch is not installed, importing ML features will raise a helpful error.

Public API:
- build_direct_map(...), build_freq_transfer(...)
- HelmholtzDirectDataset
- SimpleFNO, LocalCNN
- train_model(...), eval_relative_metrics(...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ----------------------------- Soft dependency guard -------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as _e:
    _TORCH_IMPORT_ERROR = _e
    torch = None  # type: ignore
    nn = None     # type: ignore
    Dataset = object  # fallback dummy
    DataLoader = object  # fallback dummy

import numpy as np

from .config import GridSpec, SolverOptions
from .loads import build_load, RandomPointSource
from .operators import assemble_operator
from .solvers import gmres_solve


# =============================================================================
# Utilities: complex <-> 2-channel real tensors
# =============================================================================

def _require_torch():
    if torch is None:
        raise ImportError(
            "This feature requires PyTorch. Install with:\n"
            "    pip install torch torchvision\n"
            f"(Original import error: {_TORCH_IMPORT_ERROR})"
        )

def np_complex_to_2ch(x: np.ndarray) -> np.ndarray:
    """(H,W) or (N,H,W) complex -> (..., 2, H, W) real with channels [Re, Im]."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]  # (1,H,W)
    re = np.real(x)
    im = np.imag(x)
    return np.stack([re, im], axis=1)  # (N,2,H,W)

def np_k_to_channel(k: float | np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Return a k-parameter map as single channel (H, W)."""
    if np.isscalar(k):
        return np.full(shape, float(k), dtype=np.float32)
    kk = np.array(k, dtype=np.float32)
    if kk.shape != shape:
        kk = np.broadcast_to(kk, shape).astype(np.float32)
    return kk


# =============================================================================
# Dataset builders
# =============================================================================

class HelmholtzDirectDataset(Dataset):
    """
    Torch dataset for pairs (input -> target) where:
      input  : (C_in, H, W) real; e.g. [Re(rhs), Im(rhs), k?]
      target : (2, H, W) real;   [Re(u), Im(u)]
    """
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        _require_torch()
        assert inputs.ndim == 4 and targets.ndim == 4, "inputs/targets must be (N,C,H,W)"
        assert inputs.shape[0] == targets.shape[0], "N mismatch"
        self.x = torch.from_numpy(inputs.astype(np.float32)).contiguous()
        self.y = torch.from_numpy(targets.astype(np.float32)).contiguous()

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]


def build_direct_map(
    n: int,
    grid: GridSpec,
    k_range: Tuple[float, float] = (20.0, 40.0),
    include_k_channel: bool = True,
    seed: int = 0,
    gmres_tol: float = 1e-6,
) -> HelmholtzDirectDataset:
    """
    Synthesize a dataset by sampling random point sources and k in [kmin, kmax],
    solving A(k) u = b with GMRES.

    Returns a HelmholtzDirectDataset:
      x = (rhs [2ch], optional k-channel)   -> shape (N, C_in, H, W)
      y = (solution [2ch])                  -> shape (N, 2,    H, W)
    """
    _require_torch()
    rng = np.random.default_rng(seed)

    H, W = grid.shape
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for _ in range(n):
        k = float(rng.uniform(*k_range))

        # Random point source
        spec = RandomPointSource(seed=int(rng.integers(0, 1_000_000)))
        b = build_load(spec, grid).reshape(grid.shape)  # complex (H,W)

        # Assemble and solve
        A = assemble_operator(grid=grid, k=k, kind="helmholtz")
        opts = SolverOptions(tol=gmres_tol) if gmres_tol is not None else SolverOptions()
        u = gmres_solve(A, b.ravel(), options=opts).solution.reshape(grid.shape)

        # Build input/target tensors (numpy)
        rhs_2ch = np_complex_to_2ch(b)  # (1,2,H,W)
        u_2ch   = np_complex_to_2ch(u)  # (1,2,H,W)

        if include_k_channel:
            k_map = np_k_to_channel(k, (H, W))[None, None, ...]  # (1,1,H,W)
            x = np.concatenate([rhs_2ch, k_map], axis=1)         # (1,3,H,W)
        else:
            x = rhs_2ch                                          # (1,2,H,W)

        xs.append(x[0])
        ys.append(u_2ch[0])

    X = np.stack(xs, axis=0)  # (N,C_in,H,W)
    Y = np.stack(ys, axis=0)  # (N,2,H,W)
    return HelmholtzDirectDataset(X, Y)


def build_freq_transfer(
    n: int,
    grid: GridSpec,
    omega: float,
    omega_p: float,
    seed: int = 0,
    gmres_tol: float = 1e-6,
) -> HelmholtzDirectDataset:
    """
    Build a dataset mapping u(omega) -> u(omega_p).

    Input channels: u(omega) [2ch]
    Target       : u(omega_p) [2ch]
    """
    _require_torch()
    rng = np.random.default_rng(seed)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    k  = float(omega)
    kp = float(omega_p)

    for _ in range(n):
        spec = RandomPointSource(seed=int(rng.integers(0, 1_000_000)))
        b = build_load(spec, grid).reshape(grid.shape)

        A_w  = assemble_operator(grid=grid, k=k,  kind="helmholtz")
        A_wp = assemble_operator(grid=grid, k=kp, kind="helmholtz")
        opts = SolverOptions(tol=gmres_tol) if gmres_tol is not None else SolverOptions()

        u_w  = gmres_solve(A_w,  b.ravel(), options=opts).solution.reshape(grid.shape)
        u_wp = gmres_solve(A_wp, b.ravel(), options=opts).solution.reshape(grid.shape)

        x = np_complex_to_2ch(u_w)[0]    # (2,H,W)
        y = np_complex_to_2ch(u_wp)[0]   # (2,H,W)

        xs.append(x)
        ys.append(y)

    X = np.stack(xs, axis=0)  # (N,2,H,W)
    Y = np.stack(ys, axis=0)  # (N,2,H,W)
    return HelmholtzDirectDataset(X, Y)


# =============================================================================
# Models (guarded so import works even if torch is missing)
# =============================================================================

if torch is None:
    # Placeholders that error on use but allow importing src.ml safely
    class _TorchMissing:
        def __init__(self, *args, **kwargs):
            _require_torch()
        def __call__(self, *args, **kwargs):
            _require_torch()

    SimpleFNO = _TorchMissing
    LocalCNN  = _TorchMissing

else:
    class SpectralConv2d(nn.Module):
        """
        Minimal spectral convolution (FNO-style):
        - rFFT2 -> keep low-frequency modes -> learn weights -> irFFT2
        """
        def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
            super().__init__()
            self.in_ch, self.out_ch = in_ch, out_ch
            self.modes1, self.modes2 = modes1, modes2
            scale = 1.0 / max(1, in_ch * out_ch)
            self.wr = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)
            self.wi = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)

        def _mul(self, a, wr, wi):
            a_r, a_i = a.real, a.imag
            out_r = torch.einsum("bcxy, ckom -> bkom", a_r, wr) - torch.einsum("bcxy, ckom -> bkom", a_i, wi)
            out_i = torch.einsum("bcxy, ckom -> bkom", a_r, wi) + torch.einsum("bcxy, ckom -> bkom", a_i, wr)
            return torch.complex(out_r, out_i)

        def forward(self, x):
            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            m1 = min(self.modes1, x_ft.shape[-2])
            m2 = min(self.modes2, x_ft.shape[-1])
            out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.wr, self.wi)
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").real

    class SimpleFNO(nn.Module):
        """Tiny FNO-like model: (B, C_in, H, W) -> (B, 2, H, W)."""
        def __init__(self, in_ch: int = 3, width: int = 48, modes: Tuple[int, int] = (12, 12), layers: int = 4):
            super().__init__()
            self.proj_in = nn.Conv2d(in_ch, width, 1)
            self.spectral = nn.ModuleList([SpectralConv2d(width, width, *modes) for _ in range(layers)])
            self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
            self.act = nn.GELU()
            self.proj_out = nn.Conv2d(width, 2, 1)

        def forward(self, x):
            x = self.proj_in(x)
            for sc, wc in zip(self.spectral, self.w):
                x = self.act(sc(x) + wc(x))
            return self.proj_out(x)

    class LocalCNN(nn.Module):
        """Lightweight local CNN baseline."""
        def __init__(self, in_ch: int = 3, width: int = 48):
            super().__init__()
            C = width
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, C, 3, padding=1), nn.GELU(),
                nn.Conv2d(C, C, 3, padding=1),     nn.GELU(),
                nn.Conv2d(C, C, 3, padding=1),     nn.GELU(),
                nn.Conv2d(C, 2, 1),
            )
        def forward(self, x): return self.net(x)



# =============================================================================
# Training & metrics
# =============================================================================

@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]


def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Minimal supervised training loop (MSE loss). Returns (model, history_dict).
    """
    _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = len(dataset)
    n_val = int(round(val_split * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    hist = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        # ---- train
        model.train()
        tr_losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # ---- val
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                va_losses.append(loss_fn(model(x), y).item())

        hist["train"].append(float(np.mean(tr_losses)) if tr_losses else float("nan"))
        hist["val"].append(float(np.mean(va_losses)) if va_losses else float("nan"))

        if verbose:
            print(f"[{ep:03d}/{epochs}] train={hist['train'][-1]:.4e}  val={hist['val'][-1]:.4e}")

    return model, hist


def _rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + eps
    return float(num / den)

def _mag_phase(x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # x2: (2, H, W) with [Re, Im]
    u = x2[0] + 1j * x2[1]
    return np.abs(u), np.angle(u)

def eval_relative_metrics(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute aggregated metrics:
      - rel L2 mean / median / p90
      - magnitude RMSE
      - phase RMSE (phase difference wrapped to [-pi, pi])
    """
    _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()

    rels: List[float] = []
    mag_sqerr_sum = 0.0
    phase_sqerr_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            yhat = model(x)

            yh = yhat.detach().cpu().numpy()
            yt = y.detach().cpu().numpy()

            # per-sample relative L2
            for i in range(yh.shape[0]):
                rels.append(_rel_l2(yh[i], yt[i]))

            # pixelwise mag/phase RMSE (batch mean)
            batch_mag_err = []
            batch_phase_err = []
            for i in range(yh.shape[0]):
                mag_h, ph_h = _mag_phase(yh[i])
                mag_t, ph_t = _mag_phase(yt[i])
                dphi = np.angle(np.exp(1j * (ph_h - ph_t)))  # wrap diff
                batch_mag_err.append(np.mean((mag_h - mag_t) ** 2))
                batch_phase_err.append(np.mean(dphi ** 2))

            mag_sqerr_sum += float(np.mean(batch_mag_err))
            phase_sqerr_sum += float(np.mean(batch_phase_err))
            n_batches += 1

    rels_np = np.array(rels, dtype=np.float64)
    out = {
        "rel_L2_mean": float(np.mean(rels_np)),
        "rel_L2_median": float(np.median(rels_np)),
        "rel_L2_p90": float(np.quantile(rels_np, 0.90)),
        "mag_RMSE": float(np.sqrt(mag_sqerr_sum / max(n_batches, 1))),
        "phase_RMSE": float(np.sqrt(phase_sqerr_sum / max(n_batches, 1))),
    }
    return out


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # datasets
    "HelmholtzDirectDataset",
    "build_direct_map",
    "build_freq_transfer",
    # models
    "SimpleFNO",
    "LocalCNN",
    # training / metrics
    "train_model",
    "eval_relative_metrics",
    # utilities
    "np_complex_to_2ch",
]
