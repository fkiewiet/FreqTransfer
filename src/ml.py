"""
ml.py
-----
Optional ML utilities for FreqTransfer: datasets, models, training, and metrics.

This file is self-contained but depends on PyTorch.
If torch is not installed, importing functions from here will raise a helpful error.

Public API:
- build_direct_map(...), build_freq_transfer(...)
- SimpleFNO, LocalCNN
- train_model(...), eval_relative_metrics(...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# --- Soft dependency guard ----------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.fft
    from torch.utils.data import Dataset, DataLoader
except Exception as _e:
    _TORCH_IMPORT_ERROR = _e
    torch = None  # type: ignore
    nn = None     # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore

import numpy as np

from .config import GridSpec
from .loads import build_load, RandomPointSource, PointSource
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
    """(N, H, W) or (H, W) complex -> (..., 2, H, W) real with [Re, Im]."""
    if x.ndim == 2:  # (H,W)
        x = x[None, ...]
    re = np.real(x)
    im = np.imag(x)
    out = np.stack([re, im], axis=1)  # (N, 2, H, W)
    return out

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
    Dataset of pairs (rhs -> solution) for Helmholtz problems at varying k, sources, etc.

    Each item returns:
      x: input tensor (C_in, H, W) where channels can include:
         - 0..1: RHS [Re, Im]
         - optionally +1: k-parameter channel (constant image)
      y: target tensor (2, H, W) = solution [Re, Im]
    """
    def __init__(
        self,
        inputs: np.ndarray,   # (N, C_in, H, W), real
        targets: np.ndarray,  # (N, 2, H, W), real
    ):
        _require_torch()
        assert inputs.ndim == 4 and targets.ndim == 4
        assert inputs.shape[0] == targets.shape[0]
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

    Returns a Dataset with x=(rhs [2ch], optional k-channel) and y=(solution [2ch]).
    """
    _require_torch()
    rng = np.random.default_rng(seed)

    H, W = grid.shape
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for i in range(n):
        k = rng.uniform(*k_range)
        # Random point source
        spec = RandomPointSource(seed=rng.integers(0, 1_000_000))
        b = build_load(spec, grid).reshape(grid.shape)

        # Assemble and solve
        A = assemble_operator(grid=grid, k=float(k), kind="helmholtz")
        res = gmres_solve(A, b.ravel(), options=None if gmres_tol is None else type("O",(object,),{"tol":gmres_tol})())
        u = res.solution.reshape(grid.shape)

        # Build input/target tensors (numpy first)
        rhs_2ch = np_complex_to_2ch(b)  # (1, 2, H, W)
        u_2ch = np_complex_to_2ch(u)    # (1, 2, H, W)

        if include_k_channel:
            k_map = np_k_to_channel(k, (H, W))[None, None, ...]  # (1, 1, H, W)
            x = np.concatenate([rhs_2ch, k_map], axis=1)         # (1, 3, H, W)
        else:
            x = rhs_2ch  # (1, 2, H, W)

        xs.append(x[0])
        ys.append(u_2ch[0])

    X = np.stack(xs, axis=0)  # (N, C_in, H, W)
    Y = np.stack(ys, axis=0)  # (N, 2, H, W)
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
    We form RHS at omega and solve twice (at ω and ω'), producing paired fields.

    Input channels: u(omega) [2ch] + optional ω channel (omitted for brevity).
    Target: u(omega_p) [2ch].
    """
    _require_torch()
    rng = np.random.default_rng(seed)
    H, W = grid.shape
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    k = omega  # for this sandbox, treat omega as k (same units for simplicity)
    k_p = omega_p

    for i in range(n):
        spec = RandomPointSource(seed=rng.integers(0, 1_000_000))
        b = build_load(spec, grid).reshape(grid.shape)

        A_w = assemble_operator(grid=grid, k=float(k), kind="helmholtz")
        u_w = gmres_solve(A_w, b.ravel(), options=None if gmres_tol is None else type("O",(object,),{"tol":gmres_tol})()).solution.reshape(grid.shape)

        A_wp = assemble_operator(grid=grid, k=float(k_p), kind="helmholtz")
        u_wp = gmres_solve(A_wp, b.ravel(), options=None if gmres_tol is None else type("O",(object,),{"tol":gmres_tol})()).solution.reshape(grid.shape)

        x = np_complex_to_2ch(u_w)[0]   # (2, H, W)
        y = np_complex_to_2ch(u_wp)[0]  # (2, H, W)

        xs.append(x)
        ys.append(y)

    X = np.stack(xs, axis=0)  # (N, 2, H, W)
    Y = np.stack(ys, axis=0)  # (N, 2, H, W)
    return HelmholtzDirectDataset(X, Y)


# =============================================================================
# Models
# =============================================================================

# ---- SimpleFNO (compact Fourier layer + pointwise MLP) -----------------------

class SpectralConv2d(nn.Module):
    """
    Minimal spectral convolution (FNO-style):
    - FFT2 input channels
    - keep low-frequency modes (modes1, modes2)
    - learn complex weights in Fourier space
    - iFFT2 back to spatial
    """
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        _require_torch()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        # Complex weights as two real tensors
        self.wr = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)
        self.wi = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)

    def compl_mul2d(self, a, wr, wi):
        # a: (B, C_in, H, W) complex in Fourier domain
        # wr/wi: (C_in, C_out, m1, m2) real
        a_r, a_i = a.real, a.imag
        w_r, w_i = wr, wi
        # Multiply only the kept modes window
        out_r = torch.einsum("bcxy, ckom -> bkom", a_r, w_r) - torch.einsum("bcxy, ckom -> bkom", a_i, w_i)
        out_i = torch.einsum("bcxy, ckom -> bkom", a_r, w_i) + torch.einsum("bcxy, ckom -> bkom", a_i, w_r)
        return torch.complex(out_r, out_i)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1) complex
        m1 = min(self.modes1, x_ft.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_ch, H, W//2 + 1, dtype=torch.cfloat, device=x.device)
        # Low-frequency block
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.wr, self.wi)
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").real  # back to real
        return x


class SimpleFNO(nn.Module):
    """
    A tiny FNO-like network:
    Input: (B, C_in, H, W)  -> Output: (B, 2, H, W)
    """
    def __init__(self, in_ch: int = 3, width: int = 48, modes: Tuple[int, int] = (12, 12), layers: int = 4):
        super().__init__()
        _require_torch()
        self.in_ch = in_ch
        self.width = width
        self.layers = layers
        m1, m2 = modes

        self.proj_in = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, m1, m2) for _ in range(layers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(width, 2, 1)  # output is 2 channels: [Re, Im]

    def forward(self, x):
        x = self.proj_in(x)
        for sc, wc in zip(self.spectral, self.w):
            y = sc(x)
            z = wc(x)
            x = self.act(y + z)
        return self.proj_out(x)


# ---- LocalCNN (lightweight U-Net-ish encoder-decoder) ------------------------

class LocalCNN(nn.Module):
    """
    A small local convolutional model for baselines.
    """
    def __init__(self, in_ch: int = 3, width: int = 48):
        super().__init__()
        _require_torch()
        C = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


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
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Minimal supervised training loop (MSE loss).
    Returns (model, history_dict).
    """
    _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split into train/val
    n = len(dataset)
    n_val = int(round(val_split * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    hist = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                va_losses.append(loss_fn(yhat, y).item())

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
    re, im = x2[0], x2[1]
    u = re + 1j * im
    return np.abs(u), np.angle(u)

def eval_relative_metrics(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 16,
    device: str | None = None,
) -> Dict[str, float]:
    """
    Compute:
      - rel L2 mean / median / p90
      - magnitude RMSE
      - phase RMSE
    """
    _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()

    rels: List[float] = []
    mag_sqerr_sum = 0.0
    phase_sqerr_sum = 0.0
    n_pix_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)

            # move to cpu numpy
            yh = yhat.detach().cpu().numpy()
            yt = y.detach().cpu().numpy()

            # per-sample relative L2
            for i in range(yh.shape[0]):
                rels.append(_rel_l2(yh[i], yt[i]))

            # pixelwise mag/phase RMSE
            for i in range(yh.shape[0]):
                mag_h, ph_h = _mag_phase(yh[i])
                mag_t, ph_t = _mag_phase(yt[i])
                # unwrap phase difference to [-pi, pi]
                dphi = np.angle(np.exp(1j * (ph_h - ph_t)))
                mag_sqerr_sum += float(np.mean((mag_h - mag_t) ** 2))
                phase_sqerr_sum += float(np.mean(dphi ** 2))
                n_pix_total += 1

    rels_np = np.array(rels, dtype=np.float64)
    out = {
        "rel_L2_mean": float(np.mean(rels_np)),
        "rel_L2_median": float(np.median(rels_np)),
        "rel_L2_p90": float(np.quantile(rels_np, 0.90)),
        "mag_RMSE": float(np.sqrt(mag_sqerr_sum / max(n_pix_total, 1))),
        "phase_RMSE": float(np.sqrt(phase_sqerr_sum / max(n_pix_total, 1))),
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
