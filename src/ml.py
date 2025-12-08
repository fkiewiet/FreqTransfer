from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Soft dependency guard for PyTorch
# ----------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as _e:
    _TORCH_IMPORT_ERROR = _e
    torch = None          # type: ignore
    nn = None             # type: ignore
    Dataset = object      # type: ignore
    DataLoader = object   # type: ignore

from .config import GridSpec, SolverOptions
from .loads import build_load, RandomPointSource
from .operators import assemble_operator
from .solvers import gmres_solve

# Alle data-gerelateerde hulpfuncties komen nu uit data.py
from .data import (
    np_complex_to_2ch,
    HelmholtzFreqTransferDataset,
    PrecomputedFreqDataset,
    get_freq_dataset,
    AmpNormWrapper,
    StdNormWrapper,
    compute_input_stats,
)


# ----------------------------------------------------------------------
# Torch availability helper
# ----------------------------------------------------------------------
def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for this functionality but could not be imported. "
            f"Original error: {_TORCH_IMPORT_ERROR}"
        )


# ----------------------------------------------------------------------
# Kleine helper: wavenumber-k kanaal
# ----------------------------------------------------------------------
def np_k_to_channel(k: float | np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Maak een (H,W)-array gevuld met k, eventueel uit een array gebroadcast.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber (scalar of array).
    shape : (H, W)
        Gewenste ruimtelijke shape.

    Returns
    -------
    ndarray of shape (H, W), dtype float32
    """
    if np.isscalar(k):
        return np.full(shape, float(k), dtype=np.float32)
    kk = np.array(k, dtype=np.float32)
    if kk.shape != shape:
        kk = np.broadcast_to(kk, shape).astype(np.float32)
    return kk


# =============================================================================
# Direct map datasets (rechtstreeks A(k)u = b oplossen)
# =============================================================================

class HelmholtzDirectDataset(Dataset):
    """
    Torch dataset voor paren (input -> target) waar:
      input  : (C_in, H, W) real; bijv. [Re(rhs), Im(rhs), k-channel]
      target : (2,    H, W) real; [Re(u), Im(u)]
    """
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        _require_torch()
        assert inputs.ndim == 4 and targets.ndim == 4, "inputs/targets must be (N,C,H,W)"
        assert inputs.shape[0] == targets.shape[0], "N mismatch"
        self.x = torch.from_numpy(inputs.astype(np.float32)).contiguous()
        self.y = torch.from_numpy(targets.astype(np.float32)).contiguous()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def build_direct_map(
    n: int,
    grid: GridSpec,
    k_range: Tuple[float, float] = (20.0, 40.0),
    include_k_channel: bool = True,
    seed: int = 0,
    gmres_tol: float = 1e-6,
) -> HelmholtzDirectDataset:
    """
    Synthesise a dataset by sampling random point sources and k in [kmin, kmax],
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
    Build a *direct* dataset mapping u(omega) -> u(omega_p) (zonder PML-shell).

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

    SimpleFNO = _TorchMissing  # type: ignore
    LocalCNN  = _TorchMissing  # type: ignore

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
            # weights for real & imaginary parts, shape (in_ch, out_ch, modes1, modes2)
            self.wr = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)
            self.wi = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)

        def _mul(self, a, wr, wi):
            """
            a : (B, in_ch, m1, m2) complex
            wr, wi : (in_ch, out_ch, modes1, modes2)

            We mix channels per frequency:
                out_ft[b, out_c, kx, ky] =
                    Σ_in_c (a[b, in_c, kx, ky] * (wr + i wi)[in_c, out_c, kx, ky])
            """
            a_r, a_i = a.real, a.imag      # (B, Cin, m1, m2)

            # restrict weights to the actually used modes
            m1, m2 = a_r.shape[-2], a_r.shape[-1]
            wr_use = wr[:, :, :m1, :m2]    # (Cin, Cout, m1, m2)
            wi_use = wi[:, :, :m1, :m2]

            # per-frequency channel mixing: bcxy, ckxy -> bkxy
            out_r = torch.einsum("bcxy, ckxy -> bkxy", a_r, wr_use) - \
                    torch.einsum("bcxy, ckxy -> bkxy", a_i, wi_use)
            out_i = torch.einsum("bcxy, ckxy -> bkxy", a_r, wi_use) + \
                    torch.einsum("bcxy, ckxy -> bkxy", a_i, wr_use)

            return torch.complex(out_r, out_i)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")   # (B, Cin, H, W//2+1)

            m1 = min(self.modes1, x_ft.shape[-2])
            m2 = min(self.modes2, x_ft.shape[-1])

            out_ft = torch.zeros(
                B, self.out_ch, H, W // 2 + 1,
                dtype=torch.cfloat, device=x.device
            )

            out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.wr, self.wi)

            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").real


class SimpleFNO(nn.Module):
    """Tiny FNO-like model: (B, C_in, H, W) -> (B, out_ch, H, W)."""
    def __init__(
        self,
        in_ch: int = 3,
        width: int = 48,
        modes: Tuple[int, int] = (12, 12),
        layers: int = 4,
        out_ch: int = 2,
        use_global_skip: bool = False,
    ):
        super().__init__()
        self.use_global_skip = use_global_skip
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.proj_in = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, *modes) for _ in range(layers)]
        )
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(width, out_ch, 1)

        # For optional global skip when in_ch == out_ch (e.g. identity test)
        if use_global_skip and in_ch == out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x  # keep original input for skip-connection if desired

        x = self.proj_in(x)
        for sc, wc in zip(self.spectral, self.w):
            x = self.act(sc(x) + wc(x))
        x = self.proj_out(x)

        if self.use_global_skip and (self.skip is not None):
            # project input to out_ch and add
            x = x + self.skip(x_in)

        return x


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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

# =============================================================================
# Training & metrics
# =============================================================================

def relative_l2_loss(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12):
    """
    Compute batch-wise relative L2 loss:

        L = mean_b( ||yhat_b - y_b|| / (||y_b|| + eps) )
    """
    B = y.shape[0]
    diff = yhat - y
    num = torch.linalg.norm(diff.view(B, -1), dim=1)
    den = torch.linalg.norm(y.view(B, -1), dim=1) + eps
    return (num / den).mean()


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
    loss_type: str = "mse",     # "mse" or "rel_l2"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Minimal supervised training loop.

    loss_type:
        "mse"     -> standard mean-squared-error loss
        "rel_l2"  -> batch-wise relative L2 loss (scale-invariant)
    """
    _require_torch()

    # ----------------------------------------------------------
    # Device selection
    # ----------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------
    # Train/val split
    # ----------------------------------------------------------
    n = len(dataset)
    n_val = int(round(val_split * n))
    n_train = n - n_val

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    # ----------------------------------------------------------
    # Model + optimiser
    # ----------------------------------------------------------
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ----------------------------------------------------------
    # Choose loss function
    # ----------------------------------------------------------
    lt = loss_type.lower()
    if lt == "mse":
        def loss_fn(yhat, y):
            return nn.functional.mse_loss(yhat, y)
    elif lt in ("rel_l2", "relative_l2"):
        def loss_fn(yhat, y):
            return relative_l2_loss(yhat, y)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'mse' or 'rel_l2'.")

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    hist = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        # ---- Train phase ----
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

        # ---- Validation phase ----
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                va_losses.append(loss_fn(model(x), y).item())

        # Record history
        hist["train"].append(float(np.mean(tr_losses)) if tr_losses else float("nan"))
        hist["val"].append(float(np.mean(va_losses)) if va_losses else float("nan"))

        # Logging
        if verbose:
            print(f"[{ep:03d}/{epochs}] train={hist['train'][-1]:.4e}  val={hist['val'][-1]:.4e}")

    return model, hist



def _rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + eps
    return float(num / den)


def _mag_phase(x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x2: (2, H, W) with [Re, Im]
    returns (|u|, arg(u)).
    """
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
                dphi = np.angle(np.exp(1j * (ph_h - ph_t)))  # wrap diff to [-pi, pi]
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


def evaluate_transfer_model(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cpu",
) -> np.ndarray:
    """
    Evaluate a trained frequency-transfer model on a dataset of
    (u_src, u_tgt) pairs.

    Returns an array of relative L2 errors per sample.
    """
    _require_torch()
    model = model.to(device).eval()
    rel_L2s: List[float] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            u_src, u_tgt = dataset[i]
            u_src = u_src.unsqueeze(0).to(device)  # (1, 2, Ny, Nx)
            u_tgt = u_tgt.unsqueeze(0).to(device)

            u_pred = model(u_src)  # (1, 2, Ny, Nx)

            diff = u_pred - u_tgt
            num  = torch.linalg.norm(diff.view(1, -1), ord=2, dim=-1)
            den  = torch.linalg.norm(u_tgt.view(1, -1), ord=2, dim=-1) + 1e-12
            rel_L2 = (num / den).item()
            rel_L2s.append(rel_L2)

    rel_L2s_arr = np.array(rel_L2s)
    print(
        f"rel_L2 mean={rel_L2s_arr.mean():.3e}, "
        f"median={np.median(rel_L2s_arr):.3e}, "
        f"p90={np.percentile(rel_L2s_arr, 90):.3e}"
    )
    return rel_L2s_arr


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # datasets (direct)
    "HelmholtzDirectDataset",
    "build_direct_map",
    "build_freq_transfer",
    # datasets (freq-transfer & cache) – doorgelust uit data.py
    "HelmholtzFreqTransferDataset",
    "PrecomputedFreqDataset",
    "get_freq_dataset",
    "AmpNormWrapper",
    "StdNormWrapper",
    "compute_input_stats",
    # models
    "SimpleFNO",
    "LocalCNN",
    # training / metrics
    "train_model",
    "eval_relative_metrics",
    "evaluate_transfer_model",
    # utilities
    "np_complex_to_2ch",
    "np_k_to_channel",
    #loss
    "relative_l2_loss",   

]
