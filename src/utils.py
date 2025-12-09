"""
utils.py
--------
Non-core utilities for the FreqTransfer project:

Plotting:
  - plot_field(u, shape, which="magnitude") -> Axes
  - plot_residuals(result, tol: float|None=None) -> Axes

I/O:
  - save_field(path, u, shape) -> None
  - load_field(path) -> np.ndarray
  - save_config(path, cfg) -> None
  - load_config(path) -> dict

Sweeps (no plotting inside):
  - sweep_k(grid, ks, rhs_builder, solver_opts, fd=None, pml=None, m=None) -> list[SolverResult]
  - sweep_sources(grid, seeds, k, solver_opts, fd=None, pml=None, m=None) -> list[SolverResult]
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

# YAML is optional but recommended (it's in requirements.txt). Fallback to JSON if missing.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from .config import GridSpec, FDConfig, PMLConfig, SolverOptions
from .operators import assemble_operator
from .solvers import SolverResult, gmres_solve
from .loads import RandomPointSource, build_load


# =============================================================================
# Plotting
# =============================================================================

def plot_residuals(result: SolverResult, tol: float | None = None):
    """
    Plot GMRES relative residual history (semilogy).
    """
    fig, ax = plt.subplots()
    y = result.residuals or []
    if len(y) > 0:
        ax.semilogy(y, marker="o", linewidth=1)
    if tol is not None:
        ax.axhline(tol, linestyle="--")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"Relative residual $\|r_k\|_2 / \|b\|_2$")
    ax.set_title("GMRES Residual Convergence")
    ax.grid(True, which="both", alpha=0.25)
    return ax


def plot_field(u: np.ndarray, shape: Tuple[int, ...], which: str = "magnitude"):
    """
    Visualize a complex field with imshow.

    Parameters
    ----------
    u : np.ndarray
        Complex array, either flattened (N,) or shaped (H,W).
    shape : tuple[int,...]
        Target shape for display (e.g., (H, W)).
    which : {"magnitude","real","imag","phase"}
    """
    U = u.reshape(shape) if u.ndim == 1 or u.size == int(np.prod(shape)) else u
    fig, ax = plt.subplots()

    if which == "real":
        im = ax.imshow(np.real(U), origin="lower")
        ax.set_title("Re(u)")
    elif which == "imag":
        im = ax.imshow(np.imag(U), origin="lower")
        ax.set_title("Im(u)")
    elif which == "phase":
        im = ax.imshow(np.angle(U), origin="lower")
        ax.set_title("arg(u)")
    else:
        im = ax.imshow(np.abs(U), origin="lower")
        ax.set_title("|u|")

    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x-index")
    ax.set_ylabel("y-index")
    ax.grid(False)
    return ax


# =============================================================================
# I/O helpers
# =============================================================================

def _to_path(path) -> Path:
    return Path(path).expanduser().resolve()

def save_field(path, u: np.ndarray, shape: Tuple[int, ...]) -> None:
    """
    Save a complex field to a compressed NPZ with shape metadata.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    U = u.reshape(shape)
    np.savez_compressed(p, field=U.astype(np.complex128), shape=np.array(shape, dtype=np.int64))

def load_field(path) -> np.ndarray:
    """
    Load a complex field saved by save_field(...). Returns shaped array.
    """
    p = _to_path(path)
    with np.load(p, allow_pickle=False) as z:
        U = z["field"]
        return U

def save_config(path, cfg) -> None:
    """
    Save a config/dataclass/dict to YAML (if available) or JSON.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    obj = asdict(cfg) if is_dataclass(cfg) else (cfg if isinstance(cfg, dict) else vars(cfg))

    if yaml is not None and p.suffix.lower() in {".yml", ".yaml"}:
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
    else:
        # Default to JSON
        with open(p.with_suffix(".json") if p.suffix == "" else p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

def load_config(path) -> dict:
    """
    Load a config saved by save_config(...). Tries YAML first if extension is .yml/.yaml.
    """
    p = _to_path(path)
    if p.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Fallback to JSON
    with open(p if p.suffix else p.with_suffix(".json"), "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Sweeps (numerical; no plotting)
# =============================================================================

def sweep_k(
    grid: GridSpec,
    ks: Iterable[float],
    rhs_builder: Callable[[GridSpec], np.ndarray],
    solver_opts: Optional[SolverOptions] = None,
    fd: Optional[FDConfig] = None,
    pml: Optional[PMLConfig] = None,
    m: Optional[np.ndarray] = None,
) -> List[SolverResult]:
    """
    Solve for multiple k values with a fixed RHS builder.

    Parameters
    ----------
    grid : GridSpec
        Domain specification.
    ks : iterable of float
        Wavenumbers to sweep.
    rhs_builder : Callable[[GridSpec], np.ndarray]
        Function that returns a flattened complex RHS for the given grid.
    solver_opts : SolverOptions | None
        GMRES options (defaults inside gmres_solve if None).
    fd, pml, m : optional
        Discretization / PML / slowness-squared overrides.

    Returns
    -------
    list[SolverResult]
    """
    results: List[SolverResult] = []
    b = rhs_builder(grid)
    for k in ks:
        A = assemble_operator(grid=grid, k=float(k), kind="helmholtz", fd=fd, pml=pml, m=m)
        res = gmres_solve(A, b, options=solver_opts)
        results.append(res)
    return results


def sweep_sources(
    grid: GridSpec,
    seeds: Iterable[int],
    k: float,
    solver_opts: Optional[SolverOptions] = None,
    fd: Optional[FDConfig] = None,
    pml: Optional[PMLConfig] = None,
    m: Optional[np.ndarray] = None,
) -> List[SolverResult]:
    """
    Solve for multiple randomized point sources at fixed k.

    Parameters
    ----------
    grid : GridSpec
    seeds : iterable of int
        RNG seeds for RandomPointSource.
    k : float
        Wavenumber.
    solver_opts : SolverOptions | None
    fd, pml, m : optional

    Returns
    -------
    list[SolverResult]
    """
    results: List[SolverResult] = []
    for s in seeds:
        b = build_load(RandomPointSource(seed=int(s)), grid)
        A = assemble_operator(grid=grid, k=float(k), kind="helmholtz", fd=fd, pml=pml, m=m)
        res = gmres_solve(A, b, options=solver_opts)
        results.append(res)
    return results



# =============================================================================
# Ladder
# =============================================================================



def run_single_pair(omega_src, omega_tgt):
    print("="*80)
    print(f"### Frequency Transfer {omega_src} → {omega_tgt} ###")
    print("="*80)

    # --- 1. Load or generate dataset ---
    freq_ds_raw = get_freq_dataset(
        grid=grid,
        pml=pml_cfg,
        omega_src=omega_src,
        omega_tgt=omega_tgt,
        N_samples=N_samples,
        omega_to_k=omega_to_k,
        overwrite=False,
    )

    # --- 2. Add coordinate + omega channels ---
    freq_ds = OmegaChannelWrapper(freq_ds_raw)
    freq_ds = CoordWrapper(freq_ds, grid=grid, normalise=True)

    x0, y0 = freq_ds[0]
    in_ch = x0.shape[0]
    print(f"Input channels = {in_ch}")

    # --- 3. Define model ---
    model = SimpleFNO(
        in_ch=in_ch,
        width=48,
        modes=(12, 12),
        layers=4,
        out_ch=2,
    ).to(device)

    # --- 4. Train ---
    model, hist = train_model(
        model=model,
        dataset=freq_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=0.2,
        loss_type="mse",
        device=device,
    )

    # --- 5. Evaluate ---
    metrics = eval_relative_metrics(model, freq_ds, batch_size=batch_size, device=device)
    print("\nMetrics:", metrics)

    return {
        "omega_src": omega_src,
        "omega_tgt": omega_tgt,
        "metrics": metrics,
    }



# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    # plotting
    "plot_field", "plot_residuals",
    # io
    "save_field", "load_field", "save_config", "load_config",
    # sweeps
    "sweep_k", "sweep_sources",
]
