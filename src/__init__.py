"""
Unified import interface for the FreqTransfer project.
Clean, stable API for notebooks and scripts.

This module lazily exposes ML symbols so that importing `src` never fails even
when PyTorch isn't installed. Accessing an ML symbol triggers on-demand import.
"""

# --- Core configuration and types ---
from .config import GridSpec, FDConfig, PMLConfig, SolverOptions

# --- Loads (RHS definitions) ---
from .loads import PointSource, PlaneWave, RandomPointSource, build_load

# --- Operators (assembly of linear systems) ---
from .operators import (
    assemble_operator,
    helmholtz_operator,
    laplacian_operator,
    make_extended_grid,
    solve_with_pml_shell,
)

# --- Solvers (GMRES / direct) ---
from .solvers import (
    SolverResult,
    gmres_solve,
    direct_solve,
    direct_solve_auto,
)

# --- Utilities (plotting, I/O, sweeps) ---
from .utils import (
    plot_field, plot_residuals,
    save_field, load_field, save_config, load_config,
    sweep_k, sweep_sources,
)

from .ml import (
    SimpleFNO,
    LocalCNN,
    train_model,
    eval_relative_metrics,
    HelmholtzFreqTransferDataset,
    evaluate_transfer_model,
)

# --- ML symbols (lazy import) ---
_ML_EXPORTS = {
    "build_direct_map",
    "build_freq_transfer",
    "SimpleFNO",
    "LocalCNN",
    "train_model",
    "eval_relative_metrics",
}

def __getattr__(name: str):
    """Lazy-load ML submodule on first access to any ML symbol."""
    if name in _ML_EXPORTS:
        from . import ml as _ml
        try:
            return getattr(_ml, name)
        except AttributeError as e:
            # Re-raise with a clearer message if symbol is missing in ml.py
            raise AttributeError(f"'src.ml' has no attribute '{name}'") from e
    raise AttributeError(f"module 'src' has no attribute '{name}'")

def __dir__():
    # Make ML symbols visible to dir() and auto-complete
    return sorted(set(globals().keys()) | _ML_EXPORTS)

__all__ = [
    # Config
    "GridSpec", "FDConfig", "PMLConfig", "SolverOptions",
    # Loads
    "PointSource", "PlaneWave", "RandomPointSource", "build_load",
    # Operators
    "laplacian_operator", "helmholtz_operator", "assemble_operator",
    # Solvers
    "SolverResult", "gmres_solve", "direct_solve", "direct_solve_auto",
    # Utils
    "plot_field", "plot_residuals",
    "save_field", "load_field", "save_config", "load_config",
    "sweep_k", "sweep_sources",
    # ML (listed for discoverability; resolved lazily via __getattr__)
    *sorted(_ML_EXPORTS),
]
