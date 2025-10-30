"""
Unified import interface for the FreqTransfer project.
Clean, stable API for notebooks and scripts.
"""

# --- Core configuration and types ---
from .config import GridSpec, FDConfig, PMLConfig, SolverOptions

# --- Loads (RHS definitions) ---
from .loads import PointSource, PlaneWave, RandomPointSource, build_load

# --- Operators (assembly of linear systems) ---
from .operators import (
    laplacian_operator,
    helmholtz_operator,
    assemble_operator,
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

# --- Optional ML tools (import lazily if torch is installed) ---
try:
    from .ml import (
        build_direct_map, build_freq_transfer,
        SimpleFNO, LocalCNN, train_model, eval_relative_metrics,
    )
    _HAS_ML = True
except Exception:
    _HAS_ML = False

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
]

if _HAS_ML:
    __all__.extend([
        "build_direct_map", "build_freq_transfer",
        "SimpleFNO", "LocalCNN",
        "train_model", "eval_relative_metrics",
    ])
