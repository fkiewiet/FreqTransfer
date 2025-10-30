"""
Unified import interface for the FreqTransfer project.
Clean, stable API for notebooks and scripts.
"""

# --- Core configuration and types ---
from .config import GridSpec, FDConfig, PMLConfig, SolverOptions

# --- Grid utilities ---
from .grid import spacing, mesh, size

# --- Loads (RHS definitions) ---
from .loads import PointSource, PlaneWave, RandomPointSource, build_load

# --- Operators (assembly of linear systems & frequency transfer) ---
from .operators import (
    laplacian_operator,
    helmholtz_operator,
    assemble_operator,
)
try:
    # Optional: keep transfer separate if desired
    from .transfer import frequency_transfer
except ImportError:
    frequency_transfer = None

# --- Solvers (GMRES / direct) ---
from .solvers import SolverResult, gmres_solve, direct_solve, direct_solve_auto

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
except ImportError:
    # Torch not installed — skip ML modules silently
    pass

__all__ = [
    # Config
    "GridSpec", "FDConfig", "PMLConfig", "SolverOptions",
    # Grid
    "spacing", "mesh", "size",
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

# Add optional items if available
if frequency_transfer:
    __all__.append("frequency_transfer")
try:
    __all__.extend([
        "build_direct_map", "build_freq_transfer",
        "SimpleFNO", "LocalCNN",
        "train_model", "eval_relative_metrics",
    ])
except NameError:
    pass
