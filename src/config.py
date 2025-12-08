"""
config.py
----------
Holds lightweight configuration dataclasses and global constants
for the FreqTransfer project.
This module has *no heavy dependencies* besides NumPy and typing.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from pathlib import Path

def omega_to_k(omega: float) -> float:
    c = 1.0
    return omega / c

# ----------------------------------------------------------------------
# === Numerical configuration dataclasses ===
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    """
    Defines the spatial grid layout.

    Parameters
    ----------
    dims : int
        Number of spatial dimensions (1, 2, or 3).
    shape : tuple[int, ...]
        Number of grid points per dimension, e.g. (60, 60).
    lengths : tuple[float, ...]
        Physical domain lengths in each direction, e.g. (1.0, 1.0).
    """
    dims: int
    shape: tuple[int, ...]
    lengths: tuple[float, ...]

    @property
    def spacing(self) -> tuple[float, ...]:
        """Grid spacing Δx per dimension."""
        return tuple(L / (n - 1) for L, n in zip(self.lengths, self.shape))

    @property
    def N(self) -> int:
        """Total number of points."""
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass(frozen=True)
class FDConfig:
    """
    Finite-difference scheme configuration.
    """
    bc: Literal["dirichlet", "neumann"] = "dirichlet"
    dtype: type = np.complex128


@dataclass(frozen=True)
class PMLConfig:
    """
    Perfectly Matched Layer (PML) configuration for absorbing boundaries.
    """
    thickness: int = 8        # number of grid cells
    m: int = 3                # polynomial exponent
    sigma_max: float = 50.0   # maximum damping coefficient


@dataclass(frozen=True)
class SolverOptions:
    """
    Options for GMRES and direct solvers.
    """
    tol: float = 1e-6
    restart: int = 200
    maxiter: Optional[int] = None





# ----------------------------------------------------------------------
# === Project-level constants ===
# ----------------------------------------------------------------------

# Default numeric precision
DTYPE = np.complex128

# Default directories (created on import)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# === Export list ===
# ----------------------------------------------------------------------

__all__ = [
    "GridSpec",
    "FDConfig",
    "PMLConfig",
    "SolverOptions",
    "DTYPE",
    "DATA_DIR",
    "RESULTS_DIR",
]
