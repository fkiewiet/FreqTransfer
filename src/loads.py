"""
loads.py
--------
Defines source terms (right-hand sides) for Helmholtz-type problems.
Each load is described by a simple dataclass and constructed on a given grid.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import GridSpec


# ----------------------------------------------------------------------
# === Load type definitions ===
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PointSource:
    """
    A complex point source at a specific location.

    Parameters
    ----------
    location : tuple[float, ...]
        Physical coordinates of the source within the domain.
    amplitude : complex
        Source amplitude.
    phase : float
        Phase angle [radians].
    sigma_h : float | None
        If provided, spreads the source into a Gaussian with
        standard deviation = sigma_h * min(grid spacing).
    """
    location: Tuple[float, ...]
    amplitude: complex = 1.0 + 0j
    phase: float = 0.0
    sigma_h: Optional[float] = None


@dataclass(frozen=True)
class PlaneWave:
    """
    Continuous plane-wave load.

    Parameters
    ----------
    kvec : tuple[float, ...]
        Wave vector components [1/m].
    amplitude : complex
        Amplitude of the wave.
    phase : float
        Global phase offset [radians].
    """
    kvec: Tuple[float, ...]
    amplitude: complex = 1.0 + 0j
    phase: float = 0.0


@dataclass(frozen=True)
class RandomPointSource:
    """
    Randomized point source with random amplitude, phase, and position.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    amp_range : tuple[float, float]
        Range for uniform amplitude sampling.
    phase_range : tuple[float, float]
        Range for uniform phase sampling (radians).
    """
    seed: int = 0
    amp_range: Tuple[float, float] = (0.5, 1.5)
    phase_range: Tuple[float, float] = (0.0, 2 * np.pi)


# ----------------------------------------------------------------------
# === RHS builder ===
# ----------------------------------------------------------------------

def build_load(spec, grid: GridSpec) -> np.ndarray:
    """
    Builds the complex right-hand side (flattened vector) for a given source spec.

    Parameters
    ----------
    spec : PointSource | PlaneWave | RandomPointSource
        Source specification.
    grid : GridSpec
        Grid on which to define the RHS.

    Returns
    -------
    b : np.ndarray
        Flattened complex RHS vector (shape = grid.N, dtype = complex128).
    """
    # Mesh coordinates
    axes = [np.linspace(0.0, L, n) for L, n in zip(grid.lengths, grid.shape)]
    X = np.meshgrid(*axes, indexing="ij")
    b = np.zeros(grid.shape, dtype=np.complex128)

    # --- Point source ---
    if isinstance(spec, PointSource):
        loc = spec.location
        amp = spec.amplitude * np.exp(1j * spec.phase)

        if spec.sigma_h is None:
            # nearest grid node
            idx = tuple(int(round(x / L * (n - 1))) for x, L, n in zip(loc, grid.lengths, grid.shape))
            b[idx] += amp
        else:
            # Gaussian smearing
            hs = grid.spacing
            sigma = spec.sigma_h * min(hs)
            r2 = sum((Xi - loc[d]) ** 2 for d, Xi in enumerate(X))
            g = np.exp(-0.5 * r2 / sigma**2)
            b += amp * g / g.sum()
        return b.ravel()

    # --- Plane wave ---
    if isinstance(spec, PlaneWave):
        arg = sum(k * Xi for k, Xi in zip(spec.kvec, X)) + spec.phase
        b = spec.amplitude * np.exp(1j * arg)
        return b.ravel()

    # --- Randomized point source ---
    if isinstance(spec, RandomPointSource):
        rng = np.random.default_rng(spec.seed)
        amp = rng.uniform(*spec.amp_range)
        phase = rng.uniform(*spec.phase_range)
        loc = tuple(rng.uniform(0.0, L) for L in grid.lengths)
        return build_load(PointSource(loc, amp * np.exp(1j * phase)), grid)

    raise TypeError(f"Unsupported load specification: {type(spec)}")


# ----------------------------------------------------------------------
# === Public symbols ===
# ----------------------------------------------------------------------

__all__ = [
    "PointSource",
    "PlaneWave",
    "RandomPointSource",
    "build_load",
]
