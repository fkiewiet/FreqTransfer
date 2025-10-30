"""
operators.py
-------------
Constructs finite-difference operators for Helmholtz problems.
Includes support for:
  - standard Laplacian operator
  - Helmholtz operator (-Δ - k² m(x))
  - optional PML (Perfectly Matched Layer) damping
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix

from .config import GridSpec, FDConfig, PMLConfig


# ----------------------------------------------------------------------
# === Utility: 1D finite-difference Laplacian ===
# ----------------------------------------------------------------------

def _lap1d(n: int, h: float, bc: str = "dirichlet") -> csc_matrix:
    """1D Laplacian with Dirichlet or Neumann BC."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    A = diags([off, main, off], [-1, 0, 1], shape=(n, n), dtype=float).tocsc()

    if bc == "neumann":
        A[0, 0] = -1.0
        A[-1, -1] = -1.0
    elif bc != "dirichlet":
        raise ValueError(f"Unknown boundary condition: {bc}")

    return A / (h**2)


# ----------------------------------------------------------------------
# === Multi-dimensional Laplacian ===
# ----------------------------------------------------------------------

def laplacian_operator(grid: GridSpec, fd: FDConfig) -> csc_matrix:
    """
    Construct N-D Laplacian with Kronecker products.
    Supports up to 3D.
    """
    shape = grid.shape
    spacings = grid.spacing
    bc = fd.bc

    if grid.dims == 1:
        return _lap1d(shape[0], spacings[0], bc)

    # Start with first dimension
    L = _lap1d(shape[0], spacings[0], bc)
    I = eye(shape[0], format="csc")

    for dim in range(1, grid.dims):
        Ld = _lap1d(shape[dim], spacings[dim], bc)
        # Expand dimensionality using Kronecker sums
        L = kron(L, eye(shape[dim], format="csc")) + kron(I, Ld)
        I = kron(I, eye(shape[dim], format="csc"))

    return L.astype(fd.dtype)


# ----------------------------------------------------------------------
# === PML helper: compute σ(x) profile ===
# ----------------------------------------------------------------------

def _pml_profile(n: int, h: float, thickness: int, m: int, sigma_max: float) -> np.ndarray:
    """
    Generate 1D polynomial PML profile σ(x) increasing toward boundaries.
    """
    sigma = np.zeros(n)
    for i in range(thickness):
        s = (1 - (i + 0.5) / thickness) ** m
        sigma[i] = sigma_max * s
        sigma[-i - 1] = sigma_max * s
    return sigma


def _pml_stretching(grid: GridSpec, pml: PMLConfig) -> tuple[np.ndarray, ...]:
    """
    Compute complex coordinate stretching factors (1 + i σ / ω)⁻¹ for each axis.
    """
    sigmas = []
    for n, h in zip(grid.shape, grid.spacing):
        sigmas.append(_pml_profile(n, h, pml.thickness, pml.m, pml.sigma_max))
    return tuple(sigmas)


# ----------------------------------------------------------------------
# === Helmholtz operator (with optional PML) ===
# ----------------------------------------------------------------------

def helmholtz_operator(
    grid: GridSpec,
    k: float,
    fd: Optional[FDConfig] = None,
    pml: Optional[PMLConfig] = None,
    m: Optional[np.ndarray] = None,
) -> csc_matrix:
    """
    Assemble the Helmholtz operator:
        L_ω u = -Δu - k² m(x) u
    Optionally applies PML damping by modifying the Laplacian coefficients.
    """
    if fd is None:
        fd = FDConfig()
    if m is None:
        m = np.ones(grid.shape, dtype=float)

    L = laplacian_operator(grid, fd)

    # --- PML modification ---
    if pml is not None:
        sigmas = _pml_stretching(grid, pml)
        # For each axis, rescale derivatives by (1 + iσ/k)^-1
        # Approximate multiplicative damping for simplicity
        damp = np.ones(grid.N, dtype=fd.dtype)
        for sigma in sigmas:
            factor = 1.0 / (1.0 + 1j * sigma / k)
            # broadcast across that axis
            shape_full = [1] * grid.dims
            axis = len(sigmas) - len(shape_full)
            shape_full[axis] = len(sigma)
            damp_axis = factor.reshape(shape_full)
            damp *= np.broadcast_to(damp_axis, grid.shape).ravel()
        # Scale Laplacian entries (diagonal scaling)
        D = diags(damp, 0, format="csc", dtype=fd.dtype)
        L = D @ L @ D

    # --- Add -k² m(x) term ---
    M = diags((k**2) * m.ravel(), 0, format="csc", dtype=fd.dtype)
    A = (-L - M).astype(fd.dtype)

    return A


# ----------------------------------------------------------------------
# === High-level wrapper (for backward compatibility) ===
# ----------------------------------------------------------------------

def assemble_operator(
    grid: GridSpec,
    k: float,
    kind: str = "helmholtz",
    fd: Optional[FDConfig] = None,
    pml: Optional[PMLConfig] = None,
    m: Optional[np.ndarray] = None,
) -> csc_matrix:
    """
    Unified operator assembler.
    Supported kinds: 'laplacian', 'helmholtz'.
    """
    if kind.lower() == "laplacian":
        return laplacian_operator(grid, fd or FDConfig())
    if kind.lower() == "helmholtz":
        return helmholtz_operator(grid, k, fd=fd, pml=pml, m=m)
    raise ValueError(f"Unsupported operator kind: {kind!r}")


# ----------------------------------------------------------------------
# === Public symbols ===
# ----------------------------------------------------------------------

__all__ = [
    "laplacian_operator",
    "helmholtz_operator",
    "assemble_operator",
]
