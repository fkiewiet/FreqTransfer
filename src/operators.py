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
    'h' is currently unused (profile is in index space), kept for future
    physical-coordinate variants.
    """
    sigma = np.zeros(n)
    if thickness <= 0:
        return sigma
    thickness = min(thickness, n // 2)  # safety: don't overlap the whole domain
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
        L_k u = -Δu - k² m(x) u

    Optionally applies PML damping by modifying the Laplacian coefficients
    via a diagonal complex scaling (approximate coordinate stretching).
    """
    if fd is None:
        fd = FDConfig()
    if m is None:
        m = np.ones(grid.shape, dtype=float)
    else:
        m = np.asarray(m, dtype=float).reshape(grid.shape)

    # Base Laplacian
    L = laplacian_operator(grid, fd)

    # --- PML modification (approximate) ---
    if pml is not None and pml.thickness > 0 and pml.sigma_max > 0.0:
        sigmas = _pml_stretching(grid, pml)  # tuple of 1D arrays, one per axis

        # Build index grid once: coords[axis] has shape grid.shape and holds indices along that axis
        coords = np.meshgrid(
            *[np.arange(n) for n in grid.shape],
            indexing="ij",
        )

        # Start with unit damping everywhere
        damp = np.ones(grid.shape, dtype=fd.dtype)

        # For each axis, compute (1 + i σ/k)^-1 and multiply into the full-grid factor
        for axis, sigma_1d in enumerate(sigmas):
            factor_1d = 1.0 / (1.0 + 1j * sigma_1d / k)
            damp *= factor_1d[coords[axis]]

        # Diagonal scaling of the Laplacian
        D = diags(damp.ravel(), 0, format="csc", dtype=fd.dtype)
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


def make_extended_grid(grid_phys: GridSpec, pml: PMLConfig) -> GridSpec:
    """
    Create a grid that adds a PML 'shell' of thickness pml.thickness
    *around* the physical grid.

    - The physical domain spacing is preserved.
    - The interior block of the extended grid corresponds to the original
      physical domain [0, Lx] x [0, Ly].
    """
    T = pml.thickness
    ny, nx = grid_phys.shape
    Ly, Lx = grid_phys.lengths
    hy, hx = grid_phys.spacing  # assuming GridSpec exposes this

    # New shape: interior + PML shell on both sides
    ny_ext = ny + 2 * T
    nx_ext = nx + 2 * T

    # Extend physical lengths so that spacing stays the same:
    # Lx_ext = Lx + 2*T*hx  ⇒ hx_ext = hx
    Ly_ext = Ly + 2 * T * hy
    Lx_ext = Lx + 2 * T * hx

    grid_ext = GridSpec(
        dims=grid_phys.dims,
        shape=(ny_ext, nx_ext),
        lengths=(Ly_ext, Lx_ext),
    )
    return grid_ext


# ----------------------------------------------------------------------
# === Public symbols ===
# ----------------------------------------------------------------------

__all__ = [
    "laplacian_operator",
    "helmholtz_operator",
    "assemble_operator",
    "make_extended_grid",
    
]
