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


# ---------------------------------------------------------------------------
# PML shell helpers: extend grid and crop interior
# ---------------------------------------------------------------------------

def make_extended_grid(grid_phys: GridSpec, pml: PMLConfig) -> GridSpec:
    """
    Create an extended grid that adds a PML 'shell' of thickness pml.thickness
    around the physical grid.

    The interior block of the extended grid corresponds to the original
    physical domain with the same spacing; the PML cells live outside.
    """
    T = pml.thickness
    ny, nx = grid_phys.shape
    Ly, Lx = grid_phys.lengths
    hy, hx = grid_phys.spacing  # assumes GridSpec exposes spacing

    ny_ext = ny + 2 * T
    nx_ext = nx + 2 * T
    Ly_ext = Ly + 2 * T * hy
    Lx_ext = Lx + 2 * T * hx

    grid_ext = GridSpec(
        dims=grid_phys.dims,
        shape=(ny_ext, nx_ext),
        lengths=(Ly_ext, Lx_ext),
    )
    return grid_ext


def solve_with_pml_shell(
    grid_phys: GridSpec,
    k: float,
    b_phys: np.ndarray,
    pml: PMLConfig,
    kind: str = "helmholtz",
):
    """
    Solve Helmholtz/Poisson on an extended grid with a PML shell outside
    the physical domain.

    Parameters
    ----------
    grid_phys : GridSpec
        Physical domain grid (e.g. 48x48 or 96x96).
    k : float
        Wavenumber.
    b_phys : np.ndarray
        Right-hand side on the physical grid (flattened, size = grid_phys.N).
    pml : PMLConfig
        PML configuration (thickness, order, sigma_max).
    kind : {"helmholtz", "laplacian"}
        Operator kind to assemble.

    Returns
    -------
    u_phys_flat : np.ndarray
        Solution restricted to the physical grid, flattened (size = grid_phys.N).
    u_ext_2d : np.ndarray
        Solution on the extended grid, 2D array of shape grid_ext.shape.
    grid_ext : GridSpec
        Extended grid specification.
    """
    # 🔹 IMPORTANT: local import to avoid circular import problems
    from .solvers import direct_solve

    T = pml.thickness
    ny, nx = grid_phys.shape

    # 1. Build extended grid
    grid_ext = make_extended_grid(grid_phys, pml)
    ny_ext, nx_ext = grid_ext.shape

    # 2. Embed RHS into extended grid (zero outside physical domain)
    b_ext_2d = np.zeros(grid_ext.shape, dtype=b_phys.dtype)
    b_ext_2d[T:T+ny, T:T+nx] = b_phys.reshape(grid_phys.shape)
    b_ext = b_ext_2d.ravel()

    # 3. Assemble operator with PML on the extended grid
    A_ext = assemble_operator(grid_ext, k=k, kind=kind, pml=pml)

    # 4. Direct solve on the extended grid
    res_ext = direct_solve(A_ext, b_ext)
    u_ext_2d = res_ext.solution.reshape(grid_ext.shape)

    # 5. Crop interior block back to physical grid
    u_phys_2d = u_ext_2d[T:T+ny, T:T+nx]
    u_phys = u_phys_2d.ravel()

    return u_phys, u_ext_2d, grid_ext
