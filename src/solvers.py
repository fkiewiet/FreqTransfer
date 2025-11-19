"""
solvers.py
----------
Linear solvers for Helmholtz-type systems.

Provides:
- SolverResult: container for results and residual history
- gmres_solve: GMRES with **relative** residual tracking
- direct_solve: sparse LU (via splu)
- direct_solve_auto: convenience helper to assemble + direct-solve Helmholtz
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import gmres as _gmres, splu

from .config import GridSpec, FDConfig, PMLConfig, SolverOptions
from .operators import assemble_operator


# ----------------------------------------------------------------------
# === Result container ===
# ----------------------------------------------------------------------

@dataclass
class SolverResult:
    """
    Container for linear solver results.

    Attributes
    ----------
    solution : np.ndarray
        The computed solution vector (flattened).
    residuals : list[float]
        History of **relative** residual norms per iteration.
    converged : bool
        True if the solver reports convergence.
    info : int
        SciPy info code (0 = success; >0 = no. of iterations; <0 = breakdown).
    """
    solution: np.ndarray
    residuals: List[float]
    converged: bool
    info: int


# ----------------------------------------------------------------------
# === GMRES wrapper with relative residual history ===
# ----------------------------------------------------------------------

def gmres_solve(
    A,
    b: np.ndarray,
    options: Optional[SolverOptions] = None,
    x0: Optional[np.ndarray] = None,
    M=None,
) -> SolverResult:
    """
    Solve Ax = b using GMRES and record **relative** residuals.
    Compatible with both legacy (tol=) and new (rtol=/atol=) SciPy APIs.
    """
    if options is None:
        options = SolverOptions()

    b = np.asarray(b).astype(np.complex128, copy=False)
    b_norm = np.linalg.norm(b) or 1.0

    rel_hist: list[float] = []

    def _callback(rk):
        # SciPy may pass a vector or a scalar norm
        try:
            rk_norm = float(np.linalg.norm(rk))
        except Exception:
            rk_norm = float(rk)
        rel_hist.append(rk_norm / b_norm)

    # Try legacy API first (tol=); fallback to new API (rtol=/atol=, callback_type)
    try:
        x, info = _gmres(
            A, b,
            x0=x0, M=M,
            tol=options.tol,                      # legacy
            restart=options.restart,
            maxiter=options.maxiter,
            callback=_callback,
        )
    except TypeError:
        # New SciPy API
        kw = dict(
            x0=x0, M=M,
            rtol=options.tol, atol=0.0,          # new API
            restart=options.restart,
            maxiter=options.maxiter,
            callback=_callback,
        )
        try:
            # Prefer explicit callback_type if supported
            x, info = _gmres(A, b, callback_type="pr_norm", **kw)
        except TypeError:
            x, info = _gmres(A, b, **kw)

    if not rel_hist:
        rel = np.linalg.norm(A @ x - b) / b_norm
        rel_hist.append(float(rel))

    return SolverResult(
        solution=x,
        residuals=rel_hist,
        converged=(info == 0),
        info=info,
    )



# ----------------------------------------------------------------------
# === Direct sparse solve (LU) ===
# ----------------------------------------------------------------------

def direct_solve(A, b: np.ndarray) -> SolverResult:
    """
    Solve Ax = b with sparse LU (splu). Best for moderate sizes or as a reference.

    Parameters
    ----------
    A : sparse matrix
        System matrix (will be converted to CSC if needed).
    b : np.ndarray
        Right-hand side (flattened).

    Returns
    -------
    SolverResult
    """
    if not issparse(A) or not isinstance(A, csc_matrix):
        A = csc_matrix(A)

    b = np.asarray(b).astype(np.complex128, copy=False)
    lu = splu(A)
    x = lu.solve(b)

    return SolverResult(
        solution=x,
        residuals=[],     # direct solve doesn't iterate
        converged=True,
        info=0,
    )


# ----------------------------------------------------------------------
# === High-level convenience: assemble + direct solve ===
# ----------------------------------------------------------------------

def direct_solve_auto(
    shape: tuple[int, ...],
    lengths: tuple[float, ...],
    k: float,
    rhs: np.ndarray,
    use_pml: bool = False,
    pml: Optional[PMLConfig] = None,
    fd: Optional[FDConfig] = None,
    m: Optional[np.ndarray] = None,
) -> SolverResult:
    """
    Assemble a Helmholtz operator for the given geometry and directly solve it.

    Parameters
    ----------
    shape : tuple[int, ...]
        Grid shape, e.g. (60, 60).
    lengths : tuple[float, ...]
        Physical lengths, e.g. (1.0, 1.0).
    k : float
        Wavenumber |k|.
    rhs : np.ndarray
        Flattened complex RHS (length = prod(shape)).
    use_pml : bool
        If True, use a default PML unless 'pml' is provided.
    pml : PMLConfig | None
        Explicit PML configuration.
    fd : FDConfig | None
        Finite-difference configuration (BCs, dtype).
    m : np.ndarray | None
        Optional inhomogeneous slowness-squared field m(x).

    Returns
    -------
    SolverResult
    """
    grid = GridSpec(dims=len(shape), shape=shape, lengths=lengths)
    if use_pml and pml is None:
        pml = PMLConfig()

    A = assemble_operator(grid=grid, k=k, kind="helmholtz", fd=fd, pml=pml, m=m)
    return direct_solve(A, rhs)



def solve_with_pml_shell(
    grid_phys: GridSpec,
    k: float,
    b_phys: np.ndarray,
    pml: PMLConfig,
    kind: str = "helmholtz",
):
    """
    Solve Helmholtz on an *extended* grid where the PML shell lives
    outside the physical domain.

    - grid_phys: original physical grid (e.g. 48x48)
    - b_phys: RHS on the physical grid, flattened with shape (grid_phys.N,)

    Returns:
        u_phys_flat: solution on the interior (physical) grid, flattened
        u_ext_flat: full extended solution (optional to use / visualize)
    """
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

    # 4. Solve
    res_ext = direct_solve(A_ext, b_ext)
    u_ext_2d = res_ext.solution.reshape(grid_ext.shape)

    # 5. Crop interior block back to physical domain
    u_phys_2d = u_ext_2d[T:T+ny, T:T+nx]
    u_phys = u_phys_2d.ravel()

    return u_phys, u_ext_2d, grid_ext
