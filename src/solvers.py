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

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        System matrix.
    b : np.ndarray
        Right-hand side (flattened).
    options : SolverOptions | None
        Solver tolerances; defaults to tol=1e-6, restart=200, maxiter=None.
    x0 : np.ndarray | None
        Initial guess.
    M : LinearOperator | None
        (Optional) Preconditioner.

    Returns
    -------
    SolverResult
    """
    if options is None:
        options = SolverOptions()

    b = np.asarray(b).astype(np.complex128, copy=False)
    b_norm = np.linalg.norm(b) or 1.0

    rel_hist: list[float] = []

    def _callback(rk):
        # SciPy may pass a residual vector or a scalar norm depending on version
        try:
            rk_norm = float(np.linalg.norm(rk))
        except Exception:
            rk_norm = float(rk)
        rel_hist.append(rk_norm / b_norm)

    # Use legacy 'tol' for widest SciPy compatibility
    x, info = _gmres(
        A,
        b,
        x0=x0,
        M=M,
        tol=options.tol,
        restart=options.restart,
        maxiter=options.maxiter,
        callback=_callback,
    )

    # Ensure we have at least the final residual if callback didn't fire
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
