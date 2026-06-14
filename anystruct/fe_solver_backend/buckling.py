"""Linear eigenvalue buckling helpers.

The first production buckling path is intentionally modest: element routines
assemble a reference geometric stiffness matrix ``KG`` and this module solves
the constrained generalized problem

    K phi = lambda KG phi

Positive eigenvalues are critical load multipliers relative to the supplied
reference compression state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .assembly import build_constraint_transformation
from .matrix_assembly import assemble_geometric_stiffness_matrix, assemble_stiffness_matrix

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass
class BucklingMode:
    """One positive eigenvalue buckling mode."""

    mode_number: int
    load_factor: float
    eigenvalue: float
    mode_shape: np.ndarray
    reduced_mode_shape: np.ndarray
    modal_stiffness: float
    modal_geometric_stiffness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_number": self.mode_number,
            "load_factor": self.load_factor,
            "eigenvalue": self.eigenvalue,
            "modal_stiffness": self.modal_stiffness,
            "modal_geometric_stiffness": self.modal_geometric_stiffness,
            "mode_shape": self.mode_shape.tolist(),
        }


@dataclass
class BucklingResult:
    """Result bundle from the linear eigenvalue buckling solve."""

    modes: List[BucklingMode]
    num_modes_requested: int
    solver_status: str
    constraint_info: Dict[str, Any]
    assembly_info: Dict[str, Any]

    @property
    def num_modes_returned(self) -> int:
        return len(self.modes)

    @property
    def critical_load_factor(self) -> Optional[float]:
        if not self.modes:
            return None
        return self.modes[0].load_factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_status": self.solver_status,
            "num_modes_requested": self.num_modes_requested,
            "num_modes_returned": self.num_modes_returned,
            "critical_load_factor": self.critical_load_factor,
            "constraint_info": self.constraint_info,
            "assembly_info": self.assembly_info,
            "modes": [mode.to_dict() for mode in self.modes],
        }


def _as_symmetric_dense(matrix: sparse.spmatrix) -> np.ndarray:
    dense = np.asarray(matrix.toarray(), dtype=float)
    return 0.5 * (dense + dense.T)


def _normalize_mode(full_mode: np.ndarray, reduced_mode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = float(np.max(np.abs(full_mode))) if full_mode.size else 0.0
    if scale <= 0.0:
        return full_mode, reduced_mode
    return full_mode / scale, reduced_mode / scale


def solve_eigenvalue_buckling(
    model: "FEModel",
    element_states: Optional[Any] = None,
    num_modes: int = 3,
    eigen_tolerance: float = 1.0e-8,
    dense_size_limit: int = 200,
) -> BucklingResult:
    """Solve ``K phi = lambda KG phi`` for positive buckling factors.

    ``element_states`` is passed to
    :func:`fe_solver.matrix_assembly.assemble_geometric_stiffness_matrix`.
    Beam elements accept axial reference compression and shell elements accept
    membrane resultant prestress (compression positive, or tension-positive
    ``membrane_forces``).
    """
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    K, stiffness_info = assemble_stiffness_matrix(model)
    KG, geometric_info = assemble_geometric_stiffness_matrix(model, element_states)
    zero_load = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    K_red, _, T, _, _, constraint_info = build_constraint_transformation(K, zero_load, model)
    KG_red = (T.T @ KG @ T).tocsr()

    assembly_info = {
        "stiffness": stiffness_info,
        "geometric_stiffness": geometric_info,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(K_red.shape[0]),
    }

    if K_red.shape[0] == 0:
        return BucklingResult([], num_modes, "empty_reduced_system", constraint_info, assembly_info)
    if KG_red.nnz == 0:
        return BucklingResult([], num_modes, "zero_geometric_stiffness", constraint_info, assembly_info)

    # Work with the inverted symmetric pencil KG phi = mu K phi where K is
    # positive definite after constraint elimination.  The largest mu
    # correspond to the smallest positive load factors lambda = 1/mu, and the
    # symmetric formulation lets both the Lanczos and the dense solver work on
    # well-posed problems (KG itself may be indefinite or singular).
    n_red = int(K_red.shape[0])
    K_sym = (0.5 * (K_red + K_red.T)).tocsr()
    KG_sym = (0.5 * (KG_red + KG_red.T)).tocsr()
    k = min(max(num_modes * 4, num_modes + 2), n_red - 1)

    eigenvectors = None
    if n_red > dense_size_limit and 1 <= k < n_red:
        try:
            _, eigenvectors = sparse_linalg.eigsh(KG_sym.tocsc(), k=k, M=K_sym.tocsc(), which="LA")
        except Exception:
            eigenvectors = None
    if eigenvectors is None:
        K_dense = _as_symmetric_dense(K_red)
        KG_dense = _as_symmetric_dense(KG_red)
        try:
            _, eigenvectors = linalg.eigh(KG_dense, K_dense)
        except linalg.LinAlgError:
            # Singular K (e.g. unconstrained model): fall back to the general
            # nonsymmetric pencil and let the Rayleigh filtering sort it out.
            _, eigenvectors = linalg.eig(K_dense, KG_dense)

    candidates: List[tuple[float, np.ndarray, float, float]] = []
    for i in range(eigenvectors.shape[1]):
        reduced_mode = np.asarray(np.real(eigenvectors[:, i]), dtype=float)
        mode_norm = float(np.linalg.norm(reduced_mode))
        if not np.isfinite(mode_norm) or mode_norm <= 0.0:
            continue
        reduced_mode = reduced_mode / mode_norm
        modal_geometric = float(reduced_mode @ (KG_sym @ reduced_mode))
        modal_stiffness = float(reduced_mode @ (K_sym @ reduced_mode))
        if modal_geometric <= eigen_tolerance or modal_stiffness <= 0.0:
            continue
        rayleigh_value = modal_stiffness / modal_geometric
        if rayleigh_value <= eigen_tolerance or not np.isfinite(rayleigh_value):
            continue
        candidates.append((rayleigh_value, reduced_mode, modal_stiffness, modal_geometric))

    candidates.sort(key=lambda item: item[0])
    modes: List[BucklingMode] = []
    for mode_number, (value, reduced_mode, modal_stiffness, modal_geometric) in enumerate(
        candidates[:num_modes],
        start=1,
    ):
        full_mode = np.asarray(T @ reduced_mode, dtype=float).reshape(-1)
        full_mode, reduced_mode = _normalize_mode(full_mode, reduced_mode)
        modes.append(
            BucklingMode(
                mode_number=mode_number,
                load_factor=float(value),
                eigenvalue=float(value),
                mode_shape=full_mode,
                reduced_mode_shape=reduced_mode,
                modal_stiffness=modal_stiffness,
                modal_geometric_stiffness=modal_geometric,
            )
        )

    status = "ok" if modes else "no_positive_modes"
    return BucklingResult(modes, num_modes, status, constraint_info, assembly_info)
