"""Sparse/dense free-vibration modal analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from .assembly import build_constraint_transformation, build_reduced_rigid_body_modes
from .cases import make_result_case
from .linalg import FactorizationCache, MatrixClass, cached_inverse_operator
from .matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass
class ModalMode:
    """One free-vibration mode."""

    mode_number: int
    eigenvalue: float
    angular_frequency: float
    frequency_hz: float
    period: Optional[float]
    mode_shape: np.ndarray
    reduced_mode_shape: np.ndarray
    modal_mass: float
    modal_stiffness: float
    residual_norm: float
    rigid_body_correlation: float
    is_rigid_body: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_number": int(self.mode_number),
            "eigenvalue": float(self.eigenvalue),
            "angular_frequency": float(self.angular_frequency),
            "frequency_hz": float(self.frequency_hz),
            "period": None if self.period is None else float(self.period),
            "mode_shape": self.mode_shape.tolist(),
            "modal_mass": float(self.modal_mass),
            "modal_stiffness": float(self.modal_stiffness),
            "residual_norm": float(self.residual_norm),
            "rigid_body_correlation": float(self.rigid_body_correlation),
            "is_rigid_body": bool(self.is_rigid_body),
        }


@dataclass
class ModalResult:
    """Result bundle from modal analysis."""

    modes: List[ModalMode]
    num_modes_requested: int
    solver_status: str
    constraint_info: Dict[str, Any]
    nullspace_info: Dict[str, Any]
    assembly_info: Dict[str, Any]
    diagnostics: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None

    @property
    def num_modes_returned(self) -> int:
        return len(self.modes)

    @property
    def frequencies_hz(self) -> np.ndarray:
        return np.asarray([mode.frequency_hz for mode in self.modes], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_status": self.solver_status,
            "num_modes_requested": int(self.num_modes_requested),
            "num_modes_returned": int(self.num_modes_returned),
            "frequencies_hz": self.frequencies_hz.tolist(),
            "constraint_info": self.constraint_info,
            "nullspace_info": self.nullspace_info,
            "assembly_info": self.assembly_info,
            "diagnostics": self.diagnostics,
            "result_case": self.result_case,
            "modes": [mode.to_dict() for mode in self.modes],
        }


def _sym(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    return (0.5 * (matrix + matrix.T)).tocsr()


def _dense_eigensolve(K: sparse.spmatrix, M: sparse.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
    Kd = np.asarray(K.toarray(), dtype=float)
    Md = np.asarray(M.toarray(), dtype=float)
    Kd = 0.5 * (Kd + Kd.T)
    Md = 0.5 * (Md + Md.T)
    return linalg.eigh(Kd, Md)


def _sparse_eigensolve(
    K: sparse.spmatrix,
    M: sparse.spmatrix,
    num_modes: int,
    shift: Optional[float],
    factorization_cache: Optional[FactorizationCache] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = int(K.shape[0])
    k = min(max(num_modes + 4, num_modes), n - 1)
    if shift is None:
        values, vectors = sparse_linalg.eigsh(K.tocsc(), k=k, M=M.tocsc(), which="SM")
        return values, vectors, {"shift_invert": False}
    shift_matrix = (K - float(shift) * M).tocsc()
    cache = factorization_cache or FactorizationCache(name="modal_shift_invert", max_entries=2)
    operator, handle = cached_inverse_operator(
        shift_matrix,
        MatrixClass.SYMMETRIC_INDEFINITE,
        cache=cache,
    )
    values, vectors = sparse_linalg.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=float(shift), which="LM", OPinv=operator)
    return values, vectors, {
        "shift_invert": True,
        "shift_factorization": handle.diagnostics(),
        "factorization_cache": cache.diagnostics(),
    }


def _deterministic_sign(vector: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(vector))) if vector.size else 0
    if vector.size and vector[idx] < 0.0:
        return -vector
    return vector


def _orthogonality_error(modes: List[ModalMode], M_red: sparse.spmatrix) -> float:
    if not modes:
        return 0.0
    Phi = np.column_stack([mode.reduced_mode_shape for mode in modes])
    gram = np.asarray(Phi.T @ (M_red @ Phi), dtype=float)
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def solve_free_vibration(
    model: "FEModel",
    num_modes: int = 6,
    shift: Optional[float] = None,
    dense_size_limit: int = 200,
    eigen_tolerance: float = 1.0e-9,
    rigid_body_frequency_tolerance: float = 1.0e-6,
    factorization_cache: Optional[FactorizationCache] = None,
) -> ModalResult:
    """Solve ``K phi = omega^2 M phi`` with the common constraint transform."""
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    model.apply_boundary_conditions()
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    K_red, _, T, _, independent_dofs, constraint_info = build_constraint_transformation(K, zero, model)
    M_red = (T.T @ M @ T).tocsr()
    Q, nullspace_info = build_reduced_rigid_body_modes(model, independent_dofs, int(K.shape[0]))

    assembly_info = {
        "stiffness": stiffness_info,
        "mass": mass_info,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(K_red.shape[0]),
    }
    settings = {
        "num_modes": int(num_modes),
        "shift": None if shift is None else float(shift),
        "dense_size_limit": int(dense_size_limit),
        "eigen_tolerance": float(eigen_tolerance),
        "rigid_body_frequency_tolerance": float(rigid_body_frequency_tolerance),
        "factorization_cache": None if factorization_cache is None else factorization_cache.name,
    }

    if K_red.shape[0] == 0:
        diagnostics = {"status": "empty_reduced_system"}
        result_case = make_result_case(
            name="modal",
            analysis_type="modal",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
        ).to_dict()
        return ModalResult([], num_modes, "empty_reduced_system", constraint_info, nullspace_info, assembly_info, diagnostics, result_case)

    K_sym = _sym(K_red)
    M_sym = _sym(M_red)
    n_red = int(K_sym.shape[0])
    try:
        sparse_diagnostics: Dict[str, Any] = {}
        if n_red <= dense_size_limit or n_red <= num_modes + 1:
            eigenvalues, eigenvectors = _dense_eigensolve(K_sym, M_sym)
            solver_kind = "dense_scipy_eigh"
        else:
            eigenvalues, eigenvectors, sparse_diagnostics = _sparse_eigensolve(
                K_sym,
                M_sym,
                num_modes,
                shift,
                factorization_cache=factorization_cache,
            )
            solver_kind = "sparse_scipy_eigsh"
    except Exception as exc:
        diagnostics = {"status": "failed", "error": str(exc)}
        result_case = make_result_case(
            name="modal",
            analysis_type="modal",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
        ).to_dict()
        return ModalResult([], num_modes, "failed", constraint_info, nullspace_info, assembly_info, diagnostics, result_case)

    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])

    modes: List[ModalMode] = []
    for value, vector in zip(eigenvalues, eigenvectors.T):
        if len(modes) >= num_modes:
            break
        if not np.isfinite(value):
            continue
        reduced = np.asarray(vector, dtype=float).reshape(-1)
        modal_mass = float(reduced @ (M_sym @ reduced))
        if modal_mass <= eigen_tolerance:
            continue
        reduced = reduced / np.sqrt(modal_mass)
        reduced = _deterministic_sign(reduced)
        modal_mass = float(reduced @ (M_sym @ reduced))
        modal_stiffness = float(reduced @ (K_sym @ reduced))
        eig = max(float(value), 0.0) if abs(float(value)) <= eigen_tolerance else float(value)
        omega = float(np.sqrt(max(eig, 0.0)))
        frequency = omega / (2.0 * np.pi)
        residual = np.asarray(K_sym @ reduced - eig * (M_sym @ reduced), dtype=float).reshape(-1)
        denominator = max(float(np.linalg.norm(K_sym @ reduced)) + abs(eig) * float(np.linalg.norm(M_sym @ reduced)), 1.0)
        residual_norm = float(np.linalg.norm(residual) / denominator)
        rigid_corr = float(np.max(np.abs(Q.T @ reduced))) if Q.shape[1] else 0.0
        is_rigid = bool(frequency <= rigid_body_frequency_tolerance or rigid_corr > 0.90)
        full = np.asarray(T @ reduced, dtype=float).reshape(-1)
        modes.append(
            ModalMode(
                mode_number=len(modes) + 1,
                eigenvalue=eig,
                angular_frequency=omega,
                frequency_hz=frequency,
                period=None if frequency <= 0.0 else 1.0 / frequency,
                mode_shape=full,
                reduced_mode_shape=reduced,
                modal_mass=modal_mass,
                modal_stiffness=modal_stiffness,
                residual_norm=residual_norm,
                rigid_body_correlation=rigid_corr,
                is_rigid_body=is_rigid,
            )
        )

    status = "ok" if modes else "no_modes"
    diagnostics = {
        "status": status,
        "solver": solver_kind,
        **sparse_diagnostics,
        "max_residual_norm": max((mode.residual_norm for mode in modes), default=0.0),
        "mass_orthogonality_error": _orthogonality_error(modes, M_sym),
        "num_rigid_body_modes": int(sum(1 for mode in modes if mode.is_rigid_body)),
    }
    result_case = make_result_case(
        name="modal",
        analysis_type="modal",
        assembly_info=assembly_info,
        solver_info={"convergence_info": diagnostics},
        recovery={"modes": num_modes, "num_modes_returned": len(modes)},
        settings=settings,
    ).to_dict()
    return ModalResult(modes, num_modes, status, constraint_info, nullspace_info, assembly_info, diagnostics, result_case)
