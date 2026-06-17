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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .assembly import build_constraint_transformation
from .cases import make_result_case
from .linalg import FactorizationCache, MatrixClass, cached_inverse_operator
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
    residual_norm: float = 0.0
    validity_status: str = "ok"
    repeated_group: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_number": self.mode_number,
            "load_factor": self.load_factor,
            "eigenvalue": self.eigenvalue,
            "modal_stiffness": self.modal_stiffness,
            "modal_geometric_stiffness": self.modal_geometric_stiffness,
            "residual_norm": self.residual_norm,
            "validity_status": self.validity_status,
            "repeated_group": self.repeated_group,
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
    result_case: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None

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
            "result_case": self.result_case,
            "diagnostics": self.diagnostics or {},
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


def _mode_residual(
    K: sparse.spmatrix,
    KG: sparse.spmatrix,
    reduced_mode: np.ndarray,
    load_factor: float,
) -> float:
    residual = np.asarray(K @ reduced_mode - float(load_factor) * (KG @ reduced_mode), dtype=float).reshape(-1)
    denominator = max(
        float(np.linalg.norm(K @ reduced_mode))
        + abs(float(load_factor)) * float(np.linalg.norm(KG @ reduced_mode)),
        1.0,
    )
    return float(np.linalg.norm(residual) / denominator)


def _factor_in_range(value: float, load_factor_range: Optional[Tuple[Optional[float], Optional[float]]]) -> bool:
    if load_factor_range is None:
        return True
    lower, upper = load_factor_range
    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def _sort_key(value: float, shift_load_factor: Optional[float]) -> Tuple[float, float]:
    if shift_load_factor is None:
        return (float(value), 0.0)
    return (abs(float(value) - float(shift_load_factor)), float(value))


def _assign_repeated_groups(modes: List[BucklingMode], tolerance: float) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not modes:
        return groups
    current = [modes[0]]
    group_index = 1
    for mode in modes[1:]:
        reference = current[0].load_factor
        relative = abs(mode.load_factor - reference) / max(abs(reference), 1.0)
        if relative <= tolerance:
            current.append(mode)
        else:
            if len(current) > 1:
                for item in current:
                    item.repeated_group = group_index
                groups.append(
                    {
                        "group": group_index,
                        "mode_numbers": [int(item.mode_number) for item in current],
                        "load_factors": [float(item.load_factor) for item in current],
                        "relative_spread": float(
                            (max(item.load_factor for item in current) - min(item.load_factor for item in current))
                            / max(abs(reference), 1.0)
                        ),
                    }
                )
                group_index += 1
            current = [mode]
    if len(current) > 1:
        reference = current[0].load_factor
        for item in current:
            item.repeated_group = group_index
        groups.append(
            {
                "group": group_index,
                "mode_numbers": [int(item.mode_number) for item in current],
                "load_factors": [float(item.load_factor) for item in current],
                "relative_spread": float(
                    (max(item.load_factor for item in current) - min(item.load_factor for item in current))
                    / max(abs(reference), 1.0)
                ),
            }
        )
    return groups


def solve_eigenvalue_buckling(
    model: "FEModel",
    element_states: Optional[Any] = None,
    num_modes: int = 3,
    eigen_tolerance: float = 1.0e-8,
    dense_size_limit: int = 200,
    shift_load_factor: Optional[float] = None,
    load_factor_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    search_factor: int = 4,
    repeated_tolerance: float = 1.0e-3,
    allow_dense_fallback: bool = False,
    factorization_cache: Optional[FactorizationCache] = None,
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

    settings = {
        "num_modes": num_modes,
        "dense_size_limit": dense_size_limit,
        "eigen_tolerance": eigen_tolerance,
        "shift_load_factor": shift_load_factor,
        "load_factor_range": None if load_factor_range is None else list(load_factor_range),
        "search_factor": search_factor,
        "repeated_tolerance": repeated_tolerance,
        "allow_dense_fallback": allow_dense_fallback,
        "factorization_cache": None if factorization_cache is None else factorization_cache.name,
    }

    if K_red.shape[0] == 0:
        diagnostics = {"status": "empty_reduced_system"}
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": "none" if element_states is None else type(element_states).__name__},
        ).to_dict()
        return BucklingResult([], num_modes, "empty_reduced_system", constraint_info, assembly_info, result_case, diagnostics)
    if KG_red.nnz == 0:
        diagnostics = {"status": "zero_geometric_stiffness"}
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": "none" if element_states is None else type(element_states).__name__},
        ).to_dict()
        return BucklingResult([], num_modes, "zero_geometric_stiffness", constraint_info, assembly_info, result_case, diagnostics)

    # Work with the inverted symmetric pencil KG phi = mu K phi where K is
    # positive definite after constraint elimination.  The largest mu
    # correspond to the smallest positive load factors lambda = 1/mu, and the
    # symmetric formulation lets both the Lanczos and the dense solver work on
    # well-posed problems (KG itself may be indefinite or singular).
    n_red = int(K_red.shape[0])
    K_sym = (0.5 * (K_red + K_red.T)).tocsr()
    KG_sym = (0.5 * (KG_red + KG_red.T)).tocsr()
    k = min(max(num_modes * max(int(search_factor), 1), num_modes + 2), n_red - 1)

    eigenvectors = None
    solver_kind = "not_started"
    sparse_error = None
    shift_invert_diagnostics: Dict[str, Any] = {"shift_invert": False}
    if n_red > dense_size_limit and 1 <= k < n_red:
        try:
            if shift_load_factor is None:
                _, eigenvectors = sparse_linalg.eigsh(KG_sym.tocsc(), k=k, M=K_sym.tocsc(), which="LA")
            else:
                sigma = 1.0 / float(shift_load_factor)
                shift_matrix = (KG_sym - sigma * K_sym).tocsc()
                cache = factorization_cache or FactorizationCache(name="buckling_shift_invert", max_entries=2)
                operator, handle = cached_inverse_operator(
                    shift_matrix,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    cache=cache,
                )
                _, eigenvectors = sparse_linalg.eigsh(
                    KG_sym.tocsc(),
                    k=k,
                    M=K_sym.tocsc(),
                    sigma=sigma,
                    which="LM",
                    OPinv=operator,
                )
                shift_invert_diagnostics = {
                    "shift_invert": True,
                    "shift_inverse_sigma": float(sigma),
                    "shift_factorization": handle.diagnostics(),
                    "factorization_cache": cache.diagnostics(),
                }
            solver_kind = "sparse_scipy_eigsh_inverted_pencil"
        except Exception as exc:
            sparse_error = str(exc)
            eigenvectors = None
    if eigenvectors is None and (n_red <= dense_size_limit or allow_dense_fallback):
        K_dense = _as_symmetric_dense(K_red)
        KG_dense = _as_symmetric_dense(KG_red)
        try:
            _, eigenvectors = linalg.eigh(KG_dense, K_dense)
            solver_kind = "dense_scipy_eigh_inverted_pencil"
        except linalg.LinAlgError:
            # Singular K (e.g. unconstrained model): fall back to the general
            # nonsymmetric pencil and let the Rayleigh filtering sort it out.
            _, eigenvectors = linalg.eig(K_dense, KG_dense)
            solver_kind = "dense_scipy_eig_general_pencil"
    if eigenvectors is None:
        diagnostics = {
            "status": "failed",
            "solver": solver_kind,
            "sparse_error": sparse_error,
            "reason": "sparse eigensolve failed and dense fallback is disabled for this reduced size",
        }
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": "none" if element_states is None else type(element_states).__name__},
        ).to_dict()
        return BucklingResult([], num_modes, "failed", constraint_info, assembly_info, result_case, diagnostics)

    candidates: List[tuple[float, np.ndarray, float, float, float]] = []
    rejected: List[Dict[str, Any]] = []
    for i in range(eigenvectors.shape[1]):
        reduced_mode = np.asarray(np.real(eigenvectors[:, i]), dtype=float)
        mode_norm = float(np.linalg.norm(reduced_mode))
        if not np.isfinite(mode_norm) or mode_norm <= 0.0:
            rejected.append({"root_index": int(i), "reason": "invalid_or_zero_norm"})
            continue
        reduced_mode = reduced_mode / mode_norm
        modal_geometric = float(reduced_mode @ (KG_sym @ reduced_mode))
        modal_stiffness = float(reduced_mode @ (K_sym @ reduced_mode))
        if modal_geometric <= eigen_tolerance or modal_stiffness <= 0.0:
            rejected.append(
                {
                    "root_index": int(i),
                    "reason": "nonpositive_modal_terms",
                    "modal_stiffness": float(modal_stiffness),
                    "modal_geometric_stiffness": float(modal_geometric),
                }
            )
            continue
        rayleigh_value = modal_stiffness / modal_geometric
        if rayleigh_value <= eigen_tolerance or not np.isfinite(rayleigh_value):
            rejected.append({"root_index": int(i), "reason": "invalid_load_factor", "load_factor": float(rayleigh_value)})
            continue
        if not _factor_in_range(rayleigh_value, load_factor_range):
            rejected.append({"root_index": int(i), "reason": "outside_load_factor_range", "load_factor": float(rayleigh_value)})
            continue
        residual_norm = _mode_residual(K_sym, KG_sym, reduced_mode, rayleigh_value)
        candidates.append((rayleigh_value, reduced_mode, modal_stiffness, modal_geometric, residual_norm))

    candidates.sort(key=lambda item: _sort_key(item[0], shift_load_factor))
    modes: List[BucklingMode] = []
    for mode_number, (value, reduced_mode, _modal_stiffness, _modal_geometric, _residual_norm) in enumerate(
        candidates[:num_modes],
        start=1,
    ):
        full_mode = np.asarray(T @ reduced_mode, dtype=float).reshape(-1)
        full_mode, reduced_mode = _normalize_mode(full_mode, reduced_mode)
        modal_stiffness = float(reduced_mode @ (K_sym @ reduced_mode))
        modal_geometric = float(reduced_mode @ (KG_sym @ reduced_mode))
        residual_norm = _mode_residual(K_sym, KG_sym, reduced_mode, value)
        validity = "ok" if residual_norm <= max(1.0e-6, 100.0 * eigen_tolerance) else "high_residual"
        modes.append(
            BucklingMode(
                mode_number=mode_number,
                load_factor=float(value),
                eigenvalue=float(value),
                mode_shape=full_mode,
                reduced_mode_shape=reduced_mode,
                modal_stiffness=modal_stiffness,
                modal_geometric_stiffness=modal_geometric,
                residual_norm=residual_norm,
                validity_status=validity,
            )
        )

    repeated_groups = _assign_repeated_groups(modes, repeated_tolerance)
    status = "ok" if modes else "no_positive_modes"
    diagnostics = {
        "status": status,
        "solver": solver_kind,
        **shift_invert_diagnostics,
        "sparse_error": sparse_error,
        "num_roots_considered": int(eigenvectors.shape[1]),
        "num_rejected_roots": int(len(rejected)),
        "rejected_roots": rejected[:50],
        "max_residual_norm": max((mode.residual_norm for mode in modes), default=0.0),
        "repeated_mode_groups": repeated_groups,
        "num_repeated_mode_groups": int(len(repeated_groups)),
        "sorting": "nearest_shift" if shift_load_factor is not None else "ascending_load_factor",
    }
    result_case = make_result_case(
        name="linear_buckling",
        analysis_type="linear_buckling",
        assembly_info=assembly_info,
        solver_info={"convergence_info": diagnostics},
        recovery={"modes": num_modes, "num_modes_returned": len(modes)},
        settings=settings,
        metadata={"prestress_state_source": "none" if element_states is None else type(element_states).__name__},
    ).to_dict()
    return BucklingResult(modes, num_modes, status, constraint_info, assembly_info, result_case, diagnostics)
