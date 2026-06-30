"""Validation helpers for FE solver verification tests and benchmarks.

The functions in this module are intentionally lightweight.  They do not define
new solver behaviour; they inspect assembled models, load vectors, solver
metadata and reconstructed displacement fields so tests can lock the solver
architecture before eigenvalue buckling and nonlinear work are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


@dataclass(frozen=True)
class LoadResultant:
    """Global resultant force and moment of a load vector."""

    force: np.ndarray
    moment: np.ndarray

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(self.force))

    @property
    def moment_norm(self) -> float:
        return float(np.linalg.norm(self.moment))


@dataclass(frozen=True)
class ShellPatchSummary:
    """Compact shell verification summary."""

    element_id: int
    strain_energy: float
    stiffness_symmetry_error: float
    max_membrane_spread: float
    max_bending_spread: float


@dataclass(frozen=True)
class ProductionValidationIssue:
    """Structured production model validation diagnostic."""

    code: str
    severity: str
    entity_type: str
    entity_id: Optional[int]
    message: str
    measured: Optional[float] = None
    limit: Optional[float] = None
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "message": self.message,
            "measured": self.measured,
            "limit": self.limit,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class ProductionValidationReport:
    """Production guardrail report for a model before analysis."""

    status: str
    issues: Tuple[ProductionValidationIssue, ...]
    mesh_quality: Dict[str, Any]
    revision_signature: Dict[str, int]

    @property
    def errors(self) -> Tuple[ProductionValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> Tuple[ProductionValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "mesh_quality": self.mesh_quality,
            "revision_signature": self.revision_signature,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def dof_order_signature(model: "FEModel") -> Dict[int, List[Tuple[int, str]]]:
    """Return node DOF indices with local DOF names.

    This is used by tests to protect the required ordering:
    ux, uy, uz, rx, ry, rz.
    """
    signature: Dict[int, List[Tuple[int, str]]] = {}
    dof_manager = model.mesh.dof_manager
    for node_id, node in model.mesh.nodes.items():
        entries: List[Tuple[int, str]] = []
        for dof in node.dofs:
            _node_id, _local_index, name = dof_manager.get_dof_info(dof)
            entries.append((int(dof), name))
        signature[int(node_id)] = entries
    return signature


def load_vector_resultant(model: "FEModel", load_vector: np.ndarray) -> LoadResultant:
    """Compute global force and moment resultants from a full load vector.

    The moment is taken about the model coordinate origin.
    """
    load_vector = np.asarray(load_vector, dtype=float).reshape(-1)
    force = np.zeros(3, dtype=float)
    moment = np.zeros(3, dtype=float)
    for node in model.mesh.nodes.values():
        node_force = load_vector[node.dofs[:3]]
        node_moment = load_vector[node.dofs[3:6]]
        r = node.coords()
        force += node_force
        moment += np.cross(r, node_force) + node_moment
    return LoadResultant(force=force, moment=moment)


def load_case_resultant(model: "FEModel", load_case: "LoadCase") -> LoadResultant:
    """Assemble a load case and return global force/moment resultants."""
    load_vector = load_case.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
    return load_vector_resultant(model, load_vector)


def mpc_constraint_residuals(model: "FEModel", displacements: np.ndarray) -> Dict[str, float]:
    """Return residuals for all element-provided linear MPC constraints.

    A constraint is interpreted as:

        u_slave = sum(coeff_i * u_master_i) + value

    The residual returned is lhs - rhs.  A correct reconstructed displacement
    field should give residuals close to zero for every MPC.
    """
    u = np.asarray(displacements, dtype=float).reshape(-1)
    residuals: Dict[str, float] = {}
    counter = 0
    for element in model.mesh.elements.values():
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is None:
            continue
        for constraint in getter(model.mesh) or []:
            slave = int(constraint["slave"])
            masters = constraint.get("masters", {})
            value = float(constraint.get("value", 0.0))
            rhs = value
            for master, coefficient in masters.items():
                rhs += float(coefficient) * float(u[int(master)])
            label = str(constraint.get("label", f"mpc_{counter}"))
            residuals[label] = float(u[slave] - rhs)
            counter += 1
    return residuals


def shell_element_patch_summary(model: "FEModel", element_id: int, element_displacements: np.ndarray) -> ShellPatchSummary:
    """Return compact shell element diagnostics for a supplied element displacement field."""
    element = model.mesh.get_element(element_id)
    if element is None:
        raise ValueError(f"Element {element_id} not found")
    material = model.get_material(element.material_name)
    u = np.asarray(element_displacements, dtype=float).reshape(-1)
    if u.shape != (element.total_dofs,):
        raise ValueError(f"Element displacement shape {u.shape} does not match {(element.total_dofs,)}")

    K = element.compute_stiffness_matrix(model.mesh, material)
    stresses = element.compute_stresses(model.mesh, u, material)
    membrane_arrays = [
        np.asarray(stresses[key], dtype=float)
        for key in ("membrane_xx", "membrane_yy", "membrane_xy")
        if key in stresses
    ]
    bending_arrays = [
        np.asarray(stresses[key], dtype=float)
        for key in ("bending_xx", "bending_yy", "bending_xy")
        if key in stresses
    ]

    max_membrane_spread = max((float(np.max(values) - np.min(values)) for values in membrane_arrays), default=0.0)
    max_bending_spread = max((float(np.max(values) - np.min(values)) for values in bending_arrays), default=0.0)
    return ShellPatchSummary(
        element_id=int(element_id),
        strain_energy=float(u @ K @ u),
        stiffness_symmetry_error=float(np.linalg.norm(K - K.T)),
        max_membrane_spread=max_membrane_spread,
        max_bending_spread=max_bending_spread,
    )


def max_abs(values: Iterable[float]) -> float:
    """Return max absolute value, or 0.0 for an empty iterable."""
    data = [abs(float(value)) for value in values]
    return max(data) if data else 0.0


def nullspace_diagnostics(solver_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract nullspace-related diagnostics in a stable shape for tests."""
    convergence = solver_info.get("convergence_info", {}) or {}
    nullspace = solver_info.get("nullspace_info", {}) or {}
    return {
        "constraint_method": solver_info.get("constraint_method", ""),
        "rank": int(nullspace.get("rank", convergence.get("nullspace_rank", 0)) or 0),
        "relative_rigid_body_load_imbalance": float(convergence.get("relative_rigid_body_load_imbalance", 0.0) or 0.0),
        "augmented_residual_norm": float(convergence.get("augmented_residual_norm", 0.0) or 0.0),
        "gauge_residual_norm": float(convergence.get("gauge_residual_norm", 0.0) or 0.0),
        "status": convergence.get("status", "unknown"),
        "warnings": list(convergence.get("warnings", []) or []),
    }


def _issue(
    code: str,
    severity: str,
    entity_type: str,
    entity_id: Optional[int],
    message: str,
    *,
    measured: Optional[float] = None,
    limit: Optional[float] = None,
    suggestion: str = "",
) -> ProductionValidationIssue:
    return ProductionValidationIssue(
        code=code,
        severity=severity,
        entity_type=entity_type,
        entity_id=None if entity_id is None else int(entity_id),
        message=message,
        measured=None if measured is None else float(measured),
        limit=None if limit is None else float(limit),
        suggestion=suggestion,
    )


def _shell_corner_metrics(coords: np.ndarray) -> Dict[str, float]:
    corner = np.asarray(coords[:4], dtype=float)
    edges = [
        corner[1] - corner[0],
        corner[2] - corner[1],
        corner[3] - corner[2],
        corner[0] - corner[3],
    ]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    min_edge = min(lengths) if lengths else 0.0
    max_edge = max(lengths) if lengths else 0.0
    aspect_ratio = max_edge / max(min_edge, 1.0e-15)
    normal_raw = np.cross(corner[1] - corner[0], corner[2] - corner[0])
    normal_norm = float(np.linalg.norm(normal_raw))
    signed_area = 0.5 * normal_norm + 0.5 * float(np.linalg.norm(np.cross(corner[2] - corner[0], corner[3] - corner[0])))
    warp = 0.0
    if normal_norm > 1.0e-15:
        normal = normal_raw / normal_norm
        warp = abs(float(np.dot(corner[3] - corner[0], normal))) / max(sum(lengths) / 4.0, 1.0e-15)
    return {
        "min_edge": min_edge,
        "max_edge": max_edge,
        "aspect_ratio": aspect_ratio,
        "warp": warp,
        "area": signed_area,
    }


def _q8_midside_deviation(coords: np.ndarray) -> float:
    if np.asarray(coords).shape[0] != 8:
        return 0.0
    pairs = ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7))
    deviations = []
    for i, j, midside in pairs:
        midpoint = 0.5 * (coords[i] + coords[j])
        edge_length = max(float(np.linalg.norm(coords[j] - coords[i])), 1.0e-15)
        deviations.append(float(np.linalg.norm(coords[midside] - midpoint)) / edge_length)
    return max(deviations) if deviations else 0.0


def validate_production_model(
    model: "FEModel",
    load_cases: Optional[Sequence["LoadCase"]] = None,
    *,
    allow_free_mechanisms: bool = False,
    aspect_ratio_limit: float = 8.0,
    warp_limit: float = 0.08,
    midside_deviation_limit: float = 0.20,
) -> ProductionValidationReport:
    """Validate model inputs and mesh quality before production analysis.

    The function is deliberately conservative: it catches invalid or
    unsupported configurations early and returns structured diagnostics with
    entity IDs and corrective hints.  It does not run a full analysis.
    """
    from .assembly import build_constraint_transformation, build_reduced_rigid_body_modes
    from .elements import CoupledBeamShellElement, ShellElement
    from .matrix_assembly import assemble_stiffness_matrix

    issues: List[ProductionValidationIssue] = []
    mesh_quality: Dict[str, Any] = {
        "shell_count": 0,
        "beam_count": 0,
        "max_aspect_ratio": 1.0,
        "max_warp": 0.0,
        "max_q8_midside_deviation": 0.0,
        "min_edge_length": None,
    }

    for name, material in model.materials.items():
        entity_id = None
        if not np.isfinite(material.elastic_modulus) or material.elastic_modulus <= 0.0:
            issues.append(_issue("MAT001", "error", "material", entity_id, f"Material {name!r} has invalid elastic modulus.", measured=material.elastic_modulus, limit=0.0, suggestion="Use a positive SI elastic modulus."))
        if not np.isfinite(material.poisson_ratio) or not (-0.99 < material.poisson_ratio < 0.5):
            issues.append(_issue("MAT002", "error", "material", entity_id, f"Material {name!r} has unsupported Poisson ratio.", measured=material.poisson_ratio, limit=0.5, suggestion="Use -0.99 < nu < 0.5 for isotropic elastic material."))
        if not np.isfinite(material.density) or material.density < 0.0:
            issues.append(_issue("MAT003", "error", "material", entity_id, f"Material {name!r} has invalid density.", measured=material.density, limit=0.0, suggestion="Use zero or positive SI density."))

    slave_dof_owner: Dict[int, int] = {}
    min_edge_values: List[float] = []
    for element_id, element in model.mesh.elements.items():
        material_name = getattr(element, "material_name", None)
        if material_name not in model.materials:
            issues.append(_issue("ELM001", "error", "element", int(element_id), f"Element references missing material {material_name!r}.", suggestion="Add the material before analysis or assign a valid material name."))
        thickness = getattr(element, "thickness", None)
        if thickness is not None and (not np.isfinite(float(thickness)) or float(thickness) <= 0.0):
            issues.append(_issue("SHELL001", "error", "element", int(element_id), "Shell element has non-positive thickness.", measured=float(thickness), limit=0.0, suggestion="Use a positive shell thickness in metres."))

        if isinstance(element, ShellElement):
            mesh_quality["shell_count"] += 1
            try:
                coords = element.get_node_coordinates(model.mesh)
                metrics = _shell_corner_metrics(coords)
                min_edge_values.append(float(metrics["min_edge"]))
                mesh_quality["max_aspect_ratio"] = max(float(mesh_quality["max_aspect_ratio"]), float(metrics["aspect_ratio"]))
                mesh_quality["max_warp"] = max(float(mesh_quality["max_warp"]), float(metrics["warp"]))
                if metrics["area"] <= 1.0e-18 or metrics["min_edge"] <= 1.0e-14:
                    issues.append(_issue("MESH001", "error", "element", int(element_id), "Shell element is degenerate or has near-zero area.", measured=metrics["area"], limit=1.0e-18, suggestion="Repair node ordering/coordinates or remesh this region."))
                if metrics["aspect_ratio"] > aspect_ratio_limit:
                    issues.append(_issue("MESH002", "warning", "element", int(element_id), "Shell element aspect ratio exceeds production guidance.", measured=metrics["aspect_ratio"], limit=aspect_ratio_limit, suggestion="Refine or rebalance the mesh locally."))
                if metrics["warp"] > warp_limit:
                    issues.append(_issue("MESH003", "warning", "element", int(element_id), "Shell element warpage exceeds production guidance.", measured=metrics["warp"], limit=warp_limit, suggestion="Use smaller elements or flatten the panel representation."))
                midside = _q8_midside_deviation(coords)
                mesh_quality["max_q8_midside_deviation"] = max(float(mesh_quality["max_q8_midside_deviation"]), midside)
                if midside > midside_deviation_limit:
                    issues.append(_issue("MESH004", "warning", "element", int(element_id), "Q8/S8 midside node deviates strongly from edge midpoint.", measured=midside, limit=midside_deviation_limit, suggestion="Place midside nodes near geometric edge midpoints or regenerate the mesh."))
            except Exception as exc:
                issues.append(_issue("MESH005", "error", "element", int(element_id), f"Shell mesh-quality evaluation failed: {exc}", suggestion="Check element connectivity and node coordinates."))
        elif isinstance(element, CoupledBeamShellElement):
            pass
        else:
            mesh_quality["beam_count"] += 1

        getter = getattr(element, "get_mpc_constraints", None)
        if getter is not None:
            for constraint in getter(model.mesh) or []:
                slave = int(constraint.get("slave"))
                previous = slave_dof_owner.get(slave)
                if previous is not None and previous != int(element_id):
                    issues.append(_issue("MPC001", "error", "mpc", int(element_id), f"MPC slave DOF {slave} is constrained by multiple MPC elements.", measured=float(slave), suggestion="Ensure each dependent DOF has only one owner."))
                slave_dof_owner[slave] = int(element_id)

    if min_edge_values:
        mesh_quality["min_edge_length"] = float(min(min_edge_values))

    for load_case in load_cases or ():
        if bool(getattr(load_case, "follower_pressure", False)):
            issues.append(_issue("LOAD001", "error", "load_case", None, f"Load case {load_case.name!r} requests follower pressure, which is outside the current verified production scope.", suggestion="Use prescribed fixed-direction pressure or explicitly implement/verify follower pressure first."))
        for element_id in getattr(load_case, "pressure_loads", {}) or {}:
            element = model.mesh.get_element(int(element_id))
            if element is None:
                issues.append(_issue("LOAD002", "error", "load_case", None, f"Load case {load_case.name!r} references missing pressure element {element_id}.", measured=float(element_id), suggestion="Remove stale pressure load entries or regenerate the load case."))

    if not allow_free_mechanisms and not any(issue.severity == "error" for issue in issues):
        try:
            model.apply_boundary_conditions()
            K, _info = assemble_stiffness_matrix(model)
            zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
            _K_red, _F_red, _T, _u0, independent, _constraint_info = build_constraint_transformation(K, zero, model)
            Q, nullspace = build_reduced_rigid_body_modes(model, independent, int(K.shape[0]))
            rank = int(Q.shape[1])
            if rank > 0:
                issues.append(_issue("MECH001", "error", "model", None, "Model has unsupported free rigid-body mechanisms for ordinary production static analysis.", measured=float(rank), limit=0.0, suggestion="Add supports, use a self-equilibrated free-free workflow, or explicitly allow free mechanisms."))
            mesh_quality["rigid_body_nullspace_rank"] = rank
            mesh_quality["connected_components"] = int(nullspace.get("component_count", 0) or 0)
        except Exception as exc:
            issues.append(_issue("MECH002", "error", "model", None, f"Mechanism check failed: {exc}", suggestion="Inspect constraints, MPC topology and element connectivity."))

    status = "invalid" if any(issue.severity == "error" for issue in issues) else ("warning" if issues else "ok")
    revision_signature = getattr(model, "revision_signature", lambda: {})()
    return ProductionValidationReport(status, tuple(issues), mesh_quality, revision_signature)
