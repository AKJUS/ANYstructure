"""
Results Module

This module provides classes for storing and processing FE analysis results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Any
import numpy as np

if TYPE_CHECKING:
    from .fe_core import FEModel, FEMesh
    from .recovery import RecoveryConfig, ResourceConfig


@dataclass
class FEResult:
    """
    Complete FE analysis result.
    
    Stores displacements, stresses, reactions, and solver information.
    """
    model_name: str
    displacements: np.ndarray  # Global displacement vector
    node_displacements: Dict[int, np.ndarray] = field(default_factory=dict)
    element_stresses: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    reactions: Dict[int, np.ndarray] = field(default_factory=dict)
    solver_info: Dict[str, Any] = field(default_factory=dict)
    assembly_info: Dict[str, Any] = field(default_factory=dict)
    result_case: Optional[Dict[str, Any]] = None
    
    # Additional results
    strains: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    forces: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.displacements is not None:
            self.max_displacement = np.max(np.abs(self.displacements))
            self.max_displacement_node = np.argmax(np.abs(self.displacements))
        else:
            self.max_displacement = 0.0
            self.max_displacement_node = -1
    
    def get_node_displacement(self, node_id: int) -> Optional[np.ndarray]:
        """Get displacement for a specific node."""
        return self.node_displacements.get(node_id)
    
    def get_element_stress(self, element_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Get stresses for a specific element."""
        return self.element_stresses.get(element_id)
    
    def get_reaction(self, node_id: int) -> Optional[np.ndarray]:
        """Get reaction forces for a specific node."""
        return self.reactions.get(node_id)
    
    def get_max_displacement(self) -> Tuple[float, int]:
        """Get maximum displacement and its node ID."""
        return self.max_displacement, self.max_displacement_node
    
    def get_displacement_norm(self) -> float:
        """Get the norm of the displacement vector."""
        return np.linalg.norm(self.displacements)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'model_name': self.model_name,
            'max_displacement': self.max_displacement,
            'displacement_norm': self.get_displacement_norm(),
            'num_nodes': len(self.node_displacements),
            'num_elements': len(self.element_stresses),
            'solver_info': self.solver_info,
            'assembly_info': self.assembly_info,
            'result_case': self.result_case,
            'node_displacements': {k: v.tolist() for k, v in self.node_displacements.items()},
            'reactions': {k: v.tolist() for k, v in self.reactions.items()}
        }
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"FE Analysis Result: {self.model_name}",
            f"Max Displacement: {self.max_displacement:.6e} m",
            f"Displacement Norm: {self.get_displacement_norm():.6e} m",
            f"Number of Nodes: {len(self.node_displacements)}",
            f"Number of Elements: {len(self.element_stresses)}",
            f"Solver: {self.solver_info.get('solver_type', 'unknown')}",
            f"Convergence: {self.solver_info.get('convergence_info', {}).get('status', 'unknown')}"
        ]
        return "\n".join(lines)


@dataclass
class StressResult:
    """Stress results for a single element."""
    element_id: int
    element_type: str
    
    # Shell stresses (if applicable)
    membrane_stress_xx: Optional[np.ndarray] = None
    membrane_stress_yy: Optional[np.ndarray] = None
    membrane_stress_xy: Optional[np.ndarray] = None
    bending_stress_xx: Optional[np.ndarray] = None
    bending_stress_yy: Optional[np.ndarray] = None
    bending_stress_xy: Optional[np.ndarray] = None
    shear_stress_xz: Optional[np.ndarray] = None
    shear_stress_yz: Optional[np.ndarray] = None
    von_mises_stress: Optional[np.ndarray] = None
    
    # Beam stresses (if applicable)
    axial_stress: Optional[float] = None
    bending_stress_y: Optional[float] = None
    bending_stress_z: Optional[float] = None
    shear_stress_y: Optional[float] = None
    shear_stress_z: Optional[float] = None
    torsional_stress: Optional[float] = None
    
    def get_max_von_mises(self) -> float:
        """Get maximum von Mises stress."""
        if self.von_mises_stress is not None:
            return np.max(np.abs(self.von_mises_stress))
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'element_id': self.element_id,
            'element_type': self.element_type,
            'max_von_mises': self.get_max_von_mises()
        }
        
        if self.axial_stress is not None:
            result['axial_stress'] = self.axial_stress
        if self.bending_stress_y is not None:
            result['bending_stress_y'] = self.bending_stress_y
        if self.bending_stress_z is not None:
            result['bending_stress_z'] = self.bending_stress_z
        
        return result


@dataclass
class DisplacementResult:
    """Displacement results for a single node."""
    node_id: int
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.ux, self.uy, self.uz, self.rx, self.ry, self.rz])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'ux': self.ux,
            'uy': self.uy,
            'uz': self.uz,
            'rx': self.rx,
            'ry': self.ry,
            'rz': self.rz
        }


def create_fe_result(
    model: 'FEModel',
    displacements: np.ndarray,
    solver_info: Dict[str, Any],
    assembly_info: Dict[str, Any] = None,
    recovery_config: Optional["RecoveryConfig"] = None,
    resource_config: Optional["ResourceConfig"] = None,
) -> FEResult:
    """
    Create a complete FE result from solver output.
    
    Args:
        model: The FE model
        displacements: Global displacement vector
        solver_info: Solver information dictionary
        assembly_info: Assembly information dictionary
    
    Returns:
        FEResult object with all results
    """
    from .assembly import compute_reactions
    from .recovery import (
        default_recovery_config,
        enforce_memory_limit,
        estimate_model_memory,
        filter_reactions,
        recover_element_stresses,
        recover_element_stresses_with_report,
        recovery_metadata,
        select_node_displacements,
    )
    
    policy_requested = recovery_config is not None or resource_config is not None
    recovery = default_recovery_config(recovery_config)
    memory_estimate = estimate_model_memory(model, recovery_config=recovery) if policy_requested else None
    if memory_estimate is not None:
        enforce_memory_limit(memory_estimate, resource_config, context="create_fe_result")
    policy_metadata = recovery_metadata(recovery, resource_config, memory_estimate) if policy_requested else {}
    result_solver_info = dict(solver_info)
    if policy_requested:
        result_solver_info["recovery_policy"] = policy_metadata
    
    # Extract node displacements
    node_displacements = select_node_displacements(model, displacements, recovery)
    
    # Compute reactions (need a load case)
    reactions = {}
    if 'load_case' in solver_info:
        load_case = solver_info['load_case']
        reactions = filter_reactions(compute_reactions(model, displacements, load_case), recovery, model)
    
    # Compute stresses
    if policy_requested:
        element_stresses, stress_report = recover_element_stresses_with_report(
            model,
            displacements,
            recovery,
            resource_config=resource_config,
        )
        result_solver_info["recovery_policy"]["execution"] = {"element_stress_recovery": stress_report.to_dict()}
    else:
        element_stresses = recover_element_stresses(model, displacements, recovery)

    result_case = solver_info.get("result_case")
    if policy_requested and isinstance(result_case, dict):
        result_case = dict(result_case)
        result_case["recovery"] = {**dict(result_case.get("recovery", {})), **recovery.to_dict()}
        metadata = dict(result_case.get("metadata", {}))
        metadata.update({key: value for key, value in policy_metadata.items() if key != "recovery"})
        result_case["metadata"] = metadata
    
    return FEResult(
        model_name=model.name,
        displacements=displacements,
        node_displacements=node_displacements,
        element_stresses=element_stresses,
        reactions=reactions,
        solver_info=result_solver_info,
        assembly_info=assembly_info or {},
        result_case=result_case,
    )


def post_process_results(result: FEResult) -> Dict[str, Any]:
    """
    Perform additional post-processing on FE results.
    
    Args:
        result: The FE result to post-process
    
    Returns:
        Dictionary with post-processed results
    """
    post_processed = {
        'global': {
            'max_displacement': result.max_displacement,
            'displacement_norm': result.get_displacement_norm(),
            'total_strain_energy': 0.0  # Would need stress-strain integration
        },
        'nodes': {},
        'elements': {}
    }
    
    # Process node results
    for node_id, disp in result.node_displacements.items():
        post_processed['nodes'][node_id] = {
            'displacement_magnitude': np.linalg.norm(disp[:3]),
            'rotation_magnitude': np.linalg.norm(disp[3:])
        }
    
    # Process element results
    for elem_id, stresses in result.element_stresses.items():
        elem_result = {'max_stress': 0.0}
        
        # Find maximum stress component
        for stress_name, stress_values in stresses.items():
            if isinstance(stress_values, np.ndarray):
                max_stress = np.max(np.abs(stress_values))
            else:
                max_stress = abs(stress_values)
            
            if max_stress > elem_result['max_stress']:
                elem_result['max_stress'] = max_stress
        
        post_processed['elements'][elem_id] = elem_result
    
    return post_processed


def compare_with_analytical(result: FEResult, analytical_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare FE results with analytical/semi-analytical results.
    
    Args:
        result: FE result
        analytical_result: Analytical result dictionary
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'displacement': {},
        'stress': {},
        'overall': {}
    }
    
    # Compare displacements if available
    if 'max_displacement' in analytical_result:
        fe_max_disp = result.max_displacement
        analytical_max_disp = analytical_result['max_displacement']
        comparison['displacement']['fe'] = fe_max_disp
        comparison['displacement']['analytical'] = analytical_max_disp
        comparison['displacement']['ratio'] = fe_max_disp / analytical_max_disp if analytical_max_disp > 0 else float('inf')
        comparison['displacement']['error_percent'] = abs(fe_max_disp - analytical_max_disp) / analytical_max_disp * 100 if analytical_max_disp > 0 else float('inf')
    
    # Compare stresses if available
    if 'max_stress' in analytical_result:
        fe_max_stress = 0.0
        for elem_stresses in result.element_stresses.values():
            for stresses in elem_stresses.values():
                if isinstance(stresses, np.ndarray):
                    max_s = np.max(np.abs(stresses))
                else:
                    max_s = abs(stresses)
                if max_s > fe_max_stress:
                    fe_max_stress = max_s
        
        analytical_max_stress = analytical_result['max_stress']
        comparison['stress']['fe'] = fe_max_stress
        comparison['stress']['analytical'] = analytical_max_stress
        comparison['stress']['ratio'] = fe_max_stress / analytical_max_stress if analytical_max_stress > 0 else float('inf')
        comparison['stress']['error_percent'] = abs(fe_max_stress - analytical_max_stress) / analytical_max_stress * 100 if analytical_max_stress > 0 else float('inf')
    
    return comparison


_QUAD4_NODE_NATURAL = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
_QUAD8_NODE_NATURAL = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=float,
)

_RECOVERED_STRESS_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")


def _polynomial_basis(points: np.ndarray, num_gauss: int) -> np.ndarray:
    """Polynomial basis matched to the Gauss rule for stress extrapolation."""
    xi = points[:, 0]
    eta = points[:, 1]
    if num_gauss >= 9:
        return np.column_stack(
            [np.ones_like(xi), xi, eta, xi * eta, xi**2, eta**2, xi**2 * eta, xi * eta**2, xi**2 * eta**2]
        )
    if num_gauss >= 4:
        return np.column_stack([np.ones_like(xi), xi, eta, xi * eta])
    return np.ones((points.shape[0], 1), dtype=float)


def _gauss_to_node_extrapolation(element: Any) -> Optional[np.ndarray]:
    """Return the (num_nodes, num_gauss) extrapolation operator for one shell."""
    gauss_points = np.asarray(getattr(element, "gauss_points", ()), dtype=float).reshape(-1, 2)
    num_gauss = gauss_points.shape[0]
    if num_gauss == 0:
        return None
    if getattr(element, "num_nodes", 0) == 4:
        node_natural = _QUAD4_NODE_NATURAL
    elif getattr(element, "num_nodes", 0) == 8:
        node_natural = _QUAD8_NODE_NATURAL
    else:
        return None
    P_gauss = _polynomial_basis(gauss_points, num_gauss)
    P_nodes = _polynomial_basis(node_natural, num_gauss)
    coefficients, *_ = np.linalg.lstsq(P_gauss, np.eye(num_gauss), rcond=None)
    return P_nodes @ coefficients


def _von_mises_surface(components: Dict[str, float], suffix: str) -> float:
    sxx = components[f"global_xx_{suffix}"]
    syy = components[f"global_yy_{suffix}"]
    szz = components[f"global_zz_{suffix}"]
    sxy = components[f"global_xy_{suffix}"]
    syz = components[f"global_yz_{suffix}"]
    sxz = components[f"global_xz_{suffix}"]
    return float(
        np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def recover_nodal_stresses(
    model: "FEModel",
    displacements: np.ndarray,
    element_ids: Optional[Any] = None,
) -> Dict[str, Any]:
    """Gauss-to-node extrapolated, patch-averaged shell surface stresses.

    Shell integration-point stresses are second-order accurate inside the
    element but underestimate surface peaks on coarse meshes, especially for
    the 2x2-point S4.  This recovery evaluates the global-frame top/bottom
    surface stresses at the integration points, extrapolates them to the
    element nodes with the polynomial basis matched to the Gauss rule, and
    averages contributions from all shell elements sharing each node.  Nodal
    von Mises values are computed from the averaged global components, so the
    result is frame-consistent on distorted meshes.

    Supported for 4-node and 8-node quadrilateral shells; other element types
    are skipped.  Returns per-node averaged components/von Mises, the
    per-element nodal (unaveraged) values, and the peak recovered von Mises.
    """
    from .elements import ShellElement

    keys = [f"global_{component}_{surface}" for surface in ("top", "bot") for component in _RECOVERED_STRESS_COMPONENTS]
    selected = None if element_ids is None else {int(element_id) for element_id in element_ids}
    node_sums: Dict[int, Dict[str, float]] = {}
    node_counts: Dict[int, int] = {}
    element_nodal: Dict[int, Dict[str, np.ndarray]] = {}
    skipped: List[int] = []
    for element_id, element in model.mesh.elements.items():
        if selected is not None and int(element_id) not in selected:
            continue
        if not isinstance(element, ShellElement):
            continue
        operator = _gauss_to_node_extrapolation(element)
        if operator is None:
            skipped.append(int(element_id))
            continue
        material = model.get_material(element.material_name)
        stresses = element.compute_stresses(model.mesh, displacements, material, return_global=True)
        if any(key not in stresses for key in keys):
            skipped.append(int(element_id))
            continue
        nodal_values = {key: operator @ np.asarray(stresses[key], dtype=float).reshape(-1) for key in keys}
        element_nodal[int(element_id)] = nodal_values
        for local_index, node_id in enumerate(element.node_ids):
            entry = node_sums.setdefault(int(node_id), {key: 0.0 for key in keys})
            for key in keys:
                entry[key] += float(nodal_values[key][local_index])
            node_counts[int(node_id)] = node_counts.get(int(node_id), 0) + 1

    nodal: Dict[int, Dict[str, float]] = {}
    max_von_mises = 0.0
    max_von_mises_node: Optional[int] = None
    for node_id, sums in node_sums.items():
        count = max(node_counts.get(node_id, 1), 1)
        averaged = {key: value / count for key, value in sums.items()}
        vm_top = _von_mises_surface(averaged, "top")
        vm_bot = _von_mises_surface(averaged, "bot")
        averaged["von_mises_top"] = vm_top
        averaged["von_mises_bot"] = vm_bot
        averaged["von_mises"] = max(vm_top, vm_bot)
        nodal[node_id] = averaged
        if averaged["von_mises"] > max_von_mises:
            max_von_mises = averaged["von_mises"]
            max_von_mises_node = int(node_id)

    return {
        "method": "gauss_extrapolation_nodal_average",
        "stress_frame": "global",
        "nodal": nodal,
        "element_nodal": element_nodal,
        "max_von_mises": float(max_von_mises),
        "max_von_mises_node": max_von_mises_node,
        "skipped_element_ids": sorted(skipped),
    }
