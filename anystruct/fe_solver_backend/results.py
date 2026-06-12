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


def create_fe_result(model: 'FEModel', displacements: np.ndarray, 
                     solver_info: Dict[str, Any], 
                     assembly_info: Dict[str, Any] = None) -> FEResult:
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
    from .assembly import extract_node_displacements, compute_reactions, compute_stresses
    
    mesh = model.mesh
    
    # Extract node displacements
    node_displacements = extract_node_displacements(displacements, mesh)
    
    # Compute reactions (need a load case)
    reactions = {}
    if 'load_case' in solver_info:
        load_case = solver_info['load_case']
        reactions = compute_reactions(model, displacements, load_case)
    
    # Compute stresses
    element_stresses = compute_stresses(model, displacements)
    
    return FEResult(
        model_name=model.name,
        displacements=displacements,
        node_displacements=node_displacements,
        element_stresses=element_stresses,
        reactions=reactions,
        solver_info=solver_info,
        assembly_info=assembly_info or {}
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
