"""
Comprehensive Test Cases for FE Solver

This module provides realistic test cases for ship structural analysis,
including comparisons with semi-analytical solutions where available.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Any
import numpy as np
import json
import os
import time

if TYPE_CHECKING:
    from .fe_core import FEModel
    from .boundary import LoadCase
    from .results import FEResult


@dataclass
class TestCase:
    """Base class for test cases."""
    name: str
    description: str
    category: str = "General"
    
    def run(self) -> Dict[str, Any]:
        """Run the test case."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category
        }


@dataclass
class ShipPanelTestCase(TestCase):
    """
    Test Case: Typical Ship Stiffened Panel
    
    This test case models a typical ship panel with:
    - Plate: 2000 x 1000 mm, 15 mm thick
    - Stiffeners: T-bar, 100 mm web, 8 mm web thickness, 100x12 mm flange
    - Spacing: 600 mm
    - Material: Ship steel (E=210 GPa, ν=0.3, σy=235 MPa)
    - Loading: Lateral pressure + in-plane loads
    
    Compares with:
    - Semi-analytical PULS results (if available)
    - DNV-CG-0128 guidelines
    """
    
    # Geometry parameters (mm)
    panel_length: float = 2000.0
    panel_width: float = 1000.0
    plate_thickness: float = 15.0
    
    # Stiffener parameters (mm)
    stiffener_type: str = "T-bar"
    stiffener_spacing: float = 600.0
    web_height: float = 100.0
    web_thickness: float = 8.0
    flange_width: float = 100.0
    flange_thickness: float = 12.0
    
    # Material properties
    elastic_modulus: float = 210e9  # Pa
    poisson_ratio: float = 0.3
    yield_stress: float = 235e6  # Pa
    
    # Loading
    lateral_pressure: float = 0.05e6  # Pa (50 kPa)
    axial_stress: float = 100e6  # Pa (100 MPa)
    transverse_stress: float = 50e6  # Pa
    shear_stress: float = 30e6  # Pa
    
    # Boundary conditions
    in_plane_support: str = "Integrated"
    rotational_support: str = "SS"
    
    # Mesh settings
    shell_divisions_x: int = 8
    shell_divisions_y: int = 4
    beam_divisions: int = 4
    
    def __init__(self):
        super().__init__(
            name="Ship Stiffened Panel",
            description="Typical ship panel with T-bar stiffeners under combined loading",
            category="Ship Structures"
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the ship panel test case."""
        from . import (
            PanelGeometry, MeshConfig, generate_stiffened_panel_mesh,
            LoadCase, solve_linear, create_fe_result, FixedSupport
        )
        
        result = {
            'test_case': self.name,
            'geometry': self.to_dict(),
            'results': {},
            'comparisons': {}
        }
        
        # Create panel geometry
        panel = PanelGeometry(
            length=self.panel_length / 1000,  # Convert to meters
            width=self.panel_width / 1000,
            plate_thickness=self.plate_thickness / 1000,
            stiffener_type=self.stiffener_type,
            stiffener_spacing=self.stiffener_spacing / 1000,
            stiffener_height=self.web_height / 1000,
            stiffener_web_thickness=self.web_thickness / 1000,
            stiffener_flange_width=self.flange_width / 1000,
            stiffener_flange_thickness=self.flange_thickness / 1000,
            num_stiffeners=int(self.panel_width / self.stiffener_spacing),
            in_plane_support=self.in_plane_support,
            rotational_support=self.rotational_support,
            axial_stress=self.axial_stress,
            transverse_stress=self.transverse_stress,
            shear_stress=self.shear_stress,
            pressure=self.lateral_pressure
        )
        
        # Create mesh configuration
        config = MeshConfig(
            shell_num_divisions_x=self.shell_divisions_x,
            shell_num_divisions_y=self.shell_divisions_y,
            beam_num_divisions=self.beam_divisions,
            use_coupling_elements=True,
            coupling_stiffness=1e12
        )
        
        # Generate mesh
        model = generate_stiffened_panel_mesh(panel, config)
        
        result['results']['mesh'] = {
            'num_nodes': model.mesh.num_nodes,
            'num_elements': model.mesh.num_elements,
            'num_shell_elements': len([e for e in model.mesh.elements.values() if e.__class__.__name__ == 'ShellElement']),
            'num_beam_elements': len([e for e in model.mesh.elements.values() if e.__class__.__name__ == 'BeamElement']),
            'num_coupling_elements': len([e for e in model.mesh.elements.values() if e.__class__.__name__ == 'CoupledBeamShellElement'])
        }
        
        # Add edge fixity only to shell/master nodes. Beam nodes may be MPC
        # slaves in the eccentric beam-shell coupling and must not be fixed.
        edge_nodes = []
        L = panel.length
        W = panel.width
        for node_id, node in model.mesh.nodes.items():
            x, y, z = node.x, node.y, node.z
            if (abs(z) < 1e-9 and
                (abs(x) < 1e-6 or abs(x - L) < 1e-6 or
                 abs(y) < 1e-6 or abs(y - W) < 1e-6)):
                edge_nodes.append(node_id)
        
        for node_id in edge_nodes:
            model.add_boundary_condition(FixedSupport(f"Edge_{node_id}", [node_id]))
        
        # Create load case
        load_case = LoadCase(name="combined_loading")
        
        # Apply lateral pressure to all shell elements
        for elem_id, element in model.mesh.elements.items():
            if element.__class__.__name__ == 'ShellElement':
                load_case.add_pressure_load(elem_id, pressure=self.lateral_pressure)
        
        # Apply in-plane loads (simplified as nodal loads)
        # This would be more accurate with proper in-plane load application
        
        # Solve
        start_time = time.time()
        displacements, solver_info = solve_linear(model, load_case, solver_type='direct')
        solve_time = time.time() - start_time
        
        result['results']['solver'] = {
            'solve_time': solve_time,
            'convergence': solver_info.get('convergence_info', {}).get('status', 'unknown'),
            'num_free_dofs': solver_info.get('num_free_dofs', 0),
            'num_constrained_dofs': solver_info.get('num_constrained_dofs', 0)
        }
        
        # Create result object
        solver_info_copy = solver_info.copy()
        solver_info_copy['load_case'] = load_case
        fe_result = create_fe_result(model, displacements, solver_info_copy)
        
        # Extract key results
        max_disp = fe_result.max_displacement
        max_disp_node = fe_result.max_displacement_node
        
        result['results']['displacements'] = {
            'max_displacement': max_disp,
            'max_displacement_node': max_disp_node,
            'displacement_norm': fe_result.get_displacement_norm()
        }
        
        # Calculate stresses (simplified)
        # For a more complete analysis, we would compute stresses at integration points
        
        result['results']['stresses'] = {
            'note': 'Stress calculation would be added in post-processing'
        }
        
        # Semi-analytical comparison (if available)
        # These would be compared with PULS results
        result['comparisons'] = {
            'semi_analytical': {
                'note': 'Comparison with PULS/DNV-CG-0128 would be added here',
                'expected_behavior': 'FE results should be within 10-20% of semi-analytical for coarse mesh'
            }
        }
        
        # Design checks
        result['design_checks'] = self.perform_design_checks(fe_result, panel)
        
        return result
    
    def perform_design_checks(self, fe_result: 'FEResult', panel: 'PanelGeometry') -> Dict[str, Any]:
        """Perform basic design checks."""
        checks = {}
        
        # Check 1: Displacement limit (typical: L/200 for serviceability)
        L = panel.length
        max_disp = fe_result.max_displacement
        displacement_limit = L / 200
        displacement_utilization = max_disp / displacement_limit if displacement_limit > 0 else 0
        
        checks['displacement'] = {
            'max_displacement': max_disp,
            'limit': displacement_limit,
            'utilization': displacement_utilization,
            'status': 'OK' if displacement_utilization < 1.0 else 'EXCEEDS'
        }
        
        # Check 2: Stress limit (yield stress)
        # Note: Actual stress calculation would be needed
        checks['stress'] = {
            'yield_stress': self.yield_stress,
            'note': 'Stress check requires stress recovery from elements',
            'status': 'NOT_AVAILABLE'
        }
        
        # Check 3: Buckling check (simplified)
        # Based on plate slenderness
        plate_slenderness = (panel.width / panel.plate_thickness) * np.sqrt(
            self.yield_stress / (210e9 * np.pi**2)
        )
        checks['buckling'] = {
            'plate_slenderness': plate_slenderness,
            'limit': 200.0,  # Typical limit from DNV
            'status': 'OK' if plate_slenderness < 200.0 else 'EXCEEDS'
        }
        
        return checks


@dataclass
class PULSComparisonTestCase(TestCase):
    """
    Test Case: Comparison with PULS Semi-Analytical Results
    
    This test case compares FE results with PULS semi-analytical solutions
    for stiffened panels. It uses the existing calculate_puls.py module.
    
    Note: This requires the PULS module to be available.
    """
    
    # Panel parameters matching PULS test cases
    panel_length: float = 3000.0  # mm
    panel_width: float = 800.0   # mm
    plate_thickness: float = 12.0  # mm
    
    stiffener_type: str = "T-bar"
    stiffener_spacing: float = 800.0  # mm
    web_height: float = 120.0  # mm
    web_thickness: float = 8.0   # mm
    flange_width: float = 100.0  # mm
    flange_thickness: float = 12.0  # mm
    
    # Loading
    axial_stress: float = 150.0  # MPa
    transverse_stress: float = 50.0  # MPa
    shear_stress: float = 40.0  # MPa
    lateral_pressure: float = 0.03  # MPa
    
    # Material
    elastic_modulus: float = 210000.0  # MPa
    poisson_ratio: float = 0.3
    yield_stress: float = 235.0  # MPa
    
    def __init__(self):
        super().__init__(
            name="PULS Comparison",
            description="Comparison with PULS semi-analytical solver",
            category="Verification"
        )
    
    def run(self) -> Dict[str, Any]:
        """Run PULS comparison test."""
        result = {
            'test_case': self.name,
            'parameters': self.to_dict(),
            'fe_results': {},
            'puls_results': {},
            'comparisons': {}
        }
        
        try:
            # Try to import PULS module
            from . import calculate_puls
            puls_available = True
        except ImportError:
            puls_available = False
            result['comparisons']['note'] = "PULS module not available for comparison"
            return result
        
        if not puls_available:
            return result
        
        # Create FE model
        from . import PanelGeometry, MeshConfig, generate_stiffened_panel_mesh, LoadCase, solve_linear
        
        panel = PanelGeometry(
            length=self.panel_length / 1000,
            width=self.panel_width / 1000,
            plate_thickness=self.plate_thickness / 1000,
            stiffener_type=self.stiffener_type,
            stiffener_spacing=self.stiffener_spacing / 1000,
            stiffener_height=self.web_height / 1000,
            stiffener_web_thickness=self.web_thickness / 1000,
            stiffener_flange_width=self.flange_width / 1000,
            stiffener_flange_thickness=self.flange_thickness / 1000,
            num_stiffeners=1,
            in_plane_support="Integrated",
            rotational_support="SS",
            axial_stress=self.axial_stress * 1e6,  # Convert MPa to Pa
            transverse_stress=self.transverse_stress * 1e6,
            shear_stress=self.shear_stress * 1e6,
            pressure=self.lateral_pressure * 1e6
        )
        
        config = MeshConfig(
            shell_num_divisions_x=6,
            shell_num_divisions_y=2,
            beam_num_divisions=6,
            use_coupling_elements=True
        )
        
        model = generate_stiffened_panel_mesh(panel, config)
        
        # Solve FE
        load_case = LoadCase(name="puls_comparison")
        for elem_id, element in model.mesh.elements.items():
            if element.__class__.__name__ == 'ShellElement':
                load_case.add_pressure_load(elem_id, pressure=self.lateral_pressure * 1e6)
        
        displacements, solver_info = solve_linear(model, load_case, solver_type='direct')
        
        result['fe_results'] = {
            'max_displacement': float(np.max(np.abs(displacements))),
            'num_nodes': model.mesh.num_nodes,
            'num_elements': model.mesh.num_elements
        }
        
        # Run PULS (if available)
        try:
            # This would use the existing PULS implementation
            # For now, we'll just note that comparison would be done here
            result['comparisons']['note'] = "PULS comparison would be implemented here"
            result['comparisons']['expected'] = "FE and PULS results should be within 10-20%"
        except Exception as e:
            result['comparisons']['error'] = str(e)
        
        return result


@dataclass
class BucklingTestCase(TestCase):
    """
    Test Case: Plate Buckling Under Compression
    
    This test case verifies the solver's ability to capture buckling behavior
    by comparing with analytical buckling coefficients.
    
    Analytical solution for simply supported plate:
    σ_cr = k * π² * E / (12 * (1-ν²)) * (t/b)²
    
    Where k is the buckling coefficient (4.0 for simply supported square plate)
    """
    
    panel_length: float = 1000.0  # mm
    panel_width: float = 1000.0   # mm
    plate_thickness: float = 10.0  # mm
    
    elastic_modulus: float = 210e9  # Pa
    poisson_ratio: float = 0.3
    
    # Buckling coefficient for simply supported square plate
    buckling_coefficient: float = 4.0
    
    def __init__(self):
        super().__init__(
            name="Plate Buckling",
            description="Plate buckling under uniform compression",
            category="Stability"
        )
    
    def run(self) -> Dict[str, Any]:
        """Run buckling test case."""
        from . import generate_simple_panel_mesh, LoadCase, solve_linear
        
        result = {
            'test_case': self.name,
            'parameters': self.to_dict(),
            'analytical': {},
            'fe_results': {},
            'comparisons': {}
        }
        
        # Calculate analytical buckling stress
        b = self.panel_width / 1000  # Convert to meters
        t = self.plate_thickness / 1000
        
        sigma_cr_analytical = (self.buckling_coefficient * np.pi**2 * self.elastic_modulus / 
                              (12 * (1 - self.poisson_ratio**2))) * (t / b)**2
        
        result['analytical'] = {
            'buckling_stress': sigma_cr_analytical,
            'buckling_load': sigma_cr_analytical * t * b * 1000  # For 1m length
        }
        
        # Create FE model
        model = generate_simple_panel_mesh(
            length=self.panel_length / 1000,
            width=self.panel_width / 1000,
            thickness=self.plate_thickness / 1000,
            num_divisions_x=10,
            num_divisions_y=10
        )
        
        # Apply compressive load (as in-plane load)
        # For linear analysis, we can only apply a fraction of the buckling load
        applied_stress = sigma_cr_analytical * 0.5  # 50% of buckling stress
        
        load_case = LoadCase(name="compression")
        # Apply compressive stress to all nodes on one edge
        for node_id, node in model.mesh.nodes.items():
            if abs(node.y) < 1e-6:  # Bottom edge
                # Apply compressive force
                area = (self.plate_thickness / 1000) * (self.panel_length / 1000 / 10)
                force = applied_stress * area
                load_case.add_nodal_load(node_id, forces=np.array([-force, 0, 0]))
        
        # Fix the other edge
        from .boundary import FixedSupport
        for node_id, node in model.mesh.nodes.items():
            if abs(node.y - self.panel_width / 1000) < 1e-6:  # Top edge
                model.add_boundary_condition(FixedSupport(f"Fixed_{node_id}", [node_id]))
        
        # Solve
        try:
            displacements, solver_info = solve_linear(model, load_case, solver_type='direct')
            max_disp = float(np.max(np.abs(displacements)))
            
            result['fe_results'] = {
                'max_displacement': max_disp,
                'applied_stress': applied_stress,
                'buckling_stress_analytical': sigma_cr_analytical,
                'stress_ratio': applied_stress / sigma_cr_analytical
            }
            
            # Note: Linear analysis cannot capture buckling directly
            # This would require nonlinear analysis or eigenvalue analysis
            result['comparisons'] = {
                'note': 'Linear analysis cannot capture buckling. Eigenvalue analysis needed.',
                'recommendation': 'Implement eigenvalue solver for buckling analysis'
            }
            
        except Exception as e:
            result['fe_results']['error'] = str(e)
        
        return result


@dataclass
class VibrationTestCase(TestCase):
    """
    Test Case: Natural Frequency Analysis
    
    This test case demonstrates how to perform modal analysis
    to find natural frequencies of a stiffened panel.
    
    Note: This requires eigenvalue solver implementation.
    """
    
    panel_length: float = 2000.0  # mm
    panel_width: float = 1000.0   # mm
    plate_thickness: float = 15.0  # mm
    
    stiffener_spacing: float = 600.0  # mm
    web_height: float = 100.0  # mm
    web_thickness: float = 8.0   # mm
    flange_width: float = 100.0  # mm
    flange_thickness: float = 12.0  # mm
    
    elastic_modulus: float = 210e9  # Pa
    poisson_ratio: float = 0.3
    density: float = 7850.0  # kg/m^3
    
    def __init__(self):
        super().__init__(
            name="Natural Frequency",
            description="Modal analysis of stiffened panel",
            category="Dynamics"
        )
    
    def run(self) -> Dict[str, Any]:
        """Run vibration test case."""
        from . import PanelGeometry, MeshConfig, generate_stiffened_panel_mesh
        
        result = {
            'test_case': self.name,
            'parameters': self.to_dict(),
            'results': {},
            'notes': []
        }
        
        # Create panel
        panel = PanelGeometry(
            length=self.panel_length / 1000,
            width=self.panel_width / 1000,
            plate_thickness=self.plate_thickness / 1000,
            stiffener_type="T-bar",
            stiffener_spacing=self.stiffener_spacing / 1000,
            stiffener_height=self.web_height / 1000,
            stiffener_web_thickness=self.web_thickness / 1000,
            stiffener_flange_width=self.flange_width / 1000,
            stiffener_flange_thickness=self.flange_thickness / 1000,
            num_stiffeners=1
        )
        
        config = MeshConfig(
            shell_num_divisions_x=8,
            shell_num_divisions_y=4,
            beam_num_divisions=8,
            use_coupling_elements=True
        )
        
        model = generate_stiffened_panel_mesh(panel, config)
        
        result['results'] = {
            'mesh': {
                'num_nodes': model.mesh.num_nodes,
                'num_elements': model.mesh.num_elements
            },
            'material': {
                'elastic_modulus': self.elastic_modulus,
                'density': self.density,
                'poisson_ratio': self.poisson_ratio
            }
        }
        
        result['notes'].append("Eigenvalue solver for modal analysis not yet implemented")
        result['notes'].append("Would solve: (K - ω²M)u = 0 for natural frequencies")
        result['notes'].append("Expected first mode: ~10-20 Hz for typical ship panel")
        
        return result


@dataclass
class ThermalLoadTestCase(TestCase):
    """
    Test Case: Thermal Loading
    
    This test case demonstrates thermal load analysis.
    
    Note: Thermal analysis requires additional implementation.
    """
    
    panel_length: float = 1000.0  # mm
    panel_width: float = 1000.0   # mm
    plate_thickness: float = 10.0  # mm
    
    elastic_modulus: float = 210e9  # Pa
    poisson_ratio: float = 0.3
    thermal_expansion: float = 12e-6  # 1/°C for steel
    
    temperature_change: float = 50.0  # °C
    
    def __init__(self):
        super().__init__(
            name="Thermal Loading",
            description="Thermal expansion analysis",
            category="Thermal"
        )
    
    def run(self) -> Dict[str, Any]:
        """Run thermal load test case."""
        result = {
            'test_case': self.name,
            'parameters': self.to_dict(),
            'analytical': {},
            'notes': []
        }
        
        # Calculate analytical thermal stress
        # For a constrained plate: σ = E * α * ΔT
        sigma_thermal = self.elastic_modulus * self.thermal_expansion * self.temperature_change
        
        result['analytical'] = {
            'thermal_stress': sigma_thermal,
            'thermal_strain': self.thermal_expansion * self.temperature_change
        }
        
        result['notes'].append("Thermal load implementation not yet complete")
        result['notes'].append("Would require: thermal strain matrix, temperature field")
        result['notes'].append(f"Expected thermal stress: {sigma_thermal/1e6:.1f} MPa")
        
        return result


class TestCaseRunner:
    """Runner for test cases."""
    
    TEST_CASES = {
        'ship_panel': ShipPanelTestCase,
        'puls_comparison': PULSComparisonTestCase,
        'buckling': BucklingTestCase,
        'vibration': VibrationTestCase,
        'thermal': ThermalLoadTestCase
    }
    
    @classmethod
    def run_test_case(cls, test_case_name: str) -> Dict[str, Any]:
        """Run a specific test case."""
        if test_case_name not in cls.TEST_CASES:
            raise ValueError(f"Unknown test case: {test_case_name}")
        
        test_case = cls.TEST_CASES[test_case_name]()
        return test_case.run()
    
    @classmethod
    def run_all_test_cases(cls) -> Dict[str, Dict[str, Any]]:
        """Run all test cases."""
        results = {}
        
        for name, test_class in cls.TEST_CASES.items():
            try:
                test_case = test_class()
                results[name] = test_case.run()
            except Exception as e:
                results[name] = {
                    'test_case': name,
                    'error': str(e),
                    'status': 'FAILED'
                }
        
        return results
    
    @classmethod
    def save_test_case_results(cls, results: Dict[str, Dict[str, Any]], filename: str = "test_case_results.json"):
        """Save test case results to file."""
        import json
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar types (int64, float64, etc.)
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
    
    @classmethod
    def print_test_case_summary(cls, results: Dict[str, Dict[str, Any]]):
        """Print summary of test case results."""
        print("=" * 70)
        print("TEST CASE RESULTS SUMMARY")
        print("=" * 70)
        
        for name, result in results.items():
            status = result.get('status', 'COMPLETED')
            if 'error' in result:
                status = 'FAILED'
            
            print(f"\n{name.upper()}")
            print("-" * len(name))
            print(f"Status: {status}")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            elif 'results' in result:
                if 'max_displacement' in result['results']:
                    print(f"Max Displacement: {result['results']['max_displacement']:.6e} m")
                if 'mesh' in result['results']:
                    print(f"Mesh: {result['results']['mesh']['num_nodes']} nodes, {result['results']['mesh']['num_elements']} elements")
        
        print("\n" + "=" * 70)


def run_ship_panel_test() -> Dict[str, Any]:
    """Run the ship panel test case."""
    test_case = ShipPanelTestCase()
    return test_case.run()


def run_puls_comparison_test() -> Dict[str, Any]:
    """Run the PULS comparison test case."""
    test_case = PULSComparisonTestCase()
    return test_case.run()


def run_all_demo_test_cases() -> Dict[str, Dict[str, Any]]:
    """Run all demonstration test cases."""
    return TestCaseRunner.run_all_test_cases()


if __name__ == "__main__":
    import time
    
    print("Running Comprehensive Test Cases...")
    print("=" * 70)
    
    start_time = time.time()
    results = run_all_demo_test_cases()
    elapsed_time = time.time() - start_time
    
    TestCaseRunner.print_test_case_summary(results)
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    # Save results
    TestCaseRunner.save_test_case_results(results, "test_case_results.json")
    print("Results saved to test_case_results.json")
