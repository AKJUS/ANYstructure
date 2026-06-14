"""Cylinder shell benchmark helpers.

The benchmark is intentionally lightweight and informational.  It builds a
closed cylindrical shell mesh, applies self-equilibrated external pressure, and
reports nominal thin-cylinder membrane stresses alongside FE von Mises
percentiles.  It is a staging point for the later geometric-stiffness and
buckling steps, not a validated pressure-vessel solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .assembly import compute_stresses, solve_linear
from .boundary import LoadCase
from .elements import ShellElement
from .fe_core import FEModel


@dataclass(frozen=True)
class CylinderBenchmarkConfig:
    """Configuration for the internal cylindrical shell pressure benchmark."""

    radius: float = 3.0
    height: float = 5.0
    thickness: float = 0.01
    pressure: float = 100_000.0
    num_circumferential: int = 16
    num_height: int = 8
    use_8node_elements: bool = False
    elastic_modulus: float = 210.0e9
    poisson_ratio: float = 0.3
    density: float = 7850.0
    closed_end_axial_load: bool = True
    mid_height_band_fraction: float = 1.0

    def validate(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.height <= 0.0:
            raise ValueError("height must be positive")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")
        if self.pressure < 0.0:
            raise ValueError("pressure must be non-negative; external pressure direction is handled by the benchmark")
        if self.num_circumferential < 3:
            raise ValueError("num_circumferential must be at least 3")
        if self.num_height < 1:
            raise ValueError("num_height must be at least 1")
        if self.mid_height_band_fraction <= 0.0:
            raise ValueError("mid_height_band_fraction must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "radius": self.radius,
            "height": self.height,
            "thickness": self.thickness,
            "pressure": self.pressure,
            "num_circumferential": self.num_circumferential,
            "num_height": self.num_height,
            "use_8node_elements": self.use_8node_elements,
            "elastic_modulus": self.elastic_modulus,
            "poisson_ratio": self.poisson_ratio,
            "density": self.density,
            "closed_end_axial_load": self.closed_end_axial_load,
            "mid_height_band_fraction": self.mid_height_band_fraction,
        }


@dataclass(frozen=True)
class CylinderNominalStress:
    """Nominal thin-cylinder membrane stresses for the benchmark load."""

    hoop_stress: float
    axial_stress: float
    von_mises_stress: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "hoop_stress": self.hoop_stress,
            "axial_stress": self.axial_stress,
            "von_mises_stress": self.von_mises_stress,
        }


@dataclass(frozen=True)
class CylinderStressStatistics:
    """Von Mises stress percentile summary."""

    count: int
    minimum: float
    maximum: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass(frozen=True)
class CylinderBenchmarkResult:
    """Complete internal cylinder benchmark result."""

    config: CylinderBenchmarkConfig
    nominal: CylinderNominalStress
    all_von_mises: CylinderStressStatistics
    mid_height_von_mises: CylinderStressStatistics
    max_displacement_norm: float
    max_radial_displacement: float
    node_count: int
    element_count: int
    shell_element_count: int
    solver_status: str
    relative_rigid_body_load_imbalance: float

    @property
    def fe_max_von_mises(self) -> float:
        return self.all_von_mises.maximum

    @property
    def fe_p95_von_mises(self) -> float:
        return self.all_von_mises.p95

    @property
    def fe_mid_height_p95_von_mises(self) -> float:
        return self.mid_height_von_mises.p95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "nominal": self.nominal.to_dict(),
            "all_von_mises": self.all_von_mises.to_dict(),
            "mid_height_von_mises": self.mid_height_von_mises.to_dict(),
            "fe_max_von_mises": self.fe_max_von_mises,
            "fe_p95_von_mises": self.fe_p95_von_mises,
            "fe_mid_height_p95_von_mises": self.fe_mid_height_p95_von_mises,
            "max_displacement_norm": self.max_displacement_norm,
            "max_radial_displacement": self.max_radial_displacement,
            "node_count": self.node_count,
            "element_count": self.element_count,
            "shell_element_count": self.shell_element_count,
            "solver_status": self.solver_status,
            "relative_rigid_body_load_imbalance": self.relative_rigid_body_load_imbalance,
        }


def nominal_cylinder_membrane_stress(
    radius: float,
    thickness: float,
    pressure: float,
    closed_ends: bool = True,
) -> CylinderNominalStress:
    """Return signed nominal membrane stresses for external pressure."""
    if radius <= 0.0 or thickness <= 0.0:
        raise ValueError("radius and thickness must be positive")
    hoop = -float(pressure) * float(radius) / float(thickness)
    axial = 0.5 * hoop if closed_ends else 0.0
    von_mises = float(np.sqrt(hoop**2 + axial**2 - hoop * axial))
    return CylinderNominalStress(hoop_stress=hoop, axial_stress=axial, von_mises_stress=von_mises)


def _stats(values: Iterable[float]) -> CylinderStressStatistics:
    arr = np.asarray([float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return CylinderStressStatistics(
            count=0,
            minimum=0.0,
            maximum=0.0,
            mean=0.0,
            p50=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
        )
    return CylinderStressStatistics(
        count=int(arr.size),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50.0)),
        p90=float(np.percentile(arr, 90.0)),
        p95=float(np.percentile(arr, 95.0)),
        p99=float(np.percentile(arr, 99.0)),
    )


def _cylinder_point(radius: float, theta: float, z: float) -> Tuple[float, float, float]:
    return radius * float(np.cos(theta)), radius * float(np.sin(theta)), float(z)


def _add_closed_end_axial_loads(model: FEModel, load_case: LoadCase, config: CylinderBenchmarkConfig) -> None:
    total_cap_force = float(config.pressure) * np.pi * float(config.radius) ** 2
    nodal_force = total_cap_force / float(config.num_circumferential)
    bottom_z = 0.0
    top_z = float(config.height)
    for node_id, node in model.mesh.nodes.items():
        if abs(node.z - bottom_z) < 1.0e-12:
            load_case.add_nodal_load(node_id, forces=np.array([0.0, 0.0, nodal_force]))
        elif abs(node.z - top_z) < 1.0e-12:
            load_case.add_nodal_load(node_id, forces=np.array([0.0, 0.0, -nodal_force]))


def build_cylindrical_shell_benchmark_model(config: Optional[CylinderBenchmarkConfig] = None) -> Tuple[FEModel, LoadCase]:
    """Build a closed cylindrical shell model and self-equilibrated pressure load."""
    config = config or CylinderBenchmarkConfig()
    config.validate()

    model = FEModel(name=f"Cylinder_R{config.radius}_H{config.height}_T{config.thickness}")
    model.add_material(
        "steel",
        elastic_modulus=config.elastic_modulus,
        poisson_ratio=config.poisson_ratio,
        density=config.density,
    )
    model.current_material = "steel"

    node_id = 1
    corner_nodes: Dict[Tuple[int, int], int] = {}
    for iz in range(config.num_height + 1):
        z = config.height * iz / config.num_height
        for itheta in range(config.num_circumferential):
            theta = 2.0 * np.pi * itheta / config.num_circumferential
            corner_nodes[(iz, itheta)] = node_id
            model.add_node(node_id, *_cylinder_point(config.radius, theta, z))
            node_id += 1

    circumferential_mid_nodes: Dict[Tuple[int, int], int] = {}
    vertical_mid_nodes: Dict[Tuple[int, int], int] = {}
    if config.use_8node_elements:
        for iz in range(config.num_height + 1):
            z = config.height * iz / config.num_height
            for itheta in range(config.num_circumferential):
                theta = 2.0 * np.pi * (itheta + 0.5) / config.num_circumferential
                circumferential_mid_nodes[(iz, itheta)] = node_id
                model.add_node(node_id, *_cylinder_point(config.radius, theta, z))
                node_id += 1
        for iz in range(config.num_height):
            z = config.height * (iz + 0.5) / config.num_height
            for itheta in range(config.num_circumferential):
                theta = 2.0 * np.pi * itheta / config.num_circumferential
                vertical_mid_nodes[(iz, itheta)] = node_id
                model.add_node(node_id, *_cylinder_point(config.radius, theta, z))
                node_id += 1

    element_id = 1
    for iz in range(config.num_height):
        for itheta in range(config.num_circumferential):
            next_theta = (itheta + 1) % config.num_circumferential
            n0 = corner_nodes[(iz, itheta)]
            n1 = corner_nodes[(iz, next_theta)]
            n2 = corner_nodes[(iz + 1, next_theta)]
            n3 = corner_nodes[(iz + 1, itheta)]
            if config.use_8node_elements:
                node_ids = [
                    n0,
                    n1,
                    n2,
                    n3,
                    circumferential_mid_nodes[(iz, itheta)],
                    vertical_mid_nodes[(iz, next_theta)],
                    circumferential_mid_nodes[(iz + 1, itheta)],
                    vertical_mid_nodes[(iz, itheta)],
                ]
            else:
                node_ids = [n0, n1, n2, n3]
            model.add_element(
                element_id,
                ShellElement(element_id, node_ids, material_name="steel", thickness=config.thickness),
            )
            element_id += 1

    load_case = LoadCase("external_pressure")
    for elem_id in model.mesh.elements:
        load_case.add_pressure_load(int(elem_id), -float(config.pressure))
    if config.closed_end_axial_load:
        _add_closed_end_axial_loads(model, load_case, config)
    model.add_load_case(load_case)
    return model, load_case


def _sample_shell_von_mises(model: FEModel, displacements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stresses = compute_stresses(model, displacements)
    z_values: List[float] = []
    von_mises_values: List[float] = []
    for element_id, stress in stresses.items():
        element = model.mesh.get_element(element_id)
        if not isinstance(element, ShellElement):
            continue
        coords = element.get_node_coordinates(model.mesh)
        for idx, (xi, eta) in enumerate(element.gauss_points):
            N, _dN_dxi, _dN_deta = element.compute_shape_functions(float(xi), float(eta))
            sample_point = N @ coords
            z_values.append(float(sample_point[2]))
            von_mises_values.append(float(stress["von_mises"][idx]))
    return np.asarray(z_values, dtype=float), np.asarray(von_mises_values, dtype=float)


def _max_displacements(model: FEModel, displacements: np.ndarray) -> Tuple[float, float]:
    max_norm = 0.0
    max_radial = 0.0
    u = np.asarray(displacements, dtype=float)
    for node in model.mesh.nodes.values():
        translation = u[node.dofs[:3]]
        radial = node.coords()
        radial[2] = 0.0
        radial_norm = float(np.linalg.norm(radial))
        radial_unit = radial / radial_norm if radial_norm > 0.0 else np.zeros(3)
        max_norm = max(max_norm, float(np.linalg.norm(translation)))
        max_radial = max(max_radial, abs(float(np.dot(translation, radial_unit))))
    return max_norm, max_radial


def run_cylindrical_shell_benchmark(config: Optional[CylinderBenchmarkConfig] = None) -> CylinderBenchmarkResult:
    """Run the internal cylinder benchmark and report nominal/FE stress metrics."""
    config = config or CylinderBenchmarkConfig()
    config.validate()
    model, load_case = build_cylindrical_shell_benchmark_model(config)
    displacements, solver_info = solve_linear(model, load_case, constraint_mode="auto")

    z_values, von_mises = _sample_shell_von_mises(model, displacements)
    mid_height = 0.5 * config.height
    half_band = 0.5 * config.height / config.num_height * config.mid_height_band_fraction
    mid_mask = np.abs(z_values - mid_height) <= max(half_band, 1.0e-12)
    if not np.any(mid_mask) and z_values.size:
        nearest = int(np.argmin(np.abs(z_values - mid_height)))
        mid_mask = np.zeros_like(z_values, dtype=bool)
        mid_mask[nearest] = True

    max_norm, max_radial = _max_displacements(model, displacements)
    convergence = solver_info.get("convergence_info") or {}
    return CylinderBenchmarkResult(
        config=config,
        nominal=nominal_cylinder_membrane_stress(
            config.radius,
            config.thickness,
            config.pressure,
            closed_ends=config.closed_end_axial_load,
        ),
        all_von_mises=_stats(von_mises),
        mid_height_von_mises=_stats(von_mises[mid_mask]),
        max_displacement_norm=max_norm,
        max_radial_displacement=max_radial,
        node_count=int(model.mesh.num_nodes),
        element_count=int(model.mesh.num_elements),
        shell_element_count=sum(1 for element in model.mesh.elements.values() if isinstance(element, ShellElement)),
        solver_status=str(convergence.get("status", "unknown")),
        relative_rigid_body_load_imbalance=float(convergence.get("relative_rigid_body_load_imbalance", 0.0)),
    )
