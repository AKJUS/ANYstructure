"""Internal shell benchmark runners.

These helpers produce FE-solver-side values in the same shape as the upstream
CalculiX shell convergence tables: mesh size, node count, maximum stress,
maximum displacement and normalized values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .assembly import solve_linear
from .boundary import LoadCase
from .mesh_gen import generate_simple_panel_mesh
from .reference_cases import ShellConvergencePoint, ShellConvergenceTable, upstream_calculix_shell_reference_values


@dataclass(frozen=True)
class ShellBenchmarkResult:
    """One internal shell benchmark result point."""

    element_type: str
    mesh_size: float
    divisions_x: int
    divisions_y: int
    node_count: int
    element_count: int
    pressure: float
    max_out_of_plane_displacement: float
    max_displacement_norm: float
    max_von_mises_stress: float
    displacement_reference: float
    stress_reference: float
    displacement_normalized: float
    stress_normalized: float
    solver_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "mesh_size": self.mesh_size,
            "divisions_x": self.divisions_x,
            "divisions_y": self.divisions_y,
            "node_count": self.node_count,
            "element_count": self.element_count,
            "pressure": self.pressure,
            "max_out_of_plane_displacement": self.max_out_of_plane_displacement,
            "max_displacement_norm": self.max_displacement_norm,
            "max_von_mises_stress": self.max_von_mises_stress,
            "displacement_reference": self.displacement_reference,
            "stress_reference": self.stress_reference,
            "displacement_normalized": self.displacement_normalized,
            "stress_normalized": self.stress_normalized,
            "solver_status": self.solver_status,
        }

@dataclass(frozen=True)
class ShellBenchmarkComparisonPoint:
    """Loose/informational comparison between one external and one internal row."""

    external_size: float
    internal_size: float
    external_node_count: int
    internal_node_count: int
    external_stress_normalized: float
    internal_stress_normalized: float
    external_displacement_normalized: float
    internal_displacement_normalized: float
    stress_normalized_delta: float
    displacement_normalized_delta: float
    stress_ratio_internal_to_external: float
    displacement_ratio_internal_to_external: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "external_size": self.external_size,
            "internal_size": self.internal_size,
            "external_node_count": self.external_node_count,
            "internal_node_count": self.internal_node_count,
            "external_stress_normalized": self.external_stress_normalized,
            "internal_stress_normalized": self.internal_stress_normalized,
            "external_displacement_normalized": self.external_displacement_normalized,
            "internal_displacement_normalized": self.internal_displacement_normalized,
            "stress_normalized_delta": self.stress_normalized_delta,
            "displacement_normalized_delta": self.displacement_normalized_delta,
            "stress_ratio_internal_to_external": self.stress_ratio_internal_to_external,
            "displacement_ratio_internal_to_external": self.displacement_ratio_internal_to_external,
        }


@dataclass(frozen=True)
class ShellBenchmarkComparison:
    """Summary of a loose external-vs-internal shell benchmark comparison."""

    external_element_type: str
    internal_element_type: str
    points: Tuple[ShellBenchmarkComparisonPoint, ...]
    notes: Tuple[str, ...]

    @property
    def max_abs_stress_normalized_delta(self) -> float:
        return max((abs(point.stress_normalized_delta) for point in self.points), default=0.0)

    @property
    def max_abs_displacement_normalized_delta(self) -> float:
        return max((abs(point.displacement_normalized_delta) for point in self.points), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "external_element_type": self.external_element_type,
            "internal_element_type": self.internal_element_type,
            "max_abs_stress_normalized_delta": self.max_abs_stress_normalized_delta,
            "max_abs_displacement_normalized_delta": self.max_abs_displacement_normalized_delta,
            "points": [point.to_dict() for point in self.points],
            "notes": list(self.notes),
        }


def _reference(value: Optional[float], default: float, name: str) -> float:
    ref = float(default if value is None else value)
    if ref == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return ref


def _add_pressure_loads(load_case: LoadCase, model, pressure: float) -> None:
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(int(element_id), float(pressure))


def _max_displacements(model, displacements: np.ndarray) -> Tuple[float, float]:
    u = np.asarray(displacements, dtype=float).reshape(-1)
    max_w = 0.0
    max_norm = 0.0
    for node in model.mesh.nodes.values():
        translation = u[node.dofs[:3]]
        max_w = max(max_w, abs(float(translation[2])))
        max_norm = max(max_norm, float(np.linalg.norm(translation)))
    return max_w, max_norm


def _max_von_mises(model, displacements: np.ndarray) -> float:
    values: List[float] = []
    for element in model.mesh.elements.values():
        if not hasattr(element, "compute_stresses"):
            continue
        material = model.get_material(element.material_name)
        stresses = element.compute_stresses(model.mesh, displacements, material)
        if "von_mises" in stresses:
            values.extend(float(value) for value in np.asarray(stresses["von_mises"], dtype=float).reshape(-1))
    return max((abs(value) for value in values), default=0.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def run_simple_supported_shell_benchmark(
    length: float = 1.0,
    width: float = 1.0,
    thickness: float = 0.01,
    divisions_x: int = 4,
    divisions_y: int = 4,
    pressure: float = 1000.0,
    use_8node_elements: bool = False,
    stress_reference: Optional[float] = None,
    displacement_reference: Optional[float] = None,
) -> ShellBenchmarkResult:
    """Run a rectangular shell panel under a constant surface load."""
    references = upstream_calculix_shell_reference_values()
    sref = _reference(stress_reference, references.get("sref", 1.0), "stress_reference")
    wref = _reference(displacement_reference, references.get("wref", 1.0), "displacement_reference")

    model = generate_simple_panel_mesh(
        length,
        width,
        thickness,
        num_divisions_x=divisions_x,
        num_divisions_y=divisions_y,
        use_8node_elements=use_8node_elements,
    )
    load_case = LoadCase("constant_surface_load")
    _add_pressure_loads(load_case, model, pressure)
    displacements, solver_info = solve_linear(model, load_case)
    solver_status = str((solver_info.get("convergence_info") or {}).get("status", "unknown"))

    max_w, max_norm = _max_displacements(model, displacements)
    max_vm = _max_von_mises(model, displacements)
    return ShellBenchmarkResult(
        element_type="S8" if use_8node_elements else "S4",
        mesh_size=max(float(length) / float(divisions_x), float(width) / float(divisions_y)),
        divisions_x=int(divisions_x),
        divisions_y=int(divisions_y),
        node_count=int(model.mesh.num_nodes),
        element_count=int(len(model.mesh.elements)),
        pressure=float(pressure),
        max_out_of_plane_displacement=max_w,
        max_displacement_norm=max_norm,
        max_von_mises_stress=max_vm,
        displacement_reference=wref,
        stress_reference=sref,
        displacement_normalized=max_w / wref,
        stress_normalized=max_vm / sref,
        solver_status=solver_status,
    )


def run_simple_supported_shell_convergence(
    divisions: Sequence[int] = (2, 4, 8),
    length: float = 1.0,
    width: float = 1.0,
    thickness: float = 0.01,
    pressure: float = 1000.0,
    use_8node_elements: bool = False,
    stress_reference: Optional[float] = None,
    displacement_reference: Optional[float] = None,
) -> Tuple[ShellBenchmarkResult, ...]:
    """Run a small internal shell mesh-convergence sweep."""
    results: List[ShellBenchmarkResult] = []
    for division in divisions:
        div = int(division)
        if div <= 0:
            raise ValueError("divisions must contain positive integers")
        results.append(
            run_simple_supported_shell_benchmark(
                length=length,
                width=width,
                thickness=thickness,
                divisions_x=div,
                divisions_y=div,
                pressure=pressure,
                use_8node_elements=use_8node_elements,
                stress_reference=stress_reference,
                displacement_reference=displacement_reference,
            )
        )
    return tuple(results)


def shell_benchmark_results_to_convergence_table(
    results: Sequence[ShellBenchmarkResult],
    path: Path | str = Path("<internal>"),
) -> ShellConvergenceTable:
    """Convert internal benchmark results to the same table shape as parsed CalculiX files."""
    if not results:
        raise ValueError("results must contain at least one ShellBenchmarkResult")

    element_type = results[0].element_type
    stress_reference = results[0].stress_reference
    displacement_reference = results[0].displacement_reference
    points = []
    for result in results:
        if result.element_type != element_type:
            raise ValueError("All internal results must use the same element type")
        points.append(
            ShellConvergencePoint(
                element_type=result.element_type,
                size=result.mesh_size,
                node_count=result.node_count,
                stress_max=result.max_von_mises_stress,
                displacement_max=result.max_out_of_plane_displacement,
                stress_normalized=result.stress_normalized,
                displacement_normalized=result.displacement_normalized,
            )
        )

    return ShellConvergenceTable(
        element_type=element_type,
        path=Path(path),
        stress_reference=float(stress_reference),
        displacement_reference=float(displacement_reference),
        points=tuple(points),
    )


def write_internal_shell_convergence_table(
    results: Sequence[ShellBenchmarkResult],
    path: Path | str,
) -> Path:
    """Write an internal shell convergence table.

    The output format intentionally mirrors the upstream CalculiX text tables
    with two added normalized columns:

        # size NoN smax umax s_norm u_norm
    """
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# size NoN smax umax s_norm u_norm"]
    for result in results:
        lines.append(
            " ".join(
                [
                    f"{result.mesh_size:.12g}",
                    str(int(result.node_count)),
                    f"{result.max_von_mises_stress:.12g}",
                    f"{result.max_out_of_plane_displacement:.12g}",
                    f"{result.stress_normalized:.12g}",
                    f"{result.displacement_normalized:.12g}",
                ]
            )
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table_path


def _as_internal_table(
    internal: Union[ShellConvergenceTable, Sequence[ShellBenchmarkResult]],
) -> ShellConvergenceTable:
    if isinstance(internal, ShellConvergenceTable):
        return internal
    return shell_benchmark_results_to_convergence_table(internal)


def compare_shell_benchmark_to_reference(
    external_table: ShellConvergenceTable,
    internal: Union[ShellConvergenceTable, Sequence[ShellBenchmarkResult]],
) -> ShellBenchmarkComparison:
    """Compare external CalculiX and internal shell convergence data loosely.

    Rows are paired by order after sorting both tables from coarse to fine
    ``size``.  This is informational for now: the upstream CalculiX model does
    not yet exactly match the internal geometry, loading or supports.
    """
    internal_table = _as_internal_table(internal)
    external_points = sorted(external_table.points, key=lambda point: point.size, reverse=True)
    internal_points = sorted(internal_table.points, key=lambda point: point.size, reverse=True)
    point_count = min(len(external_points), len(internal_points))

    points: List[ShellBenchmarkComparisonPoint] = []
    for external_point, internal_point in zip(external_points[:point_count], internal_points[:point_count]):
        points.append(
            ShellBenchmarkComparisonPoint(
                external_size=external_point.size,
                internal_size=internal_point.size,
                external_node_count=external_point.node_count,
                internal_node_count=internal_point.node_count,
                external_stress_normalized=external_point.stress_normalized,
                internal_stress_normalized=internal_point.stress_normalized,
                external_displacement_normalized=external_point.displacement_normalized,
                internal_displacement_normalized=internal_point.displacement_normalized,
                stress_normalized_delta=internal_point.stress_normalized - external_point.stress_normalized,
                displacement_normalized_delta=internal_point.displacement_normalized - external_point.displacement_normalized,
                stress_ratio_internal_to_external=_safe_ratio(internal_point.stress_normalized, external_point.stress_normalized),
                displacement_ratio_internal_to_external=_safe_ratio(
                    internal_point.displacement_normalized,
                    external_point.displacement_normalized,
                ),
            )
        )

    notes = (
        "Informational only: upstream CalculiX and internal benchmark geometry/load/supports are not yet identical.",
        "Rows are matched by coarse-to-fine order, not by exact mesh topology.",
    )
    return ShellBenchmarkComparison(
        external_element_type=external_table.element_type,
        internal_element_type=internal_table.element_type,
        points=tuple(points),
        notes=notes,
    )
