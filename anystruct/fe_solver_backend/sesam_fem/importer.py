"""SESAM formatted FEM importer for ANYintelligent FEModel objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Sequence

from .diagnostics import FemDiagnostic, SesamFemError, raise_if_errors
from .document import FemElement, FemMaterial, FemSection, SesamFemDocument, read_sesam_fem_document
from .schema import DOF_NAMES, get_element_spec


@dataclass(frozen=True)
class SesamFemImportResult:
    document: SesamFemDocument
    model: object | None
    diagnostics: tuple[FemDiagnostic, ...]
    element_count_by_type: dict[int, int]


def import_sesam_fem(
    path: str | Path,
    *,
    strict: bool = True,
    build_model: bool = True,
) -> SesamFemImportResult:
    """Import a SESAM formatted FEM file.

    The document layer is always populated.  With ``build_model=True`` the
    importer creates an FEModel for supported beam/shell topology, basic
    material/section data and simple nodal boundary flags.
    """

    document = read_sesam_fem_document(path, strict=strict)
    diagnostics = list(document.diagnostics)
    model = None
    if build_model:
        model, model_diagnostics = build_fe_model_from_sesam_document(document)
        diagnostics.extend(model_diagnostics)
    if strict:
        raise_if_errors(diagnostics, "SESAM FEM import failed")
    element_count_by_type: dict[int, int] = {}
    for element in document.elements.values():
        element_count_by_type[element.type_code] = element_count_by_type.get(element.type_code, 0) + 1
    return SesamFemImportResult(document, model, tuple(diagnostics), element_count_by_type)


def build_fe_model_from_sesam_document(document: SesamFemDocument) -> tuple[object, tuple[FemDiagnostic, ...]]:
    """Build a native FEModel from supported SESAM FEM document semantics."""

    from ..boundary import BoundaryCondition
    from ..elements import BeamElement, QuadraticBeamElement, ShellElement
    from ..fe_core import FEModel

    name = f"sesam:{document.source_path.name}" if document.source_path else "sesam:fem"
    model = FEModel(name=name)
    diagnostics: list[FemDiagnostic] = []
    material_names = _add_materials(model, document, diagnostics)

    for node in document.nodes.values():
        model.add_node(node.node_id, *node.coordinates)

    for element in document.elements.values():
        spec = get_element_spec(element.type_code)
        if spec is None:
            diagnostics.append(
                FemDiagnostic(
                    "FEM103",
                    f"unsupported SESAM element type {element.type_code}",
                    context={"element_id": element.element_id},
                )
            )
            continue
        missing_nodes = [node_id for node_id in element.node_ids if node_id not in model.mesh.nodes]
        if missing_nodes:
            diagnostics.append(
                FemDiagnostic(
                    "FEM105",
                    f"element {element.element_id} references missing nodes {missing_nodes}",
                    context={"element_id": element.element_id, "missing_nodes": missing_nodes},
                )
            )
            continue
        material_name = material_names.get(element.material_id or 0, "default")
        try:
            if spec.is_shell:
                thickness = _shell_thickness(document.sections.get(element.section_id or 0))
                solver_element = ShellElement(
                    element.element_id,
                    list(element.node_ids),
                    material_name=material_name,
                    thickness=thickness,
                )
            elif spec.type_code == 23:
                cross_section = _beam_section(document.sections.get(element.section_id or 0))
                _apply_beam_orientation(cross_section, document, element)
                solver_element = QuadraticBeamElement(
                    element.element_id,
                    list(element.node_ids),
                    material_name=material_name,
                    cross_section=cross_section,
                )
            else:
                cross_section = _beam_section(document.sections.get(element.section_id or 0))
                _apply_beam_orientation(cross_section, document, element)
                solver_element = BeamElement(
                    element.element_id,
                    list(element.node_ids),
                    material_name=material_name,
                    cross_section=cross_section,
                )
        except Exception as exc:  # Element constructors perform useful validation.
            diagnostics.append(
                FemDiagnostic(
                    "FEM130",
                    f"could not construct element {element.element_id}: {exc}",
                    context={"element_id": element.element_id},
                )
            )
            continue
        _attach_sesam_element_metadata(solver_element, document, element)
        model.add_element(element.element_id, solver_element)

    for index, boundary in enumerate(document.boundaries, start=1):
        if boundary.node_id not in model.mesh.nodes:
            diagnostics.append(
                FemDiagnostic(
                    "FEM105",
                    f"boundary references missing node {boundary.node_id}",
                    context={"node_id": boundary.node_id},
                )
            )
            continue
        constraints = {
            dof_name: 0.0
            for dof_name, flag in zip(DOF_NAMES, boundary.dof_flags)
            if int(flag) != 0
        }
        if constraints:
            model.add_boundary_condition(
                BoundaryCondition(f"sesam_BNBCD_{index}", [boundary.node_id], constraints)
            )

    if document.load_records:
        from ..boundary import LoadCase
        load_case = LoadCase("sesam_imported_load")
        has_loads = False
        shell_element_ids = {eid for eid, el in model.mesh.elements.items() if el.__class__.__name__ == "ShellElement"}

        pressure_loads = {}
        for load_record in document.load_records:
            if load_record.record_name == "BEUSLO" and len(load_record.raw_values) >= 9:
                element_id = int(load_record.raw_values[4])
                if element_id in shell_element_ids:
                    # load values start at index 8
                    import math
                    load_values = [float(v) for v in load_record.raw_values[8:] if math.isfinite(float(v))]
                    if load_values:
                        pressure_loads[element_id] = pressure_loads.get(element_id, 0.0) + sum(load_values) / len(load_values)

            elif load_record.record_name == "BGRAV" and len(load_record.raw_values) >= 7:
                gx = float(load_record.raw_values[-3])
                gy = float(load_record.raw_values[-2])
                gz = float(load_record.raw_values[-1])
                if abs(gx) > 0 or abs(gy) > 0 or abs(gz) > 0:
                    load_case.set_gravity(gx, gy, gz)
                    has_loads = True

        for element_id, pressure in sorted(pressure_loads.items()):
            if pressure != 0.0:
                load_case.add_pressure_load(element_id, pressure)
                has_loads = True

        if has_loads:
            model.add_load_case(load_case)
        else:
            diagnostics.append(
                FemDiagnostic(
                    "FEM121",
                    "SESAM load records were found but yielded no active loads.",
                    severity="warning",
                    context={"load_records": len(document.load_records)},
                )
            )
    if document.dependencies:
        diagnostics.append(
            FemDiagnostic(
                "FEM122",
                "SESAM dependency records are preserved but not translated into solver MPCs yet",
                severity="warning",
                context={"dependency_records": len(document.dependencies)},
            )
        )

    setattr(model, "sesam_document", document)
    setattr(model, "sesam_import_diagnostics", tuple(diagnostics))
    return model, tuple(diagnostics)


def _element_transform_ids(document: SesamFemDocument, element: FemElement) -> tuple[int, ...]:
    reference = document.element_references.get(element.element_id)
    if reference is None:
        return ()
    if reference.nodal_transform_ids:
        return tuple(reference.nodal_transform_ids)
    if reference.transform_id is not None:
        return (reference.transform_id,)
    return ()


def _normalise_vector(values: Sequence[float]) -> tuple[float, float, float] | None:
    if len(values) < 3:
        return None
    x, y, z = (float(values[0]), float(values[1]), float(values[2]))
    length = (x * x + y * y + z * z) ** 0.5
    if length <= 1.0e-12:
        return None
    return (x / length, y / length, z / length)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, float, float] | None:
    if not vectors:
        return None
    return _normalise_vector(
        (
            sum(float(vector[0]) for vector in vectors),
            sum(float(vector[1]) for vector in vectors),
            sum(float(vector[2]) for vector in vectors),
        )
    )


def _beam_orientation_vector(document: SesamFemDocument, transform_ids: Sequence[int]) -> tuple[float, float, float] | None:
    vectors = []
    for transform_id in transform_ids:
        unit_vector = document.unit_vectors.get(transform_id)
        if unit_vector is not None:
            vectors.append(unit_vector.vector)
            continue
        transform = document.coordinate_transforms.get(transform_id)
        if transform is not None:
            vectors.append(transform.matrix[2])
    return _mean_vector(vectors)


def _apply_beam_orientation(
    cross_section: dict[str, object],
    document: SesamFemDocument,
    element: FemElement,
) -> None:
    orientation = _beam_orientation_vector(document, _element_transform_ids(document, element))
    if orientation is not None:
        cross_section["orientation"] = orientation


def _mean_coordinate_transform_axes(
    document: SesamFemDocument,
    transform_ids: Sequence[int],
) -> dict[str, tuple[float, float, float]] | None:
    rows = {"x": [], "y": [], "z": []}
    for transform_id in transform_ids:
        transform = document.coordinate_transforms.get(transform_id)
        if transform is None:
            continue
        rows["x"].append(transform.matrix[0])
        rows["y"].append(transform.matrix[1])
        rows["z"].append(transform.matrix[2])
    axes = {name: _mean_vector(vectors) for name, vectors in rows.items()}
    if any(value is None for value in axes.values()):
        return None
    return {name: value for name, value in axes.items() if value is not None}


def _attach_sesam_element_metadata(solver_element: object, document: SesamFemDocument, element: FemElement) -> None:
    transform_ids = _element_transform_ids(document, element)
    reference = document.element_references.get(element.element_id)
    if reference is not None:
        setattr(solver_element, "sesam_reference", reference)
    if not transform_ids:
        return
    setattr(solver_element, "sesam_transform_ids", transform_ids)
    spec = get_element_spec(element.type_code)
    if spec is not None and spec.is_shell:
        axes = _mean_coordinate_transform_axes(document, transform_ids)
        if axes is not None:
            setattr(solver_element, "sesam_local_axes", axes)


def _add_materials(
    model: object,
    document: SesamFemDocument,
    diagnostics: list[FemDiagnostic],
) -> dict[int, str]:
    material_names: dict[int, str] = {0: "default"}
    for material_id, material in document.materials.items():
        name = _material_name(material)
        material_names[material_id] = name
        elastic_modulus = material.elastic_modulus or 210.0e9
        poisson_ratio = material.poisson_ratio if material.poisson_ratio is not None else 0.3
        density = material.density or 0.0
        yield_stress = material.yield_stress or 0.0
        if elastic_modulus < 1.0e9:
            diagnostics.append(
                FemDiagnostic(
                    "FEM123",
                    f"material {material_id} elastic modulus is unusually small for SI units",
                    severity="warning",
                    context={"material_id": material_id, "elastic_modulus": elastic_modulus},
                )
            )
        model.add_material(
            name,
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            density=density,
            yield_stress=yield_stress,
        )
    return material_names


def _material_name(material: FemMaterial) -> str:
    suffix = _safe_name(material.name) if material.name else str(material.material_id)
    return f"sesam_material_{suffix}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def _shell_thickness(section: Optional[FemSection]) -> float:
    if section is not None and section.thickness is not None and section.thickness > 0.0:
        return float(section.thickness)
    return 0.01


def _beam_section(section: Optional[FemSection]) -> dict[str, object]:
    if section is None:
        return {}
    data: dict[str, object] = {}
    if section.area is not None:
        data["area"] = float(section.area)
    if section.iy is not None:
        data["Iy"] = float(section.iy)
    if section.iz is not None:
        data["Iz"] = float(section.iz)
    if section.torsion is not None:
        data["J"] = float(section.torsion)
    if section.web_height is not None:
        data["web_height"] = float(section.web_height)
    if section.web_thickness is not None:
        data["web_thickness"] = float(section.web_thickness)
    if section.flange_width is not None:
        data["flange_width"] = float(section.flange_width)
    if section.flange_thickness is not None:
        data["flange_thickness"] = float(section.flange_thickness)
    if data.get("flange_width", 0.0) > 0.0:
        data["section_type"] = "T"
    return data
