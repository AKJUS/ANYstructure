"""SESAM/GeniE FEM/SIF import helpers for the ANYstructure FE backend."""

from __future__ import annotations

import importlib
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


Point3D = tuple[float, float, float]
Vector3D = tuple[float, float, float]


SHELL_TYPE_BY_CODE: dict[int, str] = {
    24: "S4",
    25: "S3",
    26: "S6",
    28: "S8",
}
BEAM_TYPE_BY_CODE: dict[int, str] = {
    15: "B2",
    23: "B3",
}
ELEMENT_NODE_COUNT_BY_TYPE: dict[int, int] = {
    15: 2,
    23: 3,
    24: 4,
    25: 3,
    26: 6,
    28: 8,
}

_TAG_RE = re.compile(r"^[A-Z][A-Z0-9_+-]*$")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?"
)
_DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


class SesamSinSupportError(RuntimeError):
    """Raised when a binary SESAM SIN file cannot be converted to SIF."""


@dataclass(frozen=True)
class SesamRecord:
    tag: str
    values: tuple[float, ...]
    line_number: int


@dataclass(frozen=True)
class SesamMaterial:
    material_id: int
    elastic_modulus: float = 210.0e9
    poisson_ratio: float = 0.3
    density: float = 7850.0
    yield_stress: float = 355.0e6
    raw_values: tuple[float, ...] = ()

    @property
    def name(self) -> str:
        return f"sesam_material_{self.material_id}"


@dataclass(frozen=True)
class SesamElement:
    element_id: int
    type_code: int
    node_ids: tuple[int, ...]
    material_id: int | None = None
    property_id: int | None = None
    geometry_values: tuple[float, ...] = ()
    reference_values: tuple[float, ...] = ()

    @property
    def element_type(self) -> str:
        return SHELL_TYPE_BY_CODE.get(self.type_code) or BEAM_TYPE_BY_CODE.get(self.type_code) or f"T{self.type_code}"

    @property
    def is_shell(self) -> bool:
        return self.type_code in SHELL_TYPE_BY_CODE

    @property
    def is_beam(self) -> bool:
        return self.type_code in BEAM_TYPE_BY_CODE

    @property
    def corner_node_ids(self) -> tuple[int, ...]:
        if self.type_code == 28 and len(self.node_ids) >= 4:
            return self.node_ids[:4]
        if self.type_code == 26 and len(self.node_ids) >= 3:
            return self.node_ids[:3]
        return self.node_ids[:4]


@dataclass(frozen=True)
class SesamBoundary:
    node_id: int
    dof_flags: tuple[bool, bool, bool, bool, bool, bool]
    raw_values: tuple[float, ...] = ()

    @property
    def constraints(self) -> dict[str, float]:
        return {name: 0.0 for name, constrained in zip(_DOF_NAMES, self.dof_flags) if constrained}


@dataclass(frozen=True)
class SesamModel:
    path: str
    nodes: dict[int, Point3D]
    elements: dict[int, SesamElement]
    materials: dict[int, SesamMaterial] = field(default_factory=dict)
    shell_thicknesses: dict[int, float] = field(default_factory=dict)
    beam_sections: dict[int, dict[str, float]] = field(default_factory=dict)
    boundaries: tuple[SesamBoundary, ...] = ()
    pressure_loads: dict[int, float] = field(default_factory=dict)
    gravity: Vector3D | None = None
    record_counts: dict[str, int] = field(default_factory=dict)

    @property
    def shell_elements(self) -> dict[int, SesamElement]:
        return {element_id: element for element_id, element in self.elements.items() if element.is_shell}

    @property
    def beam_elements(self) -> dict[int, SesamElement]:
        return {element_id: element for element_id, element in self.elements.items() if element.is_beam}


@dataclass(frozen=True)
class SesamStressResult:
    path: str
    nodes: dict[int, Point3D]
    element_nodes: dict[int, tuple[int, ...]]
    components: tuple[str, ...]
    nodal_stress: dict[int, tuple[float, ...]]
    element_stress: dict[int, tuple[float, ...]]
    units: str = "Pa"


@dataclass(frozen=True)
class SesamFEImport:
    source_path: str
    sesam_model: SesamModel
    generated_geometry: dict[str, Any]
    fe_model: Any
    load_case: Any | None = None


@dataclass(frozen=True)
class SesamStaticRun:
    source_path: str
    import_result: SesamFEImport
    displacements: np.ndarray
    solver_info: dict[str, Any]
    status: str
    max_translation: float


def _line_tag(line: str) -> str | None:
    tag = line[:10].strip()
    return tag if _TAG_RE.match(tag) else None


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER_RE.finditer(text)]


def _as_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None


def _is_sin_path(path: str | os.PathLike[str]) -> bool:
    return Path(str(path)).suffix.lower() == ".sin"


@contextmanager
def _sesam_text_record_path(path: str | os.PathLike[str]) -> Iterator[str]:
    """Yield a text SIF/FEM path, converting binary SIN via optional DNV SifIO."""

    path_text = str(path)
    if not _is_sin_path(path_text):
        yield path_text
        return

    with tempfile.TemporaryDirectory(prefix="anystructure_sin_") as temp_dir:
        converted_path = str(Path(temp_dir) / (Path(path_text).stem + ".SIF"))
        convert_sin_to_sif_with_dnv_sifio(path_text, converted_path)
        yield converted_path


def convert_sin_to_sif_with_dnv_sifio(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Convert a binary SESAM ``.SIN`` file to ASCII ``.SIF`` using optional DNV SifIO.

    The dependency is intentionally loaded with dynamic imports so PyInstaller
    does not bundle the DNV .NET runtime unless a deployment explicitly opts in.
    """

    source = Path(str(source_path))
    if output_path is None:
        handle, temp_name = tempfile.mkstemp(prefix=source.stem + "_", suffix=".SIF")
        os.close(handle)
        output = Path(temp_name)
    else:
        output = Path(str(output_path))
        output.parent.mkdir(parents=True, exist_ok=True)

    try:
        importlib.import_module("dnv.net.runtime")
        sifio_module = importlib.import_module("dnv.sesam.sifapi.io")
    except Exception as exc:
        raise SesamSinSupportError(
            "SESAM .SIN is a binary database. Install optional packages "
            "'dnv-sifio' and 'dnv-net-runtime' to convert it, or export the "
            "case from GeniE/SESTRA as .SIF/.FEM before importing."
        ) from exc

    factory = getattr(sifio_module, "SesamDataFactory", None)
    if factory is None:
        raise SesamSinSupportError("dnv-sifio is installed, but SesamDataFactory is not available")

    temp_sin = output.with_suffix(".anystructure_tmp.SIN")
    try:
        with factory.CreateReader(str(source)) as reader:
            diagnostic = reader.CreateModel()
            if int(diagnostic) != 0:
                raise SesamSinSupportError(f"DNV SifIO could not create a model from {source}: diagnostic={diagnostic}")
            with factory.CreateWriter(str(temp_sin)) as writer:
                try:
                    writer.WriteSifFile(reader, str(output))
                except TypeError:
                    writer.WriteSifFile(str(output))
    except SesamSinSupportError:
        raise
    except Exception as exc:
        raise SesamSinSupportError(f"DNV SifIO failed to convert {source} to SIF: {exc}") from exc
    finally:
        try:
            temp_sin.unlink()
        except OSError:
            pass

    return str(output)


def iter_sesam_records(path: str | os.PathLike[str]) -> Iterator[SesamRecord]:
    """Yield SESAM records with numeric continuations appended."""

    current_tag: str | None = None
    current_values: list[float] = []
    current_line = 0

    def flush() -> SesamRecord | None:
        if current_tag is None:
            return None
        return SesamRecord(current_tag, tuple(current_values), current_line)

    with open(path, "r", encoding="utf-8", errors="ignore") as sesam_file:
        for line_number, raw_line in enumerate(sesam_file, start=1):
            tag = _line_tag(raw_line)
            if tag is not None:
                record = flush()
                if record is not None:
                    yield record
                current_tag = tag
                current_values = _numbers(raw_line[10:])
                current_line = line_number
                continue

            if current_tag is not None:
                current_values.extend(_numbers(raw_line))

    record = flush()
    if record is not None:
        yield record


def read_sesam_model(path: str | os.PathLike[str]) -> SesamModel:
    """Read SESAM FEM/SIF geometry, properties, supports and simple loads."""

    path_text = str(path)
    nodes: dict[int, Point3D] = {}
    geometry_records: dict[int, tuple[float, ...]] = {}
    reference_records: dict[int, tuple[float, ...]] = {}
    materials: dict[int, SesamMaterial] = {}
    shell_thicknesses: dict[int, float] = {}
    beam_sections: dict[int, dict[str, float]] = {}
    boundaries: list[SesamBoundary] = []
    pressure_loads: defaultdict[int, float] = defaultdict(float)
    gravity: Vector3D | None = None
    record_counts: Counter[str] = Counter()

    with _sesam_text_record_path(path_text) as record_path:
        for record in iter_sesam_records(record_path):
            record_counts[record.tag] += 1
            values = record.values
            if record.tag == "GCOORD" and len(values) >= 4:
                node_id = _as_int(values[0])
                if node_id is not None:
                    nodes[node_id] = (float(values[1]), float(values[2]), float(values[3]))
            elif record.tag == "GELMNT1" and len(values) >= 5:
                element_id = _as_int(values[0])
                if element_id is not None:
                    geometry_records[element_id] = values
            elif record.tag == "GELREF1" and values:
                element_id = _as_int(values[0])
                if element_id is not None:
                    reference_records[element_id] = values
            elif record.tag == "MISOSEL" and len(values) >= 4:
                material_id = _as_int(values[0])
                if material_id is not None:
                    materials[material_id] = SesamMaterial(
                        material_id=material_id,
                        elastic_modulus=float(values[1]),
                        poisson_ratio=float(values[2]),
                        density=float(values[3]),
                        yield_stress=float(values[7]) if len(values) > 7 and values[7] > 0.0 else 355.0e6,
                        raw_values=values,
                    )
            elif record.tag == "GELTH" and len(values) >= 2:
                property_id = _as_int(values[0])
                if property_id is not None and values[1] > 0.0:
                    shell_thicknesses[property_id] = float(values[1])
            elif record.tag == "GBEAMG" and len(values) >= 6:
                section_id = _as_int(values[0])
                if section_id is not None:
                    beam_sections[section_id] = _beam_section_from_gbeamg(values)
            elif record.tag == "BNBCD" and len(values) >= 8:
                node_id = _as_int(values[0])
                if node_id is not None:
                    flags = tuple(abs(float(value)) > 0.0 for value in values[2:8])
                    boundaries.append(SesamBoundary(node_id, flags, raw_values=values))  # type: ignore[arg-type]
            elif record.tag == "BEUSLO" and len(values) >= 9:
                element_id = _as_int(values[4])
                if element_id is not None:
                    load_values = [float(value) for value in values[8:] if math.isfinite(float(value))]
                    if load_values:
                        pressure_loads[element_id] += sum(load_values) / len(load_values)
            elif record.tag == "BGRAV" and len(values) >= 7:
                gravity = (float(values[-3]), float(values[-2]), float(values[-1]))

    elements: dict[int, SesamElement] = {}
    for element_id, values in geometry_records.items():
        type_code = _as_int(values[2]) if len(values) > 2 else None
        if type_code is None:
            continue
        node_count = ELEMENT_NODE_COUNT_BY_TYPE.get(type_code)
        raw_node_ids = [
            node_id
            for node_id in (_as_int(value) for value in values[4:])
            if node_id is not None and node_id > 0
        ]
        node_ids = tuple(raw_node_ids[:node_count] if node_count is not None else raw_node_ids)
        if not node_ids:
            continue
        ref_values = reference_records.get(element_id, ())
        material_id = _as_int(ref_values[1]) if len(ref_values) > 1 else None
        property_id = _as_int(ref_values[8]) if len(ref_values) > 8 else None
        if property_id == 0:
            property_id = None
        elements[element_id] = SesamElement(
            element_id=element_id,
            type_code=type_code,
            node_ids=node_ids,
            material_id=material_id,
            property_id=property_id,
            geometry_values=values,
            reference_values=ref_values,
        )

    return SesamModel(
        path=path_text,
        nodes=nodes,
        elements=elements,
        materials=materials,
        shell_thicknesses=shell_thicknesses,
        beam_sections=beam_sections,
        boundaries=tuple(boundaries),
        pressure_loads=dict(pressure_loads),
        gravity=gravity,
        record_counts=dict(record_counts),
    )


def _beam_section_from_gbeamg(values: Sequence[float]) -> dict[str, float]:
    area = _positive_at(values, 2, 0.01)
    iy = _positive_at(values, 4, 1.0e-8)
    iz = _positive_at(values, 5, 1.0e-8)
    torsion = _positive_at(values, 3, max(iy + iz, 1.0e-8))
    section = {
        "area": area,
        "Iy": iy,
        "Iz": iz,
        "J": torsion,
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
    }
    if len(values) > 14 and abs(values[14]) > 0.0:
        section["c_y"] = abs(float(values[14]))
    if len(values) > 15 and abs(values[15]) > 0.0:
        section["c_z"] = abs(float(values[15]))
    return section


def _positive_at(values: Sequence[float], index: int, default: float) -> float:
    if len(values) <= index:
        return default
    value = float(values[index])
    return value if value > 0.0 else default


def sesam_model_to_generated_geometry(model: SesamModel) -> dict[str, Any]:
    """Convert a SESAM model to the generated-geometry shape used by the backend."""

    materials = [
        {
            "id": material.name,
            "name": material.name,
            "elastic_modulus": material.elastic_modulus,
            "poisson_ratio": material.poisson_ratio,
            "density": material.density,
            "yield_stress": material.yield_stress,
        }
        for material in sorted(model.materials.values(), key=lambda item: item.material_id)
    ]

    shells = []
    for element in sorted(model.shell_elements.values(), key=lambda item: item.element_id):
        shells.append(
            {
                "id": element.element_id,
                "node_ids": list(element.node_ids),
                "type": element.element_type,
                "thickness": _shell_thickness(model, element),
                "material": _material_name(element.material_id),
                "sesam_type": element.type_code,
                "property_id": element.property_id,
            }
        )

    beams = []
    for element in sorted(model.beam_elements.values(), key=lambda item: item.element_id):
        beams.append(
            {
                "id": element.element_id,
                "node_ids": list(element.node_ids),
                "type": element.element_type,
                "cross_section": _beam_section(model, element),
                "material": _material_name(element.material_id),
                "role": "beam",
                "sesam_type": element.type_code,
                "property_id": element.property_id,
            }
        )

    supports = []
    for index, boundary in enumerate(model.boundaries, start=1):
        constraints = boundary.constraints
        if constraints:
            supports.append(
                {
                    "name": f"sesam_support_{index}",
                    "node_ids": [boundary.node_id],
                    "dof_constraints": constraints,
                }
            )

    return {
        "name": f"SESAM import: {Path(model.path).name}",
        "source": model.path,
        "nodes": [
            {"id": node_id, "coords": list(coords)}
            for node_id, coords in sorted(model.nodes.items())
        ],
        "shells": shells,
        "beams": beams,
        "supports": supports,
        "materials": materials,
        "sesam": {
            "record_counts": dict(model.record_counts),
            "shell_element_count": len(shells),
            "beam_element_count": len(beams),
            "pressure_load_count": len(model.pressure_loads),
            "gravity": model.gravity,
        },
    }


def _material_name(material_id: int | None) -> str:
    return f"sesam_material_{material_id}" if material_id is not None else "steel"


def _shell_thickness(model: SesamModel, element: SesamElement) -> float:
    if element.property_id is not None:
        thickness = model.shell_thicknesses.get(element.property_id)
        if thickness is not None and thickness > 0.0:
            return float(thickness)
    if model.shell_thicknesses:
        return float(next(iter(model.shell_thicknesses.values())))
    return 0.01


def _beam_section(model: SesamModel, element: SesamElement) -> dict[str, float]:
    if element.property_id is not None and element.property_id in model.beam_sections:
        return dict(model.beam_sections[element.property_id])
    if model.beam_sections:
        return dict(next(iter(model.beam_sections.values())))
    return {
        "area": 0.01,
        "Iy": 1.0e-8,
        "Iz": 1.0e-8,
        "J": 1.0e-8,
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
    }


def sesam_load_case(model: SesamModel) -> Any | None:
    """Build a backend load case from BEUSLO/BGRAV records."""

    from .fe_solver_backend import LoadCase

    load_case = LoadCase("sesam_imported_load")
    has_loads = False
    shell_element_ids = set(model.shell_elements)
    for element_id, pressure in sorted(model.pressure_loads.items()):
        if element_id in shell_element_ids and pressure != 0.0:
            load_case.add_pressure_load(element_id, pressure)
            has_loads = True
    if model.gravity is not None and any(abs(component) > 0.0 for component in model.gravity):
        load_case.set_gravity(*model.gravity)
        has_loads = True
    return load_case if has_loads else None


def build_fe_model_from_sesam_fem(
    path: str | os.PathLike[str],
    config: Any | None = None,
) -> SesamFEImport:
    """Import a SESAM FEM/SIF model into the ANYstructure FEModel backend."""

    from .fe_solver_backend import AnyStructureFEMConfig, build_fe_model_from_generated_geometry

    sesam_model = read_sesam_model(path)
    generated_geometry = sesam_model_to_generated_geometry(sesam_model)
    backend_config = config or AnyStructureFEMConfig(require_idealized_member_beams=False)
    fe_model = build_fe_model_from_generated_geometry(generated_geometry, backend_config)
    load_case = sesam_load_case(sesam_model)
    if load_case is not None:
        fe_model.add_load_case(load_case)
    return SesamFEImport(
        source_path=str(path),
        sesam_model=sesam_model,
        generated_geometry=generated_geometry,
        fe_model=fe_model,
        load_case=load_case,
    )


def run_sesam_fem_static(
    path: str | os.PathLike[str],
    config: Any | None = None,
    *,
    solver_type: str = "direct",
    constraint_mode: str = "auto",
) -> SesamStaticRun:
    """Import a SESAM FEM/SIF model and run one linear static solve."""

    from .fe_solver_backend import solve_linear

    imported = build_fe_model_from_sesam_fem(path, config)
    displacements, solver_info = solve_linear(
        imported.fe_model,
        imported.load_case,
        solver_type=solver_type,
        constraint_mode=constraint_mode,
    )
    convergence = solver_info.get("convergence_info", {})
    status = str(convergence.get("status", "unknown"))
    return SesamStaticRun(
        source_path=str(path),
        import_result=imported,
        displacements=displacements,
        solver_info=solver_info,
        status=status,
        max_translation=_max_translation(displacements),
    )


def _max_translation(displacements: np.ndarray) -> float:
    vector = np.asarray(displacements, dtype=float).reshape(-1)
    if vector.size < 3:
        return 0.0
    if vector.size % 6 == 0:
        translations = vector.reshape((-1, 6))[:, :3]
        return float(np.max(np.linalg.norm(translations, axis=1))) if translations.size else 0.0
    return float(np.max(np.abs(vector)))


def read_sesam_sif_stress(
    path: str | os.PathLike[str],
    model: SesamModel | None = None,
) -> SesamStressResult:
    """Read RVSTRESS shell results as FRD-like global stress tensors."""

    path_text = str(path)
    model = model or read_sesam_model(path_text)
    nodal_sums: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=float))
    nodal_counts: defaultdict[int, int] = defaultdict(int)
    element_stress: dict[int, tuple[float, ...]] = {}

    with _sesam_text_record_path(path_text) as record_path:
        for record in iter_sesam_records(record_path):
            if record.tag != "RVSTRESS" or len(record.values) < 8:
                continue
            values = record.values
            element_id = _as_int(values[2])
            type_code = _as_int(values[4])
            if element_id is None:
                continue
            element = model.shell_elements.get(element_id)
            if element is None:
                continue
            if type_code is not None and type_code not in SHELL_TYPE_BY_CODE:
                continue
            local_stress = _mean_rvstress_triplets(values[5:])
            if local_stress is None:
                continue
            global_components = _local_membrane_to_global_components(model, element, local_stress)
            if global_components is None:
                continue
            element_stress[element_id] = global_components
            vector = np.asarray(global_components, dtype=float)
            for node_id in element.node_ids:
                nodal_sums[node_id] += vector
                nodal_counts[node_id] += 1

    nodal_stress = {
        node_id: tuple((values / max(nodal_counts[node_id], 1)).tolist())
        for node_id, values in nodal_sums.items()
    }
    element_nodes = {
        element_id: element.node_ids
        for element_id, element in model.shell_elements.items()
    }
    return SesamStressResult(
        path=path_text,
        nodes=dict(model.nodes),
        element_nodes=element_nodes,
        components=("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"),
        nodal_stress=nodal_stress,
        element_stress=element_stress,
    )


def _mean_rvstress_triplets(values: Sequence[float]) -> tuple[float, float, float] | None:
    triplets = [
        (float(values[index]), float(values[index + 1]), float(values[index + 2]))
        for index in range(0, len(values) - 2, 3)
        if all(math.isfinite(float(values[index + offset])) for offset in range(3))
    ]
    if not triplets:
        return None
    return (
        sum(item[0] for item in triplets) / len(triplets),
        sum(item[1] for item in triplets) / len(triplets),
        sum(item[2] for item in triplets) / len(triplets),
    )


def _local_membrane_to_global_components(
    model: SesamModel,
    element: SesamElement,
    local_stress: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float] | None:
    frame = _element_local_frame(model, element)
    if frame is None:
        return None
    local_x, local_y, _normal = frame
    sx, sy, txy = local_stress
    tensor = np.zeros((3, 3), dtype=float)
    x = np.asarray(local_x, dtype=float)
    y = np.asarray(local_y, dtype=float)
    tensor += sx * np.outer(x, x)
    tensor += sy * np.outer(y, y)
    tensor += txy * (np.outer(x, y) + np.outer(y, x))
    return (
        float(tensor[0, 0]),
        float(tensor[1, 1]),
        float(tensor[2, 2]),
        float(tensor[0, 1]),
        float(tensor[1, 2]),
        float(tensor[2, 0]),
    )


def _element_local_frame(model: SesamModel, element: SesamElement) -> tuple[Vector3D, Vector3D, Vector3D] | None:
    corner_ids = element.corner_node_ids
    points = [model.nodes[node_id] for node_id in corner_ids if node_id in model.nodes]
    if len(points) < 3:
        return None
    p0 = np.asarray(points[0], dtype=float)
    p1 = np.asarray(points[1], dtype=float)
    x_axis = _normalise_array(p1 - p0)
    if x_axis is None:
        return None
    normal = None
    for index in range(1, len(points) - 1):
        candidate = _normalise_array(
            np.cross(
                np.asarray(points[index], dtype=float) - p0,
                np.asarray(points[index + 1], dtype=float) - p0,
            )
        )
        if candidate is not None:
            normal = candidate
            break
    if normal is None:
        return None
    y_axis = _normalise_array(np.cross(normal, x_axis))
    if y_axis is None:
        return None
    return tuple(x_axis.tolist()), tuple(y_axis.tolist()), tuple(normal.tolist())  # type: ignore[return-value]


def _normalise_array(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        return None
    return vector / length


def read_sesam_sif_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return lightweight metadata for a SESAM SIF/FEM-style file."""

    path_text = str(path)
    counts: Counter[str] = Counter(record.tag for record in iter_sesam_records(path_text))
    model = read_sesam_model(path_text)
    result_blocks = [
        {
            "name": tag,
            "count": count,
            "components": ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"] if tag == "RVSTRESS" else [],
        }
        for tag, count in sorted(counts.items())
        if tag.startswith("RV") or tag in {"RDPOINTS"}
    ]
    return {
        "path": path_text,
        "file_size": os.path.getsize(path_text),
        "format": "SESAM SIF/FEM",
        "node_count": len(model.nodes),
        "element_count": len(model.elements),
        "shell_element_count": len(model.shell_elements),
        "beam_element_count": len(model.beam_elements),
        "materials": [material.name for material in sorted(model.materials.values(), key=lambda item: item.material_id)],
        "record_counts": dict(counts),
        "result_blocks": result_blocks,
    }


def sesam_model_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    model = read_sesam_model(path)
    type_counts = Counter(element.element_type for element in model.elements.values())
    return {
        "path": str(path),
        "node_count": len(model.nodes),
        "element_count": len(model.elements),
        "shell_element_count": len(model.shell_elements),
        "beam_element_count": len(model.beam_elements),
        "element_type_counts": dict(sorted(type_counts.items())),
        "material_count": len(model.materials),
        "shell_thickness_count": len(model.shell_thicknesses),
        "beam_section_count": len(model.beam_sections),
        "support_count": len(model.boundaries),
        "pressure_load_count": len(model.pressure_loads),
        "gravity": model.gravity,
        "record_counts": dict(model.record_counts),
    }


def element_type_counts(elements: Iterable[SesamElement] | Mapping[int, SesamElement]) -> dict[str, int]:
    values = elements.values() if isinstance(elements, Mapping) else elements
    return dict(Counter(element.element_type for element in values))
