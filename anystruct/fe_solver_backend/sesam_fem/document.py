"""Layer B typed SESAM formatted FEM document model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

from .diagnostics import FemDiagnostic, SesamFemError, raise_if_errors
from .records import FemRawRecord, read_raw_records, strict_int
from .schema import SUPPORTED_RECORDS, classify_record, get_element_spec


@dataclass(frozen=True)
class FemHeader:
    ident_values: tuple[float, ...] = ()
    ident_text: tuple[str, ...] = ()
    date_text: tuple[str, ...] = ()
    unit_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FemMaterial:
    material_id: int
    name: Optional[str] = None
    elastic_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    density: Optional[float] = None
    yield_stress: Optional[float] = None
    raw_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FemSection:
    section_id: int
    kind: str
    name: Optional[str] = None
    thickness: Optional[float] = None
    area: Optional[float] = None
    iy: Optional[float] = None
    iz: Optional[float] = None
    torsion: Optional[float] = None
    raw_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FemConceptRecord:
    record_name: str
    concept_id: Optional[int]
    text: tuple[str, ...]
    raw_values: tuple[float, ...]


@dataclass(frozen=True)
class FemCoordinate:
    coordinate_id: int
    raw_values: tuple[float, ...]


@dataclass(frozen=True)
class FemNode:
    node_id: int
    coordinates: tuple[float, float, float]
    coordinate_system_id: int = 0
    raw_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FemElementReference:
    element_id: int
    material_id: Optional[int]
    section_id: Optional[int]
    raw_values: tuple[float, ...]


@dataclass(frozen=True)
class FemElement:
    element_id: int
    type_code: int
    topology: str
    node_ids: tuple[int, ...]
    material_id: Optional[int] = None
    section_id: Optional[int] = None
    raw_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FemBoundary:
    node_id: int
    dof_flags: tuple[int, ...]
    prescribed_values: tuple[float, ...]
    raw_values: tuple[float, ...]


@dataclass(frozen=True)
class FemDependency:
    record_name: str
    raw_values: tuple[float, ...]
    text: tuple[str, ...]


@dataclass(frozen=True)
class FemLoadRecord:
    record_name: str
    load_case_id: Optional[int]
    target_id: Optional[int]
    raw_values: tuple[float, ...]
    text: tuple[str, ...]


@dataclass(frozen=True)
class SesamFemDocument:
    source_path: Optional[Path]
    header: FemHeader
    raw_records: tuple[FemRawRecord, ...]
    record_counts: Mapping[str, int]
    materials: Mapping[int, FemMaterial]
    sections: Mapping[int, FemSection]
    concepts: tuple[FemConceptRecord, ...]
    coordinate_systems: Mapping[int, FemCoordinate]
    nodes: Mapping[int, FemNode]
    elements: Mapping[int, FemElement]
    element_references: Mapping[int, FemElementReference]
    boundaries: tuple[FemBoundary, ...]
    dependencies: tuple[FemDependency, ...]
    load_records: tuple[FemLoadRecord, ...]
    unknown_records: tuple[FemRawRecord, ...]
    diagnostics: tuple[FemDiagnostic, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "record_counts": dict(self.record_counts),
            "materials": len(self.materials),
            "sections": len(self.sections),
            "nodes": len(self.nodes),
            "elements": len(self.elements),
            "element_types": dict(Counter(element.type_code for element in self.elements.values())),
            "boundaries": len(self.boundaries),
            "dependencies": len(self.dependencies),
            "loads": len(self.load_records),
            "unknown_records": len(self.unknown_records),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
        }


def read_sesam_fem_document(path: str | Path, *, strict: bool = True) -> SesamFemDocument:
    """Read a SESAM formatted sequential FEM document."""

    raw_records = read_raw_records(path, strict=strict)
    document = parse_sesam_fem_records(raw_records, source_path=Path(path), strict=False)
    from .validation import validate_sesam_fem_document

    diagnostics = tuple(document.diagnostics) + validate_sesam_fem_document(document)
    document = replace(document, diagnostics=diagnostics)
    if strict:
        raise_if_errors(diagnostics, "SESAM FEM document failed validation")
    return document


def parse_sesam_fem_records(
    raw_records: tuple[FemRawRecord, ...],
    *,
    source_path: str | Path | None = None,
    strict: bool = True,
) -> SesamFemDocument:
    """Parse raw records into the typed SESAM FEM document model."""

    diagnostics: list[FemDiagnostic] = []
    header = _parse_header(raw_records)
    record_counts = Counter(record.name for record in raw_records)
    materials, material_names = _parse_materials(raw_records, diagnostics)
    sections, section_names = _parse_sections(raw_records, diagnostics)
    materials = {
        key: replace(value, name=material_names.get(key, value.name))
        for key, value in materials.items()
    }
    sections = {
        key: replace(value, name=section_names.get(key, value.name))
        for key, value in sections.items()
    }

    coordinate_systems = _parse_coordinates(raw_records, diagnostics)
    nodes = _parse_nodes(raw_records, diagnostics)
    element_references = _parse_element_references(raw_records, diagnostics)
    elements = _parse_elements(raw_records, element_references, diagnostics)
    boundaries = _parse_boundaries(raw_records, diagnostics)
    concepts = _parse_concepts(raw_records, diagnostics)
    dependencies = _parse_dependencies(raw_records)
    load_records = _parse_loads(raw_records, diagnostics)

    unknown_records = tuple(record for record in raw_records if record.name not in SUPPORTED_RECORDS)
    for record in unknown_records:
        diagnostics.append(
            FemDiagnostic(
                "FEM110",
                f"unknown FEM record preserved as raw data: {record.name}",
                severity="warning",
                record_name=record.name,
                line_start=record.source_line_start,
                line_end=record.source_line_end,
                context={"classification": classify_record(record.name)},
            )
        )

    document = SesamFemDocument(
        source_path=Path(source_path) if source_path is not None else None,
        header=header,
        raw_records=raw_records,
        record_counts=dict(record_counts),
        materials=materials,
        sections=sections,
        concepts=concepts,
        coordinate_systems=coordinate_systems,
        nodes=nodes,
        elements=elements,
        element_references=element_references,
        boundaries=boundaries,
        dependencies=dependencies,
        load_records=load_records,
        unknown_records=unknown_records,
        diagnostics=tuple(diagnostics),
    )
    if strict:
        from .validation import validate_sesam_fem_document

        all_diagnostics = tuple(diagnostics) + validate_sesam_fem_document(document)
        raise_if_errors(all_diagnostics, "SESAM FEM records failed validation")
        document = replace(document, diagnostics=all_diagnostics)
    return document


def _parse_header(records: tuple[FemRawRecord, ...]) -> FemHeader:
    ident = next((record for record in records if record.name == "IDENT"), None)
    date = next((record for record in records if record.name == "DATE"), None)
    units = next((record for record in records if record.name == "UNITS"), None)
    return FemHeader(
        ident_values=ident.numeric_fields if ident else (),
        ident_text=ident.text_fields if ident else (),
        date_text=date.text_fields if date else (),
        unit_values=units.numeric_fields if units else (),
    )


def _parse_materials(
    records: tuple[FemRawRecord, ...],
    diagnostics: list[FemDiagnostic],
) -> tuple[dict[int, FemMaterial], dict[int, str]]:
    materials: dict[int, FemMaterial] = {}
    names: dict[int, str] = {}
    for record in records:
        if record.name == "MISOSEL":
            if not record.numeric_fields:
                diagnostics.append(_diag("FEM101", "MISOSEL record has no material id", record))
                continue
            material_id = _int_field(record.numeric_fields[0], "material id", record, diagnostics)
            if material_id is None:
                continue
            if material_id in materials:
                diagnostics.append(_diag("FEM102", f"duplicate material id {material_id}", record))
                continue
            materials[material_id] = FemMaterial(
                material_id=material_id,
                elastic_modulus=_first_in_range(record.numeric_fields[1:], 1.0e7, 1.0e13),
                poisson_ratio=_first_in_range(record.numeric_fields[1:], 0.0, 0.5),
                density=_first_in_range(record.numeric_fields[1:], 10.0, 50000.0),
                yield_stress=_first_in_range(record.numeric_fields[1:], 1.0e6, 5.0e9),
                raw_values=record.numeric_fields,
            )
        elif record.name == "TDMATER" and record.numeric_fields:
            material_id = _int_field(record.numeric_fields[0], "material id", record, diagnostics)
            if material_id is not None and record.text_fields:
                names[material_id] = " ".join(record.text_fields)
    return materials, names


def _parse_sections(
    records: tuple[FemRawRecord, ...],
    diagnostics: list[FemDiagnostic],
) -> tuple[dict[int, FemSection], dict[int, str]]:
    sections: dict[int, FemSection] = {}
    names: dict[int, str] = {}
    for record in records:
        if record.name in {"GELTH", "GBEAMG", "GIORH", "GBARM"}:
            if not record.numeric_fields:
                diagnostics.append(_diag("FEM101", f"{record.name} record has no section id", record))
                continue
            section_id = _int_field(record.numeric_fields[0], "section id", record, diagnostics)
            if section_id is None:
                continue
            if section_id in sections:
                diagnostics.append(_diag("FEM102", f"duplicate section id {section_id}", record))
                continue
            values = record.numeric_fields
            if record.name == "GELTH":
                sections[section_id] = FemSection(
                    section_id=section_id,
                    kind="shell_thickness",
                    thickness=values[1] if len(values) > 1 and values[1] > 0.0 else None,
                    raw_values=values,
                )
            else:
                sections[section_id] = FemSection(
                    section_id=section_id,
                    kind="beam_section",
                    area=values[1] if len(values) > 1 and values[1] > 0.0 else None,
                    iy=values[2] if len(values) > 2 and values[2] > 0.0 else None,
                    iz=values[3] if len(values) > 3 and values[3] > 0.0 else None,
                    torsion=values[4] if len(values) > 4 and values[4] > 0.0 else None,
                    raw_values=values,
                )
        elif record.name == "TDSECT" and record.numeric_fields:
            section_id = _int_field(record.numeric_fields[0], "section id", record, diagnostics)
            if section_id is not None and record.text_fields:
                names[section_id] = " ".join(record.text_fields)
    return sections, names


def _parse_coordinates(
    records: tuple[FemRawRecord, ...],
    diagnostics: list[FemDiagnostic],
) -> dict[int, FemCoordinate]:
    coordinates: dict[int, FemCoordinate] = {}
    for record in records:
        if record.name != "GCOORD":
            continue
        if not record.numeric_fields:
            diagnostics.append(_diag("FEM101", "GCOORD record has no coordinate id", record))
            continue
        coordinate_id = _int_field(record.numeric_fields[0], "coordinate id", record, diagnostics)
        if coordinate_id is None:
            continue
        if coordinate_id in coordinates:
            diagnostics.append(_diag("FEM102", f"duplicate coordinate id {coordinate_id}", record))
            continue
        coordinates[coordinate_id] = FemCoordinate(coordinate_id, record.numeric_fields)
    return coordinates


def _parse_nodes(records: tuple[FemRawRecord, ...], diagnostics: list[FemDiagnostic]) -> dict[int, FemNode]:
    nodes: dict[int, FemNode] = {}
    for record in records:
        if record.name != "GNODE":
            continue
        values = record.numeric_fields
        if len(values) < 4:
            diagnostics.append(_diag("FEM101", "GNODE record must contain id and coordinates", record))
            continue
        node_id = _int_field(values[0], "node id", record, diagnostics)
        if node_id is None:
            continue
        if node_id in nodes:
            diagnostics.append(_diag("FEM102", f"duplicate node id {node_id}", record))
            continue
        if len(values) >= 5:
            coordinate_id = _int_field(values[1], "node coordinate system id", record, diagnostics)
            coords = values[2:5]
        else:
            coordinate_id = 0
            coords = values[1:4]
        if coordinate_id is None:
            continue
        nodes[node_id] = FemNode(
            node_id=node_id,
            coordinate_system_id=coordinate_id,
            coordinates=(float(coords[0]), float(coords[1]), float(coords[2])),
            raw_values=values,
        )
    return nodes


def _parse_element_references(
    records: tuple[FemRawRecord, ...],
    diagnostics: list[FemDiagnostic],
) -> dict[int, FemElementReference]:
    references: dict[int, FemElementReference] = {}
    for record in records:
        if record.name != "GELREF1":
            continue
        values = record.numeric_fields
        if not values:
            diagnostics.append(_diag("FEM101", "GELREF1 record has no element id", record))
            continue
        element_id = _int_field(values[0], "element id", record, diagnostics)
        if element_id is None:
            continue
        material_id = _optional_int(values[1], "material id", record, diagnostics) if len(values) > 1 else None
        section_id = None
        for index in (8, 2):
            if len(values) > index and values[index] > 0.0:
                section_id = _optional_int(values[index], "section id", record, diagnostics)
                break
        references[element_id] = FemElementReference(element_id, material_id, section_id, values)
    return references


def _parse_elements(
    records: tuple[FemRawRecord, ...],
    references: Mapping[int, FemElementReference],
    diagnostics: list[FemDiagnostic],
) -> dict[int, FemElement]:
    elements: dict[int, FemElement] = {}
    for record in records:
        if record.name != "GELMNT1":
            continue
        values = record.numeric_fields
        if len(values) < 3:
            diagnostics.append(_diag("FEM101", "GELMNT1 record is too short", record))
            continue
        element_id = _int_field(values[0], "element id", record, diagnostics)
        if element_id is None:
            continue
        if element_id in elements:
            diagnostics.append(_diag("FEM102", f"duplicate element id {element_id}", record))
            continue
        layout = _element_layout(record, diagnostics)
        if layout is None:
            continue
        type_code, node_start = layout
        spec = get_element_spec(type_code)
        if spec is None:
            diagnostics.append(_diag("FEM103", f"unsupported SESAM element type {type_code}", record))
            continue
        if len(values) < node_start + spec.node_count:
            diagnostics.append(
                _diag(
                    "FEM104",
                    f"element {element_id} type {type_code} expects {spec.node_count} nodes",
                    record,
                )
            )
            continue
        node_ids: list[int] = []
        for value in values[node_start:node_start + spec.node_count]:
            node_id = _int_field(value, "element node id", record, diagnostics)
            if node_id is not None:
                node_ids.append(node_id)
        if len(node_ids) != spec.node_count:
            continue
        reference = references.get(element_id)
        elements[element_id] = FemElement(
            element_id=element_id,
            type_code=type_code,
            topology=spec.topology,
            node_ids=tuple(node_ids),
            material_id=reference.material_id if reference else None,
            section_id=reference.section_id if reference else None,
            raw_values=values,
        )
    return elements


def _parse_boundaries(
    records: tuple[FemRawRecord, ...],
    diagnostics: list[FemDiagnostic],
) -> tuple[FemBoundary, ...]:
    boundaries: list[FemBoundary] = []
    for record in records:
        if record.name != "BNBCD":
            continue
        values = record.numeric_fields
        if len(values) < 2:
            diagnostics.append(_diag("FEM101", "BNBCD record is too short", record))
            continue
        node_id = _int_field(values[0], "boundary node id", record, diagnostics)
        if node_id is None:
            continue
        flags = []
        for value in values[1:7]:
            flag = _optional_int(value, "boundary dof flag", record, diagnostics)
            if flag is not None:
                flags.append(flag)
        boundaries.append(
            FemBoundary(
                node_id=node_id,
                dof_flags=tuple(flags),
                prescribed_values=tuple(values[7:13]),
                raw_values=values,
            )
        )
    return tuple(boundaries)


def _parse_concepts(records: tuple[FemRawRecord, ...], diagnostics: list[FemDiagnostic]) -> tuple[FemConceptRecord, ...]:
    concepts: list[FemConceptRecord] = []
    for record in records:
        if record.name not in {"TDSCONC", "SCONCEPT", "SCONMESH"}:
            continue
        concept_id = None
        if record.numeric_fields:
            concept_id = _optional_int(record.numeric_fields[0], "concept id", record, diagnostics)
        concepts.append(FemConceptRecord(record.name, concept_id, record.text_fields, record.numeric_fields))
    return tuple(concepts)


def _parse_dependencies(records: tuple[FemRawRecord, ...]) -> tuple[FemDependency, ...]:
    return tuple(
        FemDependency(record.name, record.numeric_fields, record.text_fields)
        for record in records
        if record.name == "BLDEP"
    )


def _parse_loads(records: tuple[FemRawRecord, ...], diagnostics: list[FemDiagnostic]) -> tuple[FemLoadRecord, ...]:
    loads: list[FemLoadRecord] = []
    for record in records:
        if record.name not in {"TDLOAD", "BEUSLO", "BNLOAD", "BNACCLO", "BGRAV"}:
            continue
        load_case_id = None
        target_id = None
        if record.numeric_fields:
            load_case_id = _optional_int(record.numeric_fields[0], "load case id", record, diagnostics)
        if len(record.numeric_fields) > 1:
            target_id = _optional_int(record.numeric_fields[1], "load target id", record, diagnostics)
        loads.append(FemLoadRecord(record.name, load_case_id, target_id, record.numeric_fields, record.text_fields))
    return tuple(loads)


def _element_layout(
    record: FemRawRecord,
    diagnostics: list[FemDiagnostic],
) -> Optional[tuple[int, int]]:
    values = record.numeric_fields
    candidate_indices = (2, 1)
    for index in candidate_indices:
        if len(values) <= index:
            continue
        try:
            type_code = strict_int(values[index], field_name="element type", record=record)
        except SesamFemError as exc:
            diagnostics.extend(exc.diagnostics)
            continue
        if get_element_spec(type_code) is not None:
            return type_code, 4 if index == 2 else 2
    diagnostics.append(_diag("FEM103", "GELMNT1 record does not contain a supported element type", record))
    return None


def _first_in_range(values: tuple[float, ...], lower: float, upper: float) -> Optional[float]:
    for value in values:
        if lower < abs(float(value)) < upper:
            return float(value)
    return None


def _int_field(
    value: float,
    field_name: str,
    record: FemRawRecord,
    diagnostics: list[FemDiagnostic],
) -> Optional[int]:
    try:
        return strict_int(value, field_name=field_name, record=record)
    except SesamFemError as exc:
        diagnostics.extend(exc.diagnostics)
        return None


def _optional_int(
    value: float,
    field_name: str,
    record: FemRawRecord,
    diagnostics: list[FemDiagnostic],
) -> Optional[int]:
    if abs(float(value)) < 1.0e-12:
        return 0
    return _int_field(value, field_name, record, diagnostics)


def _diag(code: str, message: str, record: FemRawRecord) -> FemDiagnostic:
    return FemDiagnostic(
        code,
        message,
        record_name=record.name,
        line_start=record.source_line_start,
        line_end=record.source_line_end,
    )
