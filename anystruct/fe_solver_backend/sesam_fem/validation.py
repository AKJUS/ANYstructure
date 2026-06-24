"""Layer C validation checks for SESAM FEM documents."""

from __future__ import annotations

from .diagnostics import FemDiagnostic
from .document import SesamFemDocument
from .schema import KNOWN_UNSUPPORTED_STRUCTURAL_RECORDS


def validate_sesam_fem_document(document: SesamFemDocument) -> tuple[FemDiagnostic, ...]:
    """Validate document consistency without mutating it."""

    diagnostics: list[FemDiagnostic] = []
    if not document.raw_records:
        diagnostics.append(FemDiagnostic("FEM003", "document contains no records"))
        return tuple(diagnostics)
    if document.raw_records[-1].name != "IEND":
        diagnostics.append(FemDiagnostic("FEM003", "formatted FEM file is missing final IEND record"))

    node_ids = set(document.nodes)
    for element in document.elements.values():
        missing_nodes = [node_id for node_id in element.node_ids if node_id not in node_ids]
        if missing_nodes:
            diagnostics.append(
                FemDiagnostic(
                    "FEM105",
                    f"element {element.element_id} references missing nodes {missing_nodes}",
                    context={"element_id": element.element_id, "missing_nodes": missing_nodes},
                )
            )
        if element.material_id not in (None, 0) and element.material_id not in document.materials:
            diagnostics.append(
                FemDiagnostic(
                    "FEM106",
                    f"element {element.element_id} references undefined material {element.material_id}",
                    severity="warning",
                    context={"element_id": element.element_id, "material_id": element.material_id},
                )
            )
        if element.section_id not in (None, 0) and element.section_id not in document.sections:
            diagnostics.append(
                FemDiagnostic(
                    "FEM107",
                    f"element {element.element_id} references undefined section {element.section_id}",
                    severity="warning",
                    context={"element_id": element.element_id, "section_id": element.section_id},
                )
            )

    for record in document.raw_records:
        if record.name in KNOWN_UNSUPPORTED_STRUCTURAL_RECORDS:
            diagnostics.append(
                FemDiagnostic(
                    "FEM120",
                    f"{record.name} is preserved but not translated into solver semantics yet",
                    severity="warning",
                    record_name=record.name,
                    line_start=record.source_line_start,
                    line_end=record.source_line_end,
                )
            )
    return tuple(diagnostics)
