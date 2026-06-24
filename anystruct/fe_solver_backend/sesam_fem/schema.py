"""SESAM formatted FEM schema registry used by the importer/exporter.

The registry is intentionally small and explicit.  It covers the record and
element families that ANYintelligent can preserve today, while keeping unknown
records available for later, higher-fidelity gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional


@dataclass(frozen=True)
class SesamElementSpec:
    """Supported SESAM element topology description."""

    type_code: int
    name: str
    family: str
    topology: str
    node_count: int
    export_code: int
    pressure_node_count: int = 0

    @property
    def is_shell(self) -> bool:
        return self.family == "shell"

    @property
    def is_beam(self) -> bool:
        return self.family == "beam"


SESAM_ELEMENT_REGISTRY: Dict[int, SesamElementSpec] = {
    15: SesamElementSpec(15, "BEAM2", "beam", "line2", 2, 15),
    23: SesamElementSpec(23, "BEAM3", "beam", "line3", 3, 23),
    24: SesamElementSpec(24, "Q4", "shell", "quad4", 4, 24, pressure_node_count=4),
    25: SesamElementSpec(25, "T3", "shell", "tri3", 3, 25, pressure_node_count=3),
    26: SesamElementSpec(26, "T6", "shell", "tri6", 6, 26, pressure_node_count=6),
    28: SesamElementSpec(28, "Q8", "shell", "quad8", 8, 28, pressure_node_count=8),
}


METADATA_RECORDS: FrozenSet[str] = frozenset({"IDENT", "DATE", "UNITS", "IEND"})

MATERIAL_SECTION_RECORDS: FrozenSet[str] = frozenset(
    {"TDMATER", "MISOSEL", "TDSECT", "GELTH", "GBEAMG", "GIORH", "GBARM"}
)

CONCEPT_RECORDS: FrozenSet[str] = frozenset({"TDSCONC", "SCONCEPT", "SCONMESH"})

GEOMETRY_RECORDS: FrozenSet[str] = frozenset(
    {"GCOORD", "GNODE", "GELMNT1", "GELREF1", "GUNIVEC", "GECCEN"}
)

TRANSFORM_RECORDS: FrozenSet[str] = frozenset({"BNTRCOS"})

BOUNDARY_LOAD_RECORDS: FrozenSet[str] = frozenset(
    {"BNBCD", "BLDEP", "TDLOAD", "BEUSLO", "BNLOAD", "BNACCLO", "BGRAV"}
)

SUPPORTED_RECORDS: FrozenSet[str] = (
    METADATA_RECORDS
    | MATERIAL_SECTION_RECORDS
    | CONCEPT_RECORDS
    | GEOMETRY_RECORDS
    | TRANSFORM_RECORDS
    | BOUNDARY_LOAD_RECORDS
)

TEXT_RECORDS: FrozenSet[str] = frozenset({"DATE", "TDMATER", "TDSECT", "TDSCONC", "TDLOAD"})

KNOWN_UNSUPPORTED_STRUCTURAL_RECORDS: FrozenSet[str] = frozenset(
    {
        # These are intentionally not translated into solver entities yet.
        "BLDEP",
        "BEUSLO",
        "BNACCLO",
        "BGRAV",
    }
)

DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


def get_element_spec(type_code: int) -> Optional[SesamElementSpec]:
    """Return the supported element specification for a SESAM element code."""

    return SESAM_ELEMENT_REGISTRY.get(int(type_code))


def classify_record(record_name: str) -> str:
    """Classify a SESAM FEM record name for diagnostics and reporting."""

    name = record_name.upper()
    if name in METADATA_RECORDS:
        return "metadata"
    if name in MATERIAL_SECTION_RECORDS:
        return "material_section"
    if name in CONCEPT_RECORDS:
        return "concept"
    if name in GEOMETRY_RECORDS:
        return "geometry"
    if name in BOUNDARY_LOAD_RECORDS:
        return "boundary_load"
    return "unknown"
