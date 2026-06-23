"""Pure-Python SESAM formatted FEM import/export support."""

from .diagnostics import FemDiagnostic, SesamFemError
from .document import (
    FemBoundary,
    FemCoordinate,
    FemElement,
    FemElementReference,
    FemHeader,
    FemLoadRecord,
    FemMaterial,
    FemNode,
    FemSection,
    SesamFemDocument,
    parse_sesam_fem_records,
    read_sesam_fem_document,
)
from .exporter import SesamFemExportReport, export_sesam_fem, write_sesam_fem_document
from .importer import SesamFemImportResult, build_fe_model_from_sesam_document, import_sesam_fem
from .records import FemRawRecord, read_raw_records, strict_int
from .schema import SESAM_ELEMENT_REGISTRY, SesamElementSpec
from .validation import validate_sesam_fem_document

__all__ = [
    "FemBoundary",
    "FemCoordinate",
    "FemDiagnostic",
    "FemElement",
    "FemElementReference",
    "FemHeader",
    "FemLoadRecord",
    "FemMaterial",
    "FemNode",
    "FemRawRecord",
    "FemSection",
    "SESAM_ELEMENT_REGISTRY",
    "SesamElementSpec",
    "SesamFemDocument",
    "SesamFemError",
    "SesamFemExportReport",
    "SesamFemImportResult",
    "build_fe_model_from_sesam_document",
    "export_sesam_fem",
    "import_sesam_fem",
    "parse_sesam_fem_records",
    "read_raw_records",
    "read_sesam_fem_document",
    "strict_int",
    "validate_sesam_fem_document",
    "write_sesam_fem_document",
]
