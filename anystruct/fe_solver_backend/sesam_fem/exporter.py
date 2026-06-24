"""SESAM formatted FEM document writer and guarded semantic exporter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .diagnostics import FemDiagnostic, SesamFemError
from .document import SesamFemDocument
from .records import records_to_text


@dataclass(frozen=True)
class SesamFemExportReport:
    path: Path
    records_written: int
    bytes_written: int
    mode: str
    diagnostics: tuple[FemDiagnostic, ...] = ()


def write_sesam_fem_document(
    document: SesamFemDocument,
    path: str | Path,
    *,
    mode: str = "canonical",
    overwrite: bool = False,
    newline: str = "\r\n",
) -> SesamFemExportReport:
    """Write a typed SESAM FEM document back to formatted sequential FEM text."""

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise SesamFemError(f"refusing to overwrite existing file: {output_path}", code="FEM201")
    if mode not in {"canonical", "raw"}:
        raise SesamFemError(f"unsupported FEM write mode: {mode}", code="FEM202")

    if mode == "raw":
        lines: list[str] = []
        saw_iend = False
        for record in document.raw_records:
            if record.name == "IEND":
                saw_iend = True
            lines.extend(record.raw_lines)
        if not saw_iend:
            lines.append("IEND")
        text = newline.join(lines) + newline
    else:
        text = records_to_text(document.raw_records, newline=newline)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(text, encoding="ascii", newline="")
    tmp_path.replace(output_path)
    return SesamFemExportReport(
        path=output_path,
        records_written=sum(1 for record in document.raw_records if record.name != "") + (
            0 if any(record.name == "IEND" for record in document.raw_records) else 1
        ),
        bytes_written=len(text.encode("ascii")),
        mode=mode,
    )


def export_sesam_fem(model_or_document: object, path: str | Path, **kwargs: object) -> SesamFemExportReport:
    """Export a SESAM FEM document.

    Full semantic export from arbitrary FEModel objects is a later gate.  Passing
    a SesamFemDocument is supported now for validation, canonicalization and
    fixture round-trips.
    """

    if isinstance(model_or_document, SesamFemDocument):
        return write_sesam_fem_document(model_or_document, path, **kwargs)
    raise SesamFemError(
        "semantic export from FEModel to SESAM FEM is not implemented in this gate",
        code="FEM203",
        diagnostics=(
            FemDiagnostic(
                "FEM203",
                "pass a SesamFemDocument for round-trip export; FEModel export is deferred",
            ),
        ),
    )
