"""Layer A SESAM formatted sequential FEM record reader/writer helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

from .diagnostics import FemDiagnostic, SesamFemError
from .schema import TEXT_RECORDS


_RECORD_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,7}$")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
    r"(?![A-Za-z0-9_])"
)
_NUMBER_FULL_RE = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?$")
_QUOTED_TEXT_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


@dataclass(frozen=True)
class FemRawRecord:
    """Raw SESAM FEM record with parsed numeric/text views and source lines."""

    name: str
    numeric_fields: tuple[float, ...]
    text_fields: tuple[str, ...]
    source_line_start: int
    source_line_end: int
    raw_lines: tuple[str, ...]


def strict_int(
    value: float,
    *,
    field_name: str = "value",
    record: FemRawRecord | None = None,
    tolerance: float = 1.0e-7,
) -> int:
    """Convert a SESAM numeric field to int, rejecting non-integral values."""

    if not math.isfinite(float(value)):
        raise SesamFemError(
            f"{field_name} must be finite",
            code="FEM002",
            diagnostics=(_record_diag("FEM002", f"{field_name} must be finite", record),),
        )
    rounded = int(round(float(value)))
    scale = max(1.0, abs(float(value)))
    if abs(float(value) - rounded) > tolerance * scale:
        raise SesamFemError(
            f"{field_name} must be an integer, got {value!r}",
            code="FEM002",
            diagnostics=(
                _record_diag("FEM002", f"{field_name} must be an integer, got {value!r}", record),
            ),
        )
    return rounded


def read_raw_records(path: str | Path, *, strict: bool = True) -> tuple[FemRawRecord, ...]:
    """Read raw records from a SESAM formatted sequential ``.FEM`` file."""

    file_path = Path(path)
    data = file_path.read_bytes()
    _guard_supported_file(file_path, data)
    text = data.decode("latin-1")
    if text.startswith("\ufeff"):
        text = text[1:]

    lines = text.splitlines()
    diagnostics: list[FemDiagnostic] = []
    records: list[FemRawRecord] = []
    current_name: str | None = None
    current_start = 0
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_name, current_start, current_lines
        if current_name is None:
            return
        records.append(_record_from_lines(current_name, current_start, tuple(current_lines)))
        current_name = None
        current_start = 0
        current_lines = []

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        name_field = line[:8].strip().upper()
        is_record_start = bool(_RECORD_NAME_RE.match(name_field))
        if is_record_start:
            flush()
            current_name = name_field
            current_start = line_no
            current_lines = [line]
            continue
        if current_name is None:
            diagnostics.append(
                FemDiagnostic(
                    "FEM002",
                    "line does not start a formatted FEM record",
                    line_start=line_no,
                    line_end=line_no,
                )
            )
            continue
        current_lines.append(line)

    flush()
    if not records:
        diagnostics.append(FemDiagnostic("FEM002", "no formatted FEM records were found"))
    if strict and diagnostics:
        raise SesamFemError("malformed formatted FEM file", code=diagnostics[0].code, diagnostics=diagnostics)
    return tuple(records)


def parse_record_lines(lines: Sequence[str], *, start_line: int = 1) -> FemRawRecord:
    """Parse an in-memory record. Useful for focused unit tests."""

    if not lines:
        raise SesamFemError("empty record", code="FEM002")
    name = lines[0][:8].strip().upper()
    if not _RECORD_NAME_RE.match(name):
        raise SesamFemError("record name is missing or malformed", code="FEM002")
    return _record_from_lines(name, start_line, tuple(lines))


def _record_from_lines(name: str, start_line: int, lines: tuple[str, ...]) -> FemRawRecord:
    payload_parts = []
    for raw in lines:
        payload_parts.append(raw[8:] if len(raw) > 8 else "")
    payload = "\n".join(payload_parts)
    numeric_fields = _parse_numeric_fields(tuple(payload_parts), payload)
    text_fields = _parse_text_fields(name, payload)
    return FemRawRecord(
        name=name,
        numeric_fields=numeric_fields,
        text_fields=text_fields,
        source_line_start=start_line,
        source_line_end=start_line + len(lines) - 1,
        raw_lines=lines,
    )


def _parse_numeric_fields(payload_parts: tuple[str, ...], payload: str) -> tuple[float, ...]:
    free_tokens = _NUMBER_RE.findall(payload)
    free_values = tuple(float(token.replace("D", "E").replace("d", "e")) for token in free_tokens)

    fixed_values: list[float] = []
    saw_fixed = False
    for part in payload_parts:
        if len(part.rstrip()) < 16:
            continue
        line_values: list[float] = []
        for index in range(0, len(part), 16):
            chunk = part[index:index + 16].strip()
            if not chunk:
                continue
            if not _NUMBER_FULL_RE.match(chunk):
                line_values = []
                break
            line_values.append(float(chunk.replace("D", "E").replace("d", "e")))
        if line_values:
            saw_fixed = True
            fixed_values.extend(line_values)
    if saw_fixed and len(fixed_values) > len(free_values):
        return tuple(fixed_values)
    return free_values


def _parse_text_fields(name: str, payload: str) -> tuple[str, ...]:
    quoted = [match.group(1) or match.group(2) or "" for match in _QUOTED_TEXT_RE.finditer(payload)]
    scrubbed = _QUOTED_TEXT_RE.sub(" ", payload)
    scrubbed = _NUMBER_RE.sub(" ", scrubbed)
    pieces = [piece.strip(" ,;\t") for piece in re.split(r"[\r\n]+| {2,}", scrubbed)]
    text = [piece for piece in pieces if piece and re.search(r"[A-Za-z_/.-]", piece)]
    if name.upper() in TEXT_RECORDS:
        return tuple(quoted + text)
    return tuple(quoted + [piece for piece in text if not piece.isupper()])


def _guard_supported_file(path: Path, data: bytes) -> None:
    suffix = path.suffix.lower()
    if suffix == ".sin":
        raise SesamFemError(
            "SESAM .SIN direct-access files are not supported by the pure-Python FEM reader",
            code="FEM001",
        )
    # SIF files are supported because they use the same FEM text formatting
    if not data:
        raise SesamFemError("formatted FEM file is empty", code="FEM003")
    sample = data[:4096]
    if b"\x00" in sample:
        raise SesamFemError("binary or unformatted FEM input is not supported", code="FEM001")
    allowed_controls = {9, 10, 12, 13}
    control_count = sum(1 for value in sample if value < 32 and value not in allowed_controls)
    if sample and control_count / len(sample) > 0.02:
        raise SesamFemError("binary or unformatted FEM input is not supported", code="FEM001")


def _record_diag(code: str, message: str, record: FemRawRecord | None) -> FemDiagnostic:
    return FemDiagnostic(
        code,
        message,
        record_name=record.name if record else None,
        line_start=record.source_line_start if record else None,
        line_end=record.source_line_end if record else None,
    )


def format_numeric(value: float) -> str:
    """Return deterministic SESAM-style numeric formatting."""

    if math.isfinite(float(value)) and abs(float(value) - round(float(value))) < 1.0e-9:
        return f"{int(round(float(value))):16d}"
    return f"{float(value):16.8E}"


def canonical_record_lines(record: FemRawRecord) -> tuple[str, ...]:
    """Format a raw record in a deterministic, readable FEM form."""

    numeric_tokens = [format_numeric(value) for value in record.numeric_fields]
    text_suffix = ""
    if record.text_fields:
        text_suffix = "  " + "  ".join(field.strip() for field in record.text_fields if field.strip())
    if not numeric_tokens:
        return (record.name.ljust(8) + text_suffix.rstrip(),)

    lines: list[str] = []
    first = True
    for index in range(0, len(numeric_tokens), 4):
        prefix = record.name.ljust(8) if first else " " * 8
        chunk = "".join(numeric_tokens[index:index + 4])
        if first:
            chunk += text_suffix
        lines.append((prefix + chunk).rstrip())
        first = False
    return tuple(lines)


def records_to_text(records: Iterable[FemRawRecord], *, newline: str = "\r\n") -> str:
    """Format records as canonical FEM text, ensuring a final ``IEND`` record."""

    lines: list[str] = []
    saw_iend = False
    for record in records:
        if record.name == "IEND":
            saw_iend = True
        lines.extend(canonical_record_lines(record))
    if not saw_iend:
        lines.append("IEND")
    return newline.join(lines) + newline
