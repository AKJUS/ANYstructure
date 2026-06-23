"""Diagnostics and exceptions for SESAM formatted FEM support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class FemDiagnostic:
    """A structured SESAM FEM import/export diagnostic."""

    code: str
    message: str
    severity: str = "error"
    record_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    context: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.record_name is not None:
            data["record_name"] = self.record_name
        if self.line_start is not None:
            data["line_start"] = self.line_start
        if self.line_end is not None:
            data["line_end"] = self.line_end
        if self.context:
            data["context"] = dict(self.context)
        return data


class SesamFemError(ValueError):
    """Raised when a SESAM FEM file cannot be safely read or written."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "FEM000",
        diagnostics: Iterable[FemDiagnostic] | None = None,
    ) -> None:
        self.code = code
        self.diagnostics: Tuple[FemDiagnostic, ...] = tuple(diagnostics or ())
        super().__init__(f"{code}: {message}")


def has_errors(diagnostics: Iterable[FemDiagnostic]) -> bool:
    """Return True when the diagnostic collection contains errors."""

    return any(item.severity.lower() == "error" for item in diagnostics)


def raise_if_errors(diagnostics: Iterable[FemDiagnostic], message: str) -> None:
    """Raise a SesamFemError when any diagnostic has error severity."""

    items = tuple(diagnostics)
    if has_errors(items):
        first = next(item for item in items if item.severity.lower() == "error")
        raise SesamFemError(message, code=first.code, diagnostics=items)
