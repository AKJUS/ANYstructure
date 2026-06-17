"""Analysis/result provenance containers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LoadCaseRef:
    """Stable reference to an input load case or combination."""

    name: str
    factor: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisCase:
    """Minimal traceable analysis-case descriptor."""

    name: str
    analysis_type: str
    load_cases: Sequence[LoadCaseRef] = field(default_factory=tuple)
    settings: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrestressCase:
    """Reference to a prestress source used by KG or stress-stiffened analyses."""

    name: str
    source_result_case: str
    sign_convention: str = "tension_positive_resultants_to_destabilizing_KG"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultCase:
    """Traceable result provenance for an analysis run."""

    name: str
    analysis_case: AnalysisCase
    matrix_signature: Optional[str] = None
    load_signature: Optional[str] = None
    solver_backend: Optional[str] = None
    recovery: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "analysis_case": {
                "name": self.analysis_case.name,
                "analysis_type": self.analysis_case.analysis_type,
                "load_cases": [
                    {"name": item.name, "factor": item.factor, "metadata": dict(item.metadata)}
                    for item in self.analysis_case.load_cases
                ],
                "settings": dict(self.analysis_case.settings),
                "metadata": dict(self.analysis_case.metadata),
            },
            "matrix_signature": self.matrix_signature,
            "load_signature": self.load_signature,
            "solver_backend": self.solver_backend,
            "recovery": dict(self.recovery),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def load_case_ref(load_case: Any, factor: float = 1.0, metadata: Optional[Mapping[str, Any]] = None) -> LoadCaseRef:
    """Build a stable provenance reference from a load-case-like object."""
    name = "none" if load_case is None else str(getattr(load_case, "name", load_case))
    return LoadCaseRef(name=name, factor=float(factor), metadata=dict(metadata or {}))


def _json_signature(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def matrix_signature_from_info(assembly_info: Mapping[str, Any]) -> Optional[str]:
    """Hash matrix topology/revision data from assembly diagnostics."""
    if not assembly_info:
        return None
    payload: Dict[str, Any] = {}
    for key in ("stiffness", "mass", "geometric_stiffness", "damping"):
        info = assembly_info.get(key)
        if isinstance(info, Mapping):
            payload[key] = {
                "sparsity_signature": info.get("sparsity_signature"),
                "revision_signature": info.get("revision_signature"),
                "matrix_type": info.get("matrix_type"),
                "total_dofs": info.get("total_dofs"),
                "num_elements": info.get("num_elements"),
            }
    if not payload and isinstance(assembly_info.get("revision_signature"), Mapping):
        payload["revision_signature"] = assembly_info.get("revision_signature")
    return _json_signature(payload) if payload else None


def load_signature_from_info(load_info: Mapping[str, Any]) -> Optional[str]:
    """Hash load provenance data without storing the load vector itself."""
    if not load_info:
        return None
    payload = {
        "vector_type": load_info.get("vector_type"),
        "load_case": load_info.get("load_case"),
        "load_cases": load_info.get("load_cases"),
        "load_norm": load_info.get("load_norm"),
        "load_norms": load_info.get("load_norms"),
        "revision_signature": load_info.get("revision_signature"),
    }
    return _json_signature(payload)


def solver_backend_from_info(solver_info: Mapping[str, Any]) -> Optional[str]:
    """Extract the sparse backend name from standard solver diagnostics."""
    backend = solver_info.get("backend")
    if isinstance(backend, Mapping):
        return backend.get("backend")
    convergence = solver_info.get("convergence_info")
    if isinstance(convergence, Mapping):
        backend = convergence.get("backend")
        if isinstance(backend, Mapping):
            return backend.get("backend")
    return None


def make_result_case(
    *,
    name: str,
    analysis_type: str,
    load_cases: Sequence[Any] = (),
    assembly_info: Optional[Mapping[str, Any]] = None,
    solver_info: Optional[Mapping[str, Any]] = None,
    recovery: Optional[Mapping[str, Any]] = None,
    settings: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    warnings: Sequence[str] = (),
) -> ResultCase:
    """Create a ResultCase from existing solver metadata."""
    assembly_info = assembly_info or {}
    load_info = assembly_info.get("load_matrix", assembly_info.get("load", {}))
    analysis_case = AnalysisCase(
        name=name,
        analysis_type=analysis_type,
        load_cases=tuple(load_case_ref(item) for item in load_cases),
        settings=dict(settings or {}),
        metadata=dict(metadata or {}),
    )
    return ResultCase(
        name=name,
        analysis_case=analysis_case,
        matrix_signature=matrix_signature_from_info(assembly_info),
        load_signature=load_signature_from_info(load_info if isinstance(load_info, Mapping) else {}),
        solver_backend=solver_backend_from_info(solver_info or {}),
        recovery=dict(recovery or {}),
        warnings=tuple(str(item) for item in warnings),
        metadata={
            "matrix_revisions": {
                key: value.get("revision_signature")
                for key, value in assembly_info.items()
                if isinstance(value, Mapping) and value.get("revision_signature") is not None
            },
            **dict(metadata or {}),
        },
    )
