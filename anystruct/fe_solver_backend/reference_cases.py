"""Automatic discovery of CalculiX/PrePoMax reference cases.

Reference cases are discovered as input/result pairs, normally placed under one
of these directories relative to the repository root:

- tests/reference_cases/
- reference_cases/
- examples/reference_cases/

A case is any ``*.inp`` file with a matching ``*.frd`` file using the same stem
in the same directory.  Optional metadata may be supplied as ``<stem>.json``.
The discovery is intentionally file-system based so local, non-committed
benchmark cases can be used during development without changing test code.

This module also contains a small manifest for public upstream CalculiX examples
that are useful reference candidates but are not vendored into this repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_REFERENCE_ROOTS = (
    "tests/reference_cases",
    "reference_cases",
    "examples/reference_cases",
)

SHELL_CONVERGENCE_ELEMENT_TYPES = ("S3", "S4", "S4R", "S6", "S8", "S8R")


UPSTREAM_CALCULIX_REFERENCE_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "calculix_examples_shell_convergence",
        "kind": "shell_bending_convergence",
        "repository": "calculix/CalculiX-Examples",
        "ref": "master",
        "directory": "Elements/Shell",
        "readme_url": "https://github.com/calculix/CalculiX-Examples/blob/master/Elements/Shell/README.md",
        "input_url": "https://github.com/calculix/CalculiX-Examples/blob/master/Elements/Shell/shell.inp",
        "raw_base_url": "https://raw.githubusercontent.com/calculix/CalculiX-Examples/master/Elements/Shell",
        "source_files": ["README.md", "shell.inp", "shell.fbd", "test.py"],
        "description": (
            "Mesh convergence benchmark for CalculiX shell elements S3/S4/S4R/S6/S8/S8R. "
            "The upstream input includes generated mesh and boundary include files, steel material, "
            "shell thickness 5 and gravity loading, and requests element stresses and nodal displacements."
        ),
        "requires_generated_includes": True,
        "expected_outputs": ["U", "S"],
        "reference_values": {"sref": 1.848, "wref": 0.0587},
        "notes": [
            "This is a useful shell bending/convergence reference candidate, but it is not a directly vendored .inp/.frd pair.",
            "Use the upstream test.py/shell.fbd workflow to generate concrete .inp/.frd files before numerical comparison.",
        ],
    },
)


@dataclass(frozen=True)
class CalculixReferenceCase:
    """Discovered CalculiX/PrePoMax reference input/result pair."""

    name: str
    directory: Path
    inp_path: Path
    frd_path: Optional[Path]
    metadata_path: Optional[Path] = None
    kind: str = "unknown"
    node_count: int = 0
    element_count: int = 0
    bbox_min: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    bbox_max: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))

    @property
    def has_results(self) -> bool:
        return self.frd_path is not None and self.frd_path.exists()

    def metadata(self) -> Dict[str, Any]:
        if self.metadata_path is None or not self.metadata_path.exists():
            return {}
        if self.metadata_path.suffix.lower() != ".json":
            return {}
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "directory": str(self.directory),
            "inp_path": str(self.inp_path),
            "frd_path": str(self.frd_path) if self.frd_path is not None else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path is not None else None,
            "kind": self.kind,
            "node_count": self.node_count,
            "element_count": self.element_count,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "has_results": self.has_results,
        }


@dataclass(frozen=True)
class ShellConvergencePoint:
    """One row from an upstream CalculiX shell convergence result file."""

    element_type: str
    size: float
    node_count: int
    stress_max: float
    displacement_max: float
    stress_normalized: float
    displacement_normalized: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "element_type": self.element_type,
            "size": self.size,
            "node_count": self.node_count,
            "stress_max": self.stress_max,
            "displacement_max": self.displacement_max,
            "stress_normalized": self.stress_normalized,
            "displacement_normalized": self.displacement_normalized,
        }


@dataclass(frozen=True)
class ShellConvergenceTable:
    """Parsed upstream shell convergence result table for one element type."""

    element_type: str
    path: Path
    stress_reference: float
    displacement_reference: float
    points: Tuple[ShellConvergencePoint, ...]

    @property
    def finest_point(self) -> Optional[ShellConvergencePoint]:
        if not self.points:
            return None
        return min(self.points, key=lambda point: point.size)

    @property
    def largest_model_point(self) -> Optional[ShellConvergencePoint]:
        if not self.points:
            return None
        return max(self.points, key=lambda point: point.node_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "path": str(self.path),
            "stress_reference": self.stress_reference,
            "displacement_reference": self.displacement_reference,
            "points": [point.to_dict() for point in self.points],
        }


def upstream_calculix_reference_manifest() -> List[Dict[str, Any]]:
    """Return known upstream CalculiX reference candidates."""
    return [dict(case) for case in UPSTREAM_CALCULIX_REFERENCE_CASES]


def upstream_calculix_shell_reference_values() -> Dict[str, float]:
    """Return analytical/reference normalizers for the upstream shell benchmark."""
    manifest = upstream_calculix_reference_manifest()
    case = next(entry for entry in manifest if entry["name"] == "calculix_examples_shell_convergence")
    return dict(case.get("reference_values", {}))


def _iter_existing_roots(roots: Optional[Sequence[Path | str]], repo_root: Optional[Path | str]) -> List[Path]:
    base = Path.cwd() if repo_root is None else Path(repo_root)
    candidate_roots: Sequence[Path | str] = roots or DEFAULT_REFERENCE_ROOTS
    existing: List[Path] = []
    for root in candidate_roots:
        path = Path(root)
        if not path.is_absolute():
            path = base / path
        if path.exists() and path.is_dir():
            existing.append(path)
    return existing


def _case_insensitive_sidecar(inp_path: Path, suffix: str) -> Optional[Path]:
    suffix = suffix.lower()
    for candidate in inp_path.parent.iterdir():
        if candidate.is_file() and candidate.stem.lower() == inp_path.stem.lower() and candidate.suffix.lower() == suffix:
            return candidate
    return None


def _parse_inp_nodes_and_element_count(inp_path: Path, max_lines: int = 200_000) -> Tuple[np.ndarray, int]:
    """Parse node coordinates and count elements from a CalculiX/Abaqus input file.

    This parser is deliberately conservative and only reads enough to classify
    and summarize a reference case.  It is not a complete input deck parser.
    """
    nodes: List[Tuple[float, float, float]] = []
    element_count = 0
    section: Optional[str] = None

    try:
        with inp_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if line_number > max_lines:
                    break
                line = raw_line.strip()
                if not line or line.startswith("**"):
                    continue
                if line.startswith("*"):
                    keyword = line.split(",", 1)[0].strip().lower()
                    if keyword == "*node":
                        section = "node"
                    elif keyword == "*element":
                        section = "element"
                    else:
                        section = None
                    continue

                if section == "node":
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 4:
                        continue
                    try:
                        nodes.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        continue
                elif section == "element":
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 2:
                        element_count += 1
    except OSError:
        return np.zeros((0, 3), dtype=float), 0

    if not nodes:
        return np.zeros((0, 3), dtype=float), element_count
    return np.asarray(nodes, dtype=float), element_count


def classify_reference_case_from_nodes(nodes: np.ndarray) -> str:
    """Classify a reference case from node coordinates as flat_plate, cylinder or unknown."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        return "unknown"

    span = np.ptp(nodes, axis=0)
    max_span = max(float(np.max(span)), 1.0)
    if float(np.min(span)) < 1.0e-8 * max_span:
        return "flat_plate"

    for columns in ((0, 1), (0, 2), (1, 2)):
        radius = np.linalg.norm(nodes[:, columns] - np.mean(nodes[:, columns], axis=0), axis=1)
        radius_mean = float(np.mean(radius))
        radius_std = float(np.std(radius))
        if radius_mean > 0.0 and radius_std / radius_mean < 0.10:
            return "cylinder"

    return "unknown"


def summarize_inp_geometry(inp_path: Path) -> Dict[str, Any]:
    nodes, element_count = _parse_inp_nodes_and_element_count(inp_path)
    if nodes.size:
        bbox_min = tuple(float(v) for v in np.min(nodes, axis=0))
        bbox_max = tuple(float(v) for v in np.max(nodes, axis=0))
    else:
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (0.0, 0.0, 0.0)
    return {
        "kind": classify_reference_case_from_nodes(nodes),
        "node_count": int(nodes.shape[0]),
        "element_count": int(element_count),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def parse_calculix_shell_convergence_file(
    path: Path | str,
    element_type: Optional[str] = None,
    stress_reference: Optional[float] = None,
    displacement_reference: Optional[float] = None,
) -> ShellConvergenceTable:
    """Parse an upstream CalculiX shell convergence table such as ``S4.txt``.

    Expected rows follow the upstream format::

        # size NoN smax umax
        100 12 0.616489 0.013121

    Stress and displacement are normalized by the upstream analytical reference
    values unless explicit reference values are supplied.
    """
    file_path = Path(path)
    if element_type is None:
        element_type = file_path.stem.upper()
    references = upstream_calculix_shell_reference_values()
    sref = float(stress_reference if stress_reference is not None else references.get("sref", 1.0))
    wref = float(displacement_reference if displacement_reference is not None else references.get("wref", 1.0))
    if sref == 0.0 or wref == 0.0:
        raise ValueError("Shell convergence reference values must be non-zero")

    points: List[ShellConvergencePoint] = []
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"Could not read shell convergence file: {file_path}") from exc

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid shell convergence row in {file_path} line {line_number}: {line!r}")
        try:
            size = float(parts[0])
            node_count = int(float(parts[1]))
            stress_max = float(parts[2])
            displacement_max = float(parts[3])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric shell convergence row in {file_path} line {line_number}: {line!r}") from exc
        points.append(
            ShellConvergencePoint(
                element_type=str(element_type),
                size=size,
                node_count=node_count,
                stress_max=stress_max,
                displacement_max=displacement_max,
                stress_normalized=stress_max / sref,
                displacement_normalized=displacement_max / wref,
            )
        )

    return ShellConvergenceTable(
        element_type=str(element_type),
        path=file_path,
        stress_reference=sref,
        displacement_reference=wref,
        points=tuple(points),
    )


def discover_calculix_shell_convergence_tables(
    directory: Path | str,
    stress_reference: Optional[float] = None,
    displacement_reference: Optional[float] = None,
) -> List[ShellConvergenceTable]:
    """Discover and parse available upstream shell convergence ``*.txt`` files."""
    root = Path(directory)
    tables: List[ShellConvergenceTable] = []
    for element_type in SHELL_CONVERGENCE_ELEMENT_TYPES:
        path = root / f"{element_type}.txt"
        if path.exists():
            tables.append(
                parse_calculix_shell_convergence_file(
                    path,
                    element_type=element_type,
                    stress_reference=stress_reference,
                    displacement_reference=displacement_reference,
                )
            )
    return tables


def discover_calculix_reference_cases(
    roots: Optional[Sequence[Path | str]] = None,
    repo_root: Optional[Path | str] = None,
    require_frd: bool = True,
) -> List[CalculixReferenceCase]:
    """Discover CalculiX reference cases from input/result file pairs.

    Args:
        roots: Directories to search.  Relative paths are resolved against
            repo_root or the current working directory.
        repo_root: Optional base directory for relative search roots.
        require_frd: If true, only return cases with matching result files.
            If false, return input-only cases too.
    """
    cases: List[CalculixReferenceCase] = []
    seen: set[Path] = set()
    for root in _iter_existing_roots(roots, repo_root):
        for inp_path in root.rglob("*"):
            if not inp_path.is_file() or inp_path.suffix.lower() != ".inp":
                continue
            resolved = inp_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            frd_path = _case_insensitive_sidecar(inp_path, ".frd")
            if require_frd and frd_path is None:
                continue
            metadata_path = _case_insensitive_sidecar(inp_path, ".json")
            summary = summarize_inp_geometry(inp_path)
            metadata_name = None
            if metadata_path is not None:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata_name = metadata.get("name")
                except (OSError, json.JSONDecodeError):
                    metadata_name = None
            name = str(metadata_name or inp_path.stem)
            cases.append(
                CalculixReferenceCase(
                    name=name,
                    directory=inp_path.parent,
                    inp_path=inp_path,
                    frd_path=frd_path,
                    metadata_path=metadata_path,
                    kind=str(summary["kind"]),
                    node_count=int(summary["node_count"]),
                    element_count=int(summary["element_count"]),
                    bbox_min=summary["bbox_min"],
                    bbox_max=summary["bbox_max"],
                )
            )
    return sorted(cases, key=lambda case: (str(case.directory), case.name))
