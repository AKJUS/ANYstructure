"""Standalone 3D solid surface export helpers.

The GUI builds simple preview meshes as vertices plus polygon faces.  This
module turns such meshes into welded triangular boundary meshes and writes them
through numpy-stl or meshio.  Imports are lazy so the main application can still
start when optional export dependencies are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class SolidExportError(RuntimeError):
    """Raised when a 3D solid export cannot be prepared or written."""


@dataclass(frozen=True)
class SolidTriangleMesh:
    """A welded triangular surface mesh."""

    points: np.ndarray
    triangles: np.ndarray
    name: str = "ANYstructure_solid"


def mesh_dict_to_triangular_mesh(mesh: dict, name: str | None = None, weld_decimals: int = 12) -> SolidTriangleMesh:
    """Convert an ANYstructure preview mesh dict to welded triangular faces.

    Expected input format:
      {'name': str, 'vertices': [(x, y, z), ...], 'faces': [[1, 2, 3, 4], ...]}

    Face indices are 1-based, matching the local GUI mesh format.
    """

    if not isinstance(mesh, dict):
        raise SolidExportError("Mesh must be a dictionary with vertices and faces.")

    vertices = np.asarray(mesh.get("vertices", []), dtype=float)
    faces = mesh.get("faces", [])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise SolidExportError("Mesh has no valid 3D vertices.")
    if not faces:
        raise SolidExportError("Mesh has no faces to export.")

    unique_points: list[tuple[float, float, float]] = []
    index_by_key: dict[tuple[float, float, float], int] = {}
    triangles: list[list[int]] = []

    def welded_index(vertex_index: int) -> int:
        try:
            vertex = vertices[int(vertex_index) - 1]
        except Exception as exc:
            raise SolidExportError(f"Face references invalid vertex index {vertex_index!r}.") from exc
        key = tuple(round(float(coord), weld_decimals) for coord in vertex)
        if key not in index_by_key:
            index_by_key[key] = len(unique_points)
            unique_points.append(tuple(float(coord) for coord in vertex))
        return index_by_key[key]

    for face in faces:
        indices = [welded_index(index) for index in face]
        if len(indices) < 3:
            continue
        for idx in range(1, len(indices) - 1):
            tri = [indices[0], indices[idx], indices[idx + 1]]
            if not _triangle_is_degenerate(unique_points, tri):
                triangles.append(tri)

    if not triangles:
        raise SolidExportError("Mesh has no non-degenerate triangles to export.")

    return SolidTriangleMesh(
        points=np.asarray(unique_points, dtype=float),
        triangles=np.asarray(triangles, dtype=np.int64),
        name=_safe_mesh_name(name or mesh.get("name", "ANYstructure_solid")),
    )


def write_numpy_stl(filename: str | Path, mesh: SolidTriangleMesh | dict, binary: bool = True) -> None:
    """Write a triangular surface mesh using numpy-stl."""

    tri_mesh = _ensure_triangular_mesh(mesh)
    try:
        from stl import Mode as stl_mode
        from stl import mesh as stl_mesh
    except Exception as exc:
        raise SolidExportError(
            "numpy-stl is required for STL solid export. Install package 'numpy-stl'."
        ) from exc

    data = np.zeros(len(tri_mesh.triangles), dtype=stl_mesh.Mesh.dtype)
    stl_model = stl_mesh.Mesh(data)
    for tri_idx, tri in enumerate(tri_mesh.triangles):
        stl_model.vectors[tri_idx] = tri_mesh.points[tri]
    stl_model.save(str(filename), mode=stl_mode.BINARY if binary else stl_mode.ASCII)


def write_meshio(filename: str | Path, mesh: SolidTriangleMesh | dict, file_format: str | None = None) -> None:
    """Write a triangular surface mesh using meshio.

    The file format is inferred from the extension unless ``file_format`` is
    provided.  Useful PrePoMax-adjacent choices include STL, VTK, VTU and PLY.
    """

    tri_mesh = _ensure_triangular_mesh(mesh)
    try:
        import meshio
    except Exception as exc:
        raise SolidExportError("meshio is required for this solid export.") from exc

    meshio_mesh = meshio.Mesh(points=tri_mesh.points, cells=[("triangle", tri_mesh.triangles)])
    meshio.write(str(filename), meshio_mesh, file_format=file_format)


def write_solid_export(
    filename: str | Path,
    mesh: SolidTriangleMesh | dict,
    backend: str = "meshio",
    file_format: str | None = None,
    binary_stl: bool = True,
) -> None:
    """Write a 3D solid boundary surface using the selected backend."""

    backend = str(backend).lower()
    if backend in {"numpy-stl", "numpy_stl", "stl"}:
        write_numpy_stl(filename, mesh, binary=binary_stl)
        return
    if backend == "meshio":
        write_meshio(filename, mesh, file_format=file_format)
        return
    raise SolidExportError(f"Unsupported solid export backend: {backend!r}.")


def connected_component_count(mesh: SolidTriangleMesh | dict) -> int:
    """Return the number of face-connected surface components."""

    tri_mesh = _ensure_triangular_mesh(mesh)
    vertex_to_triangles: dict[int, list[int]] = {}
    for tri_idx, tri in enumerate(tri_mesh.triangles):
        for vertex_index in tri:
            vertex_to_triangles.setdefault(int(vertex_index), []).append(tri_idx)

    seen: set[int] = set()
    components = 0
    for tri_idx in range(len(tri_mesh.triangles)):
        if tri_idx in seen:
            continue
        components += 1
        stack = [tri_idx]
        seen.add(tri_idx)
        while stack:
            current = stack.pop()
            for vertex_index in tri_mesh.triangles[current]:
                for neighbour in vertex_to_triangles.get(int(vertex_index), []):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
    return components


def _ensure_triangular_mesh(mesh: SolidTriangleMesh | dict) -> SolidTriangleMesh:
    if isinstance(mesh, SolidTriangleMesh):
        return mesh
    return mesh_dict_to_triangular_mesh(mesh)


def _triangle_is_degenerate(points: Sequence[Sequence[float]], tri: Sequence[int]) -> bool:
    if len(set(tri)) < 3:
        return True
    p0 = np.asarray(points[tri[0]], dtype=float)
    p1 = np.asarray(points[tri[1]], dtype=float)
    p2 = np.asarray(points[tri[2]], dtype=float)
    return float(np.linalg.norm(np.cross(p1 - p0, p2 - p0))) <= 1e-12


def _safe_mesh_name(value: object) -> str:
    name = str(value).strip().replace(" ", "_")
    return name or "ANYstructure_solid"
