import os
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from pathlib import Path

from .importer import import_sesam_fem
from .records import read_raw_records
from .schema import get_element_spec


@dataclass(frozen=True)
class SesamStressResult:
    path: str
    nodes: dict[int, tuple[float, float, float]]
    element_nodes: dict[int, tuple[int, ...]]
    components: tuple[str, ...]
    nodal_stress: dict[int, tuple[float, ...]]
    element_stress: dict[int, tuple[float, ...]]
    units: str = "Pa"


def _mean_rvstress_membrane_components(
    values: Sequence[float],
    *,
    type_code: int | None = None,
) -> tuple[float, float, float] | None:
    """Return representative local membrane ``SIGXX, SIGYY, TAUXY`` values.

    SESAM RVSTRESS shell payloads in the reference cases use two layouts:
    lower/upper blocks of 3-component stress tuples for first-order shells and
    5-column result-point rows for second-order T6/Q8 shells.  Only the first three values are
    membrane stress components; remaining columns are result metadata/derived
    quantities and must not be averaged into the stress tensor.
    """

    payload = [float(value) for value in values if math.isfinite(float(value))]
    triplets: list[tuple[float, float, float]] = []
    if type_code in {26, 28} and len(payload) % 5 == 0:
        triplets = [
            (payload[index], payload[index + 1], payload[index + 2])
            for index in range(0, len(payload), 5)
        ]
    elif len(payload) % 6 == 0:
        triplets = [
            (payload[index], payload[index + 1], payload[index + 2])
            for index in range(0, len(payload), 3)
        ]
    elif len(payload) % 5 == 0:
        triplets = [
            (payload[index], payload[index + 1], payload[index + 2])
            for index in range(0, len(payload), 5)
        ]
    elif len(payload) % 3 == 0:
        triplets = [
            (payload[index], payload[index + 1], payload[index + 2])
            for index in range(0, len(payload), 3)
        ]
    if not triplets:
        return None
    return (
        sum(item[0] for item in triplets) / len(triplets),
        sum(item[1] for item in triplets) / len(triplets),
        sum(item[2] for item in triplets) / len(triplets),
    )


def _element_corner_node_ids(element, type_code: int) -> tuple[int, ...]:
    if type_code == 28 and len(element.node_ids) >= 8:
        return (element.node_ids[0], element.node_ids[2], element.node_ids[4], element.node_ids[6])
    if type_code == 26 and len(element.node_ids) >= 6:
        return (element.node_ids[0], element.node_ids[2], element.node_ids[4])
    spec = get_element_spec(type_code)
    if spec is None:
        return tuple(element.node_ids)
    corner_count = spec.pressure_node_count or spec.node_count
    return tuple(element.node_ids[:corner_count])


def _normalise_array(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        return None
    return vector / length


def _element_geometry_normal(points: Sequence[tuple[float, float, float]]) -> np.ndarray | None:
    p0 = np.asarray(points[0], dtype=float)
    for index in range(1, len(points) - 1):
        candidate = _normalise_array(
            np.cross(
                np.asarray(points[index], dtype=float) - p0,
                np.asarray(points[index + 1], dtype=float) - p0,
            )
        )
        if candidate is not None:
            return candidate
    return None


def _project_axis_to_plane(axis: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    return _normalise_array(axis - np.dot(axis, normal) * normal)


def _element_reference(document, element):
    reference = document.element_references.get(element.element_id)
    if reference is not None:
        return reference
    if len(element.raw_values) > 1:
        internal_id = int(round(element.raw_values[1]))
        return document.element_references.get(internal_id)
    return None


def _mean_transform_axes(document, transform_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_z: list[np.ndarray] = []
    for transform_id in transform_ids:
        transform = document.coordinate_transforms.get(transform_id)
        if transform is None:
            continue
        matrix = np.asarray(transform.matrix, dtype=float)
        if matrix.shape != (3, 3):
            continue
        rows_x.append(matrix[0])
        rows_y.append(matrix[1])
        rows_z.append(matrix[2])
    if not rows_x or not rows_y:
        return None
    x_axis = _normalise_array(np.sum(rows_x, axis=0))
    y_axis = _normalise_array(np.sum(rows_y, axis=0))
    z_axis = _normalise_array(np.sum(rows_z, axis=0)) if rows_z else None
    if x_axis is None or y_axis is None:
        return None
    return x_axis, y_axis, z_axis


def _mean_unit_vector_axis(document, transform_ids: Sequence[int]) -> np.ndarray | None:
    vectors: list[np.ndarray] = []
    for transform_id in transform_ids:
        unit_vector = document.unit_vectors.get(transform_id)
        if unit_vector is None:
            continue
        vectors.append(np.asarray(unit_vector.vector, dtype=float))
    if not vectors:
        return None
    return _normalise_array(np.sum(vectors, axis=0))


def _explicit_shell_local_frame(document, element, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    reference = _element_reference(document, element)
    if reference is None:
        return None
    transform_ids = reference.nodal_transform_ids or (() if reference.transform_id is None else (reference.transform_id,))
    if not transform_ids:
        return None
    axes = _mean_transform_axes(document, transform_ids)
    if axes is None:
        return None
    raw_x, raw_y, raw_z = axes
    if raw_z is not None and np.dot(raw_z, normal) < 0.0:
        normal = -normal

    x_axis = _project_axis_to_plane(raw_x, normal)
    y_axis = _project_axis_to_plane(raw_y, normal)
    if x_axis is None and y_axis is None:
        return None
    if x_axis is None:
        x_axis = _normalise_array(np.cross(y_axis, normal))
    if x_axis is None:
        return None
    if y_axis is None or abs(float(np.dot(x_axis, y_axis))) > 1.0e-6:
        y_axis = _normalise_array(raw_y - np.dot(raw_y, normal) * normal - np.dot(raw_y, x_axis) * x_axis)
    if y_axis is None:
        y_axis = _normalise_array(np.cross(normal, x_axis))
    if y_axis is None:
        return None
    if np.dot(np.cross(x_axis, y_axis), normal) < 0.0:
        y_axis = -y_axis
    return x_axis, y_axis, normal


def _unit_vector_shell_local_frame(document, element, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    reference = _element_reference(document, element)
    if reference is None:
        return None
    transform_ids = reference.nodal_transform_ids or (() if reference.transform_id is None else (reference.transform_id,))
    if not transform_ids:
        return None
    raw_y = _mean_unit_vector_axis(document, transform_ids)
    if raw_y is None:
        return None
    y_axis = _project_axis_to_plane(raw_y, normal)
    if y_axis is None:
        return None
    x_axis = _normalise_array(np.cross(y_axis, normal))
    if x_axis is None:
        return None
    if np.dot(np.cross(x_axis, y_axis), normal) < 0.0:
        x_axis = -x_axis
    return x_axis, y_axis, normal


def _element_local_frame(document, element) -> tuple[tuple[float,...], tuple[float,...], tuple[float,...]] | None:
    spec = get_element_spec(element.type_code)
    if spec is None:
        return None
    corner_ids = _element_corner_node_ids(element, element.type_code)
    points = [document.nodes[node_id].coordinates for node_id in corner_ids if node_id in document.nodes]
    if len(points) < 3:
        return None
    normal = _element_geometry_normal(points)
    if normal is None:
        return None
    if spec.is_shell:
        explicit_frame = _explicit_shell_local_frame(document, element, normal)
        if explicit_frame is not None:
            x_axis, y_axis, normal = explicit_frame
            return tuple(x_axis.tolist()), tuple(y_axis.tolist()), tuple(normal.tolist())
        unit_vector_frame = _unit_vector_shell_local_frame(document, element, normal)
        if unit_vector_frame is not None:
            x_axis, y_axis, normal = unit_vector_frame
            return tuple(x_axis.tolist()), tuple(y_axis.tolist()), tuple(normal.tolist())

    p0 = np.asarray(points[0], dtype=float)
    p1 = np.asarray(points[1], dtype=float)
    # In SESAM shell result records without explicit transforms, the in-plane
    # local y direction follows the first element edge.  The local x direction
    # is completed from y x normal so local membrane components can be rotated
    # into the global tensor before panel-axis projection.
    y_axis = _normalise_array(p1 - p0)
    if y_axis is None:
        return None
    x_axis = _normalise_array(np.cross(y_axis, normal))
    if x_axis is None:
        return None
    return tuple(x_axis.tolist()), tuple(y_axis.tolist()), tuple(normal.tolist())


def _local_membrane_to_global_components(
    document,
    element,
    local_stress: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float] | None:
    frame = _element_local_frame(document, element)
    if frame is None:
        return None
    local_x, local_y, _normal = frame
    sx, sy, txy = local_stress
    tensor = np.zeros((3, 3), dtype=float)
    x = np.asarray(local_x, dtype=float)
    y = np.asarray(local_y, dtype=float)
    tensor += sx * np.outer(x, x)
    tensor += sy * np.outer(y, y)
    tensor += txy * (np.outer(x, y) + np.outer(y, x))
    return (
        float(tensor[0, 0]),
        float(tensor[1, 1]),
        float(tensor[2, 2]),
        float(tensor[0, 1]),
        float(tensor[1, 2]),
        float(tensor[2, 0]),
    )


def read_sesam_sif_stress(
    path: str | os.PathLike[str],
    load_case: int | None = None,
) -> SesamStressResult:
    """Read RVSTRESS shell results as FRD-like global stress tensors."""

    path_text = str(path)
    result = import_sesam_fem(path_text, build_model=False, strict=False)
    document = result.document

    nodal_sums: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=float))
    nodal_counts: defaultdict[int, int] = defaultdict(int)
    element_stress: dict[int, tuple[float, ...]] = {}

    raw_records = read_raw_records(Path(path_text), strict=False)
    internal_to_external: dict[int, int] = {}
    for record in raw_records:
        if record.name != "GELMNT1" or len(record.numeric_fields) < 2:
            continue
        external_id = int(round(record.numeric_fields[0]))
        internal_id = int(round(record.numeric_fields[1]))
        internal_to_external[internal_id] = external_id

    for record in raw_records:
        if record.name != "RVSTRESS" or not record.numeric_fields or len(record.numeric_fields) < 8:
            continue
        values = record.numeric_fields
        if load_case is not None:
            record_load_case = int(round(values[1]))
            if record_load_case != load_case:
                continue
        result_element_id = int(values[2]) if values[2].is_integer() else int(round(values[2]))
        element_id = internal_to_external.get(result_element_id, result_element_id)
        type_code = int(values[4]) if values[4].is_integer() else int(round(values[4]))
        
        element = document.elements.get(element_id)
        if element is None:
            continue
            
        spec = get_element_spec(type_code)
        if spec is None or not spec.is_shell:
            continue
            
        local_stress = _mean_rvstress_membrane_components(values[5:], type_code=type_code)
        if local_stress is None:
            continue
            
        global_components = _local_membrane_to_global_components(document, element, local_stress)
        if global_components is None:
            continue
            
        element_stress[element_id] = global_components
        vector = np.asarray(global_components, dtype=float)
        for node_id in element.node_ids:
            nodal_sums[node_id] += vector
            nodal_counts[node_id] += 1

    nodal_stress = {
        node_id: tuple((values / max(nodal_counts[node_id], 1)).tolist())
        for node_id, values in nodal_sums.items()
    }
    
    element_nodes = {}
    for element_id, element in document.elements.items():
        spec = get_element_spec(element.type_code)
        if spec is not None and spec.is_shell:
            element_nodes[element_id] = tuple(element.node_ids)
            
    nodes = {node_id: tuple(node.coordinates) for node_id, node in document.nodes.items()}

    return SesamStressResult(
        path=path_text,
        nodes=nodes,
        element_nodes=element_nodes,
        components=("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"),
        nodal_stress=nodal_stress,
        element_stress=element_stress,
    )


def read_sesam_sif_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return lightweight metadata for a SESAM SIF/FEM-style file."""

    path_text = str(path)
    counts: Counter[str] = Counter()
    load_cases = set()
    load_case_names = {}
    for record in read_raw_records(Path(path_text), strict=False):
        counts[record.name] += 1
        if record.name == "RVSTRESS" and len(record.numeric_fields) >= 8:
            load_cases.add(int(round(record.numeric_fields[1])))
        elif record.name in ("TDLOAD", "TDRESREF") and len(record.numeric_fields) >= 2 and record.text_fields:
            lc_id = int(round(record.numeric_fields[1]))
            load_case_names[lc_id] = record.text_fields[0]

    result = import_sesam_fem(path_text, build_model=False, strict=False)
    document = result.document

    shell_count = 0
    beam_count = 0
    for element in document.elements.values():
        spec = get_element_spec(element.type_code)
        if spec:
            if spec.is_shell:
                shell_count += 1
            elif spec.is_beam:
                beam_count += 1

    return {
        "path": path_text,
        "format": "sesam_sif",
        "nodes": len(document.nodes),
        "elements": len(document.elements),
        "shell_elements": shell_count,
        "beam_elements": beam_count,
        "records": dict(counts),
        "load_cases": sorted(list(load_cases)),
        "load_case_names": load_case_names,
    }
