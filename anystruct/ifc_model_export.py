# -*- coding: utf-8 -*-
"""Proper IFC shell/surface model export for ANYstructure.

This module intentionally does not export the Matplotlib preview mesh.  It rebuilds
plate, stiffener, girder, shell, longitudinal stiffener and ring stiffener objects
from the active ANYstructure line data as IFC surface/shell plates.

Important modelling convention:
    - plate, web and flange are exported as single mid-surface plates;
    - no solid thickness is generated;
    - web/flange/plate interfaces share the same geometric line where possible;
    - thickness values are kept only as ANYstructure properties/metadata.

Install dependency:
    pip install ifcopenshell

Typical use from Application:
    ifc_model_export.export_selected_structure_from_application(self, filename)
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import functools
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable


try:
    import ifcopenshell
    import ifcopenshell.guid
except ImportError as exc:  # pragma: no cover - executed only when dependency is missing
    raise ImportError(
        "IfcOpenShell is not installed. Install it with: pip install ifcopenshell"
    ) from exc


EPS = 1.0e-9


@dataclass
class ExportSummary:
    """Small result object returned to the GUI after export."""

    filename: str
    project_name: str
    elements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    native_ifc_filename: str | None = None
    converted_filename: str | None = None
    output_format: str = "ifc"

    @property
    def element_count(self) -> int:
        return len(self.elements)


@dataclass
class SectionDimensions:
    """Section dimensions in metres."""

    spacing: float = 0.75
    plate_thk: float = 0.02
    web_h: float = 0.4
    web_thk: float = 0.012
    flange_w: float = 0.15
    flange_thk: float = 0.02
    type: str = "T"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SectionDimensions":
        return cls(
            spacing=_pos_float(data.get("spacing"), 0.75),
            plate_thk=_pos_float(data.get("plate_thk"), 0.02),
            web_h=_pos_float(data.get("web_h"), 0.4),
            web_thk=_pos_float(data.get("web_thk"), 0.012),
            flange_w=_pos_float(data.get("flange_w"), 0.15),
            flange_thk=_pos_float(data.get("flange_thk"), 0.02),
            type=str(data.get("type", "T") or "T"),
        )


@dataclass
class IfcContext:
    model: Any
    project: Any
    site: Any
    building: Any
    storey: Any
    body_context: Any
    material: Any
    summary: ExportSummary
    boolean_join_all_solids: bool = False
    solid_operands: list[Any] = field(default_factory=list)
    solid_source_elements: list[Any] = field(default_factory=list)
    length_unit: str = "m"
    length_scale: float = 1.0
    transformation_scale: float = 1.0


IFCCONVERT_FORMATS = {
    "ifc": {"extension": ".ifc", "description": "Native IFC-SPF model", "tool": "IfcOpenShell-Python"},
    "obj": {"extension": ".obj", "description": "Wavefront OBJ", "tool": "IfcConvert"},
    "dae": {"extension": ".dae", "description": "Collada DAE", "tool": "IfcConvert"},
    "glb": {"extension": ".glb", "description": "Binary glTF GLB", "tool": "IfcConvert"},
    "stp": {"extension": ".stp", "description": "STEP", "tool": "IfcConvert"},
    "igs": {"extension": ".igs", "description": "IGES", "tool": "IfcConvert"},
    "xml": {"extension": ".xml", "description": "XML", "tool": "IfcConvert"},
    "svg": {"extension": ".svg", "description": "SVG", "tool": "IfcConvert"},
    "h5": {"extension": ".h5", "description": "HDF5", "tool": "IfcConvert"},
    "ttl": {"extension": ".ttl", "description": "TTL/WKT", "tool": "IfcConvert"},
    "rdb": {"extension": ".rdb", "description": "RDB", "tool": "IfcConvert"},
    "json": {"extension": ".json", "description": "JSON", "tool": "IfcConvert"},
}


def supported_export_formats() -> dict[str, dict[str, str]]:
    """Return formats exposed in the ANYstructure IFC export dialog.

    The native .ifc file is written directly with IfcOpenShell-Python.  The other
    listed outputs follow the IfcConvert options shown in the IfcOpenShell docs
    and require the bundled/package IfcConvert executable, with PATH as a fallback.
    """
    return dict(IFCCONVERT_FORMATS)


def _normalise_export_format(output_format: str | None, filename: str | None = None) -> str:
    fmt = (output_format or "").strip().lower().lstrip(".")
    if not fmt and filename:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        fmt = ext or "ifc"
    if fmt == "step":
        fmt = "stp"
    if fmt == "iges":
        fmt = "igs"
    if fmt not in IFCCONVERT_FORMATS:
        raise ValueError("Unsupported IFC export format: " + str(output_format))
    return fmt


def _existing_executable(path: str) -> str | None:
    """Return path if it exists and can be executed as a local file."""
    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if os.path.isfile(expanded):
        return expanded
    return None


@functools.lru_cache(maxsize=1)
def _resource_ifcconvert_candidates() -> tuple[str, ...]:
    """Return IfcConvert candidates shipped inside the installed anystruct package.

    This is the important path for PyPI wheels/sdists.  If ``IfcConvert.exe`` is
    included as package data, importlib.resources resolves it from the installed
    package location without using any developer-specific absolute path.
    """
    candidates: list[str] = []
    package_names = []
    module_package = (__package__ or '').split('.')[0]
    if module_package:
        package_names.append(module_package)
    package_names.extend(['anystruct', 'ANYstructure.anystruct'])

    for package_name in package_names:
        try:
            package_files = importlib_resources.files(package_name)
        except Exception:
            continue
        for name in ('IfcConvert.exe', 'IfcConvert', 'ifcconvert.exe', 'ifcconvert'):
            try:
                candidate = package_files.joinpath(name)
                # In a normal PyPI install this is a filesystem path.  If a future
                # zipped importer is used, as_file extracts it to a temporary path.
                with importlib_resources.as_file(candidate) as resolved:
                    candidates.append(str(resolved))
            except Exception:
                pass
    return tuple(candidates)


@functools.lru_cache(maxsize=8)
def _ifcconvert_candidate_paths(ifcconvert_path: str | None = None) -> list[str]:
    """Return automatic IfcConvert locations.

    No user-selected path is required.  Resolution order is:
      1. explicit/internal override, if supplied by code
      2. ``ANYSTRUCTURE_IFCCONVERT`` environment variable
      3. IfcConvert shipped as package data in the installed ``anystruct`` package
      4. beside this module
      5. common PyInstaller locations, including ``_internal`` and ``sys._MEIPASS``
      6. PATH

    There are intentionally no hard-coded development paths such as
    ``C:\\Github\\ANYstructure\\anystruct``.  That path is valid only in the
    developer checkout and is wrong for PyPI and PyInstaller users.
    """
    candidates: list[str] = []

    if ifcconvert_path:
        candidates.append(ifcconvert_path)

    env_path = os.environ.get('ANYSTRUCTURE_IFCCONVERT', '').strip()
    if env_path:
        candidates.append(env_path)

    candidates.extend(_resource_ifcconvert_candidates())

    module_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    exe_dir = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'executable', None) else ''
    bundle_dir = getattr(sys, '_MEIPASS', '')

    bases = [
        module_dir,
        os.path.join(module_dir, 'anystruct'),
        exe_dir,
        os.path.join(exe_dir, 'anystruct') if exe_dir else '',
        os.path.join(exe_dir, '_internal') if exe_dir else '',
        os.path.join(exe_dir, '_internal', 'anystruct') if exe_dir else '',
        bundle_dir,
        os.path.join(bundle_dir, 'anystruct') if bundle_dir else '',
        os.path.join(bundle_dir, '_internal') if bundle_dir else '',
        os.path.join(bundle_dir, '_internal', 'anystruct') if bundle_dir else '',
        cwd,
        os.path.join(cwd, 'anystruct'),
    ]

    for base in bases:
        if not base:
            continue
        for name in ('IfcConvert.exe', 'IfcConvert', 'ifcconvert.exe', 'ifcconvert'):
            candidates.append(os.path.join(base, name))

    candidates.extend(['IfcConvert', 'IfcConvert.exe', 'ifcconvert', 'ifcconvert.exe'])

    unique: list[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.dirname(candidate):
            key = os.path.normcase(os.path.abspath(os.path.expanduser(os.path.expandvars(candidate))))
        else:
            key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_ifcconvert_executable(ifcconvert_path: str | None = None) -> str:
    candidates = _ifcconvert_candidate_paths(ifcconvert_path)
    for candidate in candidates:
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        local = _existing_executable(expanded)
        if local:
            return local
        found = shutil.which(expanded)
        if found:
            return found

    searched = "\n".join("  - " + path for path in candidates[:20])
    raise FileNotFoundError(
        "IfcConvert was not found automatically. Users should not have to select it manually.\n\n"
        "For PyPI, include IfcConvert.exe as package data inside the anystruct package, e.g.\n"
        "  anystruct/IfcConvert.exe\n\n"
        "For PyInstaller, add it as a binary/data file beside the anystruct package or in _internal.\n\n"
        "Searched locations include:\n" + searched
    )


def _convert_ifc_with_ifcconvert(
    native_ifc_filename: str,
    output_filename: str,
    ifcconvert_path: str | None = None,
    timeout_seconds: float = 300.0,
) -> None:
    converter = _resolve_ifcconvert_executable(ifcconvert_path)
    command = [converter, native_ifc_filename, output_filename]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "IfcConvert did not finish within " + str(int(timeout_seconds)) +
            " seconds. The export was cancelled."
        ) from error
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            "IfcConvert failed with return code " + str(completed.returncode) +
            ("\n" + details if details else "")
        )


def _temporary_filename_near(target_filename: str, suffix: str | None = None) -> str:
    """Return a unique temporary path beside ``target_filename``.

    Converters can hang or prompt internally when writing directly to an existing
    file.  A same-directory temporary file keeps the final replace atomic on the
    same filesystem while avoiding an existing output path during conversion.
    """
    target_dir = os.path.dirname(os.path.abspath(target_filename)) or os.getcwd()
    os.makedirs(target_dir, exist_ok=True)
    final_suffix = suffix if suffix is not None else os.path.splitext(target_filename)[1]
    fd, temp_filename = tempfile.mkstemp(
        prefix="anystructure_export_",
        suffix=final_suffix,
        dir=target_dir,
    )
    os.close(fd)
    try:
        os.remove(temp_filename)
    except OSError:
        pass
    return temp_filename


def _write_ifc_atomic(model: Any, target_filename: str) -> None:
    temp_filename = _temporary_filename_near(target_filename, ".ifc")
    try:
        model.write(temp_filename)
        os.replace(temp_filename, target_filename)
    except Exception:
        try:
            os.remove(temp_filename)
        except OSError:
            pass
        raise


def _convert_ifc_atomic(native_ifc_filename: str, target_filename: str,
                        ifcconvert_path: str | None = None) -> None:
    temp_filename = _temporary_filename_near(target_filename)
    try:
        _convert_ifc_with_ifcconvert(native_ifc_filename, temp_filename, ifcconvert_path=ifcconvert_path)
        os.replace(temp_filename, target_filename)
    except Exception:
        try:
            os.remove(temp_filename)
        except OSError:
            pass
        raise



def _guid() -> str:
    return ifcopenshell.guid.new()


def _pos_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(value) or value < 0.0:
        return float(default)
    return value


def _normalise_length_to_m(value: Any, default: float = 0.0) -> float:
    """Accept values stored either in metres or millimetres."""
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(value) or value <= 0.0:
        return float(default)
    return value / 1000.0 if value > 100.0 else value


def _safe_getter(obj: Any, getter_names: Iterable[str], attr_names: Iterable[str], default: float) -> float:
    if obj is None:
        return float(default)
    for getter_name in getter_names:
        try:
            getter = getattr(obj, getter_name)
            return float(getter())
        except Exception:
            pass
    for attr_name in attr_names:
        try:
            return float(getattr(obj, attr_name))
        except Exception:
            pass
    return float(default)


def _safe_string_getter(obj: Any, getter_names: Iterable[str], attr_names: Iterable[str], default: str) -> str:
    if obj is None:
        return default
    for getter_name in getter_names:
        try:
            getter = getattr(obj, getter_name)
            value = getter()
            if value:
                return str(value)
        except Exception:
            pass
    for attr_name in attr_names:
        try:
            value = getattr(obj, attr_name)
            if value:
                return str(value)
        except Exception:
            pass
    return default


def _safe_application_float(app: Any, obj: Any, getter_names: Iterable[str], attr_names: Iterable[str], default: float) -> float:
    try:
        return float(app._safe_obj_float(obj, tuple(getter_names), tuple(attr_names), default))
    except Exception:
        return _safe_getter(obj, getter_names, attr_names, default)


def _section_dimensions_from_app(app: Any, section_obj: Any) -> SectionDimensions:
    try:
        return SectionDimensions.from_mapping(app._get_section_3d_dimensions(section_obj))
    except Exception:
        return SectionDimensions(
            spacing=_safe_getter(section_obj, ("get_s",), ("spacing", "s"), 0.75),
            plate_thk=_safe_getter(section_obj, ("get_pl_thk",), ("plate_thk", "pl_thk", "thk"), 0.02),
            web_h=_safe_getter(section_obj, ("get_web_h",), ("web_h", "hw"), 0.4),
            web_thk=_safe_getter(section_obj, ("get_web_thk",), ("web_thk", "tw"), 0.012),
            flange_w=_safe_getter(section_obj, ("get_fl_w",), ("fl_w", "b"), 0.15),
            flange_thk=_safe_getter(section_obj, ("get_fl_thk",), ("fl_thk", "tf"), 0.02),
            type=_safe_string_getter(section_obj, ("get_stiffener_type",), ("stiffener_type", "stf_type"), "T"),
        )


def _positions_from_length_and_spacing(length: float, spacing: float, include_ends: bool = True,
                                       max_count: int = 80) -> list[float]:
    """Return stiffener/ring positions.

    When boundary members are requested, the input spacing is treated as the
    maximum target spacing. Positions are spread evenly between 0 and length so
    the last bay is not larger than the rest.
    """
    length = _pos_float(length, 0.0)
    spacing = _pos_float(spacing, 0.0)
    if length <= EPS:
        return [0.0]
    if spacing <= EPS:
        return [0.0, length] if include_ends else [length / 2.0]

    if include_ends:
        interval_count = max(1, int(math.ceil(length / spacing)))
        interval_count = min(interval_count, max(1, int(max_count)))
        return [float(length) * idx / interval_count for idx in range(interval_count + 1)]

    positions: list[float] = [0.0] if include_ends else []
    next_pos = spacing
    count_guard = 0
    while next_pos < length - EPS and count_guard < max_count:
        positions.append(float(next_pos))
        next_pos += spacing
        count_guard += 1

    if not positions:
        return [length / 2.0]

    # Final de-duplication/sorting guard for very small or unusual dimensions.
    clean_positions: list[float] = []
    for pos in sorted(float(p) for p in positions):
        pos = min(max(pos, 0.0), float(length))
        if not clean_positions or abs(pos - clean_positions[-1]) > max(10.0 * EPS, 1.0e-9):
            clean_positions.append(pos)
    return clean_positions or [length / 2.0]


def _ring_member_half_width(dims: SectionDimensions | None) -> float:
    """Return the axial half-width occupied by a ring member."""

    if dims is None:
        return 0.0
    return max(float(dims.web_thk), float(dims.flange_w), 0.0) / 2.0


def _ring_positions_without_heavy_frame_overlap(
        positions: Iterable[float],
        frame_positions: Iterable[float],
        ring_half_width: float,
        frame_half_width: float,
        tolerance: float = EPS,
) -> list[float]:
    """Suppress ordinary ring stiffeners whose axial footprint overlaps heavy frames."""

    ring_half_width = max(_pos_float(ring_half_width, 0.0), 0.0)
    frame_half_width = max(_pos_float(frame_half_width, 0.0), 0.0)
    tolerance = max(_pos_float(tolerance, EPS), 0.0)
    heavy_positions = [float(pos) for pos in frame_positions]
    filtered_positions: list[float] = []
    for pos in (float(value) for value in positions):
        if any(abs(pos - frame_pos) <= ring_half_width + frame_half_width + tolerance
               for frame_pos in heavy_positions):
            continue
        filtered_positions.append(pos)
    return filtered_positions


def _support_positions_from_length_and_span(length: float, span: float, max_count: int = 80) -> list[float]:
    """Return centered girder/support stations while preserving the bay span."""

    length = _pos_float(length, 0.0)
    span = _pos_float(span, 0.0)
    if length <= EPS:
        return [0.0]
    if span <= EPS:
        return [0.0, length]

    full_span_count = min(int(math.floor(length / span)), max(0, int(max_count)))
    if full_span_count <= 0:
        return []

    offset = (length - full_span_count * span) / 2.0
    positions = [offset + span * idx for idx in range(full_span_count + 1)]
    if offset <= EPS:
        positions[0] = 0.0
    if abs(positions[-1] - length) <= EPS:
        positions[-1] = length
    return positions


def _bay_ranges_from_support_positions(length: float, supports: Iterable[float], support_gap: float = 0.0) -> list[tuple[float, float]]:
    """Return member segment ranges split by internal support/girder lines."""

    length = _pos_float(length, 0.0)
    support_gap = max(_pos_float(support_gap, 0.0), 0.0)
    if length <= EPS:
        return []

    internal_supports = sorted(
        pos for pos in (float(value) for value in supports)
        if EPS < pos < length - EPS
    )
    breakpoints = [0.0] + internal_supports + [length]
    ranges: list[tuple[float, float]] = []
    for x0, x1 in zip(breakpoints[:-1], breakpoints[1:]):
        left_gap = support_gap / 2.0 if any(abs(x0 - support) <= EPS for support in internal_supports) else 0.0
        right_gap = support_gap / 2.0 if any(abs(x1 - support) <= EPS for support in internal_supports) else 0.0
        bay_x0 = max(x0 + left_gap, 0.0)
        bay_x1 = min(x1 - right_gap, length)
        if bay_x1 > bay_x0:
            ranges.append((bay_x0, bay_x1))
    return ranges


def _flat_lg_from_objects(app: Any, girder: Any, stiffener: Any, spacing: float) -> float:
    for obj in (girder, stiffener):
        if obj is None:
            continue
        for attr_name in ("girder_lg", "lg", "LG"):
            try:
                value = getattr(obj, attr_name)
                lg = _normalise_length_to_m(value, 0.0)
                if lg > EPS:
                    return lg
            except Exception:
                pass
        for getter_name in ("get_girder_lg", "get_lg", "get_LG"):
            try:
                value = getattr(obj, getter_name)()
                lg = _normalise_length_to_m(value, 0.0)
                if lg > EPS:
                    return lg
            except Exception:
                pass
    try:
        lg = _normalise_length_to_m(app._new_girder_length_LG.get(), 0.0)
        if lg > EPS:
            return lg
    except Exception:
        pass
    return max(4.0 * spacing, 0.8)


def _flat_lp_from_gui(app: Any, span: float, spacing: float) -> float:
    try:
        lp = _normalise_length_to_m(app._new_panel_length_Lp.get(), 0.0)
        if lp > EPS:
            return lp
    except Exception:
        pass
    return max(2.0 * span, 2.0 * spacing, 0.8)


def _normalise_export_length_unit(length_unit: str | None) -> tuple[str, float, str | None]:
    unit = str(length_unit or "m").strip().lower()
    if unit in {"m", "meter", "metre", "meters", "metres"}:
        return "m", 1.0, None
    if unit in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}:
        return "mm", 1000.0, "MILLI"
    raise ValueError("Unsupported export length unit: " + str(length_unit))


def _normalise_transformation_scale(transformation_scale: float | str | None) -> float:
    if transformation_scale in [None, ""]:
        return 1.0
    try:
        scale = float(transformation_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("Transformation scale must be a number.") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Transformation scale must be a positive finite number.")
    return scale


def _scale_value(value: float, length_scale: float) -> float:
    return float(value) * float(length_scale)


def _scale_point(point: Iterable[float], length_scale: float) -> tuple[float, ...]:
    return tuple(_scale_value(v, length_scale) for v in point)


def _cartesian_point(ctx: IfcContext, point: Iterable[float]) -> Any:
    return ctx.model.createIfcCartesianPoint(_scale_point(point, ctx.length_scale))


def _axis2_placement_3d(model: Any, location=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                        ref_direction=(1.0, 0.0, 0.0), length_scale: float = 1.0) -> Any:
    return model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint(_scale_point(location, length_scale)),
        model.createIfcDirection(tuple(float(v) for v in axis)),
        model.createIfcDirection(tuple(float(v) for v in ref_direction)),
    )


def _axis2_placement_2d(model: Any, location=(0.0, 0.0), ref_direction=(1.0, 0.0),
                        length_scale: float = 1.0) -> Any:
    return model.createIfcAxis2Placement2D(
        model.createIfcCartesianPoint(_scale_point(location, length_scale)),
        model.createIfcDirection(tuple(float(v) for v in ref_direction)),
    )


def _local_placement(model: Any, location=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                     ref_direction=(1.0, 0.0, 0.0), relative_to=None, length_scale: float = 1.0) -> Any:
    return model.createIfcLocalPlacement(
        relative_to,
        _axis2_placement_3d(model, location, axis, ref_direction, length_scale=length_scale),
    )


def _create_basic_context(filename: str, project_name: str, material_name: str = "Steel",
                          length_unit: str = "m", transformation_scale: float | str | None = 1.0) -> IfcContext:
    normalised_unit, unit_scale, si_prefix = _normalise_export_length_unit(length_unit)
    transformation_scale = _normalise_transformation_scale(transformation_scale)
    length_scale = unit_scale * transformation_scale
    model = ifcopenshell.file(schema="IFC4")

    world = _axis2_placement_3d(model)
    body_context = model.createIfcGeometricRepresentationContext(
        "Model",
        "Model",
        3,
        1.0e-5,
        world,
        model.createIfcDirection((0.0, 1.0, 0.0)),
    )
    body_subcontext = model.createIfcGeometricRepresentationSubContext(
        "Body",
        "Model",
        None,
        None,
        None,
        None,
        body_context,
        None,
        "MODEL_VIEW",
        None,
    )

    length_unit_obj = model.createIfcSIUnit(None, "LENGTHUNIT", si_prefix, "METRE")
    area_unit = model.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE")
    volume_unit = model.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE")
    unit_assignment = model.createIfcUnitAssignment([length_unit_obj, area_unit, volume_unit])

    project = model.createIfcProject(
        _guid(),
        None,
        project_name,
        "ANYstructure model export - swept solids, not preview mesh",
        None,
        None,
        None,
        [body_context],
        unit_assignment,
    )

    site = model.createIfcSite(_guid(), None, "Site", None, None, _local_placement(model), None, None, "ELEMENT", None, None, None, None, None)
    building = model.createIfcBuilding(_guid(), None, "ANYstructure", None, None, _local_placement(model), None, None, "ELEMENT", None, None, None)
    storey = model.createIfcBuildingStorey(_guid(), None, "Structure", None, None, _local_placement(model), None, None, "ELEMENT", 0.0)

    model.createIfcRelAggregates(_guid(), None, "Project hierarchy", None, project, [site])
    model.createIfcRelAggregates(_guid(), None, "Site hierarchy", None, site, [building])
    model.createIfcRelAggregates(_guid(), None, "Building hierarchy", None, building, [storey])

    material = model.createIfcMaterial(material_name, None, None)

    return IfcContext(
        model=model,
        project=project,
        site=site,
        building=building,
        storey=storey,
        body_context=body_subcontext,
        material=material,
        summary=ExportSummary(filename=filename, project_name=project_name),
        length_unit=normalised_unit,
        length_scale=length_scale,
        transformation_scale=transformation_scale,
    )


def _assign_to_storey(ctx: IfcContext, element: Any) -> None:
    ctx.model.createIfcRelContainedInSpatialStructure(
        _guid(), None, "Contained in ANYstructure storey", None, [element], ctx.storey
    )


def _assign_material(ctx: IfcContext, element: Any) -> None:
    ctx.model.createIfcRelAssociatesMaterial(
        _guid(), None, "Material", None, [element], ctx.material
    )


def _track_solid_operand(ctx: IfcContext, element: Any, operand: Any) -> None:
    if not ctx.boolean_join_all_solids:
        return
    ctx.solid_operands.append(operand)
    ctx.solid_source_elements.append(element)


def _remove_product_from_export_tree(ctx: IfcContext, element: Any) -> None:
    """Detach an IFC product and its direct relationships from the output tree."""
    try:
        inverse_entities = list(ctx.model.get_inverse(element))
    except Exception:
        inverse_entities = []
    for inverse in inverse_entities:
        try:
            ctx.model.remove(inverse)
        except Exception:
            pass
    try:
        ctx.model.remove(element)
    except Exception:
        pass


def _replace_solid_parts_with_single_product(ctx: IfcContext, active_line: str) -> None:
    if not ctx.boolean_join_all_solids:
        return
    if len(ctx.solid_operands) < 2:
        return

    name = f"{active_line} Complete joined model"
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "SolidModel",
        list(ctx.solid_operands),
    )
    shape = ctx.model.createIfcProductDefinitionShape(None, None, [rep])
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, "IfcBuildingElementProxy", name, placement, shape, "ELEMENT")
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    _add_property_set(ctx, element, "ANYstructureDimensions", {
        "model_type": "single_product_complete_model",
        "join_method": "single_ifc_product_multi_solid_body",
        "source_solid_part_count": int(len(ctx.solid_source_elements)),
        "thickness_exported_as_geometry": True,
    })

    for source_element in list(ctx.solid_source_elements):
        _remove_product_from_export_tree(ctx, source_element)

    ctx.summary.elements[:] = [name]
    ctx.summary.warnings.append(
        "Solid export parts were written as one IFC product with multiple solid bodies. "
        "This avoids the expensive global boolean UNION that can make downstream CAD/FE tools hang."
    )


def _add_property_set(ctx: IfcContext, element: Any, name: str, values: dict[str, Any]) -> None:
    properties = []
    for key, value in values.items():
        try:
            if isinstance(value, bool):
                nominal = ctx.model.createIfcBoolean(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                nominal = ctx.model.createIfcReal(float(value))
            else:
                nominal = ctx.model.createIfcLabel(str(value))
            properties.append(ctx.model.createIfcPropertySingleValue(str(key), None, nominal, None))
        except Exception:
            pass
    if not properties:
        return
    pset = ctx.model.createIfcPropertySet(_guid(), None, name, None, properties)
    ctx.model.createIfcRelDefinesByProperties(_guid(), None, name, None, [element], pset)


def _product_shape_from_solid(ctx: IfcContext, solid: Any) -> Any:
    """Compatibility helper retained for any legacy/internal calls.

    New ANYstructure CAD export is shell/surface based and uses
    _product_shape_from_faces().
    """
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "SweptSolid",
        [solid],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])


def _product_shape_from_csg_solid(ctx: IfcContext, solid: Any) -> Any:
    """Create a product shape from an IFC CSG/boolean solid result."""
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "CSG",
        [solid],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])


def _create_oriented_rectangular_swept_solid(
    ctx: IfcContext,
    center: tuple[float, float, float],
    local_x: tuple[float, float, float],
    local_z: tuple[float, float, float],
    xdim: float,
    ydim: float,
    depth: float,
) -> Any:
    """Create a rectangular swept solid positioned/oriented in model coordinates.

    The rectangle is centred on the local X/Y axes and extruded along local Z.
    Unlike _add_oriented_box_element(), the placement is embedded directly in the
    solid.  This makes the solid usable as an operand in IFC boolean operations.
    """
    xdim = max(float(xdim), EPS)
    ydim = max(float(ydim), EPS)
    depth = max(float(depth), EPS)
    profile = ctx.model.createIfcRectangleProfileDef(
        "AREA",
        None,
        _axis2_placement_2d(ctx.model, length_scale=ctx.length_scale),
        _scale_value(xdim, ctx.length_scale),
        _scale_value(ydim, ctx.length_scale),
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model, center, axis=local_z, ref_direction=local_x,
                            length_scale=ctx.length_scale),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        _scale_value(depth, ctx.length_scale),
    )


def _create_solid_cylinder_for_boolean(ctx: IfcContext, radius: float, z0: float, z1: float) -> Any:
    """Create a solid analytic cylinder for cutting/intersecting member roots."""
    radius = max(float(radius), EPS)
    z0 = float(z0)
    z1 = float(z1)
    depth = max(abs(z1 - z0), EPS)
    profile = ctx.model.createIfcCircleProfileDef(
        "AREA",
        None,
        _axis2_placement_2d(ctx.model, length_scale=ctx.length_scale),
        _scale_value(radius, ctx.length_scale),
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model, (0.0, 0.0, min(z0, z1)), length_scale=ctx.length_scale),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        _scale_value(depth, ctx.length_scale),
    )


def _add_cylinder_fitted_longitudinal_web_solid(
    ctx: IfcContext,
    name: str,
    radius: float,
    shell_thk: float,
    angle: float,
    length: float,
    dims: SectionDimensions,
    side_sign: float,
    predefined_type: str | None = "STUD",
    extra_properties: dict[str, Any] | None = None,
) -> Any:
    """Add a solid longitudinal web with a curved root fitted to the cylinder.

    A straight rectangular web placed tangent to a circular cylinder only touches
    the cylinder at the web centreline.  At the web toes the root line is offset
    from the cylindrical shell, which can create apparent gaps or non-joinable
    geometry after IfcConvert/PrePoMax import.

    This function intentionally lets the rectangular web blank overlap the
    cylinder slightly and then uses an analytic IFC boolean cylinder to trim it:
      * outside stiffeners: web_blank - solid_cylinder(R_outer)
      * inside stiffeners:  web_blank ∩ solid_cylinder(R_inner)

    The resulting root is cylindrical and coincident with the cylinder surface.
    """
    radius = max(float(radius), EPS)
    shell_thk = max(float(shell_thk), EPS)
    web_h = max(float(dims.web_h), EPS)
    web_t = max(float(dims.web_thk), EPS)
    length = max(float(length), EPS)
    sign = 1.0 if side_sign >= 0.0 else -1.0

    c = math.cos(angle)
    s = math.sin(angle)
    radial = (sign * c, sign * s, 0.0)

    # Interface radius where the web root must match the cylindrical shell.
    interface_r = max(radius + sign * shell_thk, EPS)

    # Small overlap gives the boolean cutter something to remove.  Use a value
    # tied to the actual web/shell dimensions so it is robust for small models.
    root_overlap = max(min(shell_thk, web_h * 0.25), 1.0e-5)

    if sign >= 0.0:
        # Blank extends from slightly inside the outside cylinder surface to the
        # stiffener tip.  Difference removes the inner part and leaves a curved root.
        blank_inner_r = max(interface_r - root_overlap, EPS)
        blank_outer_r = interface_r + web_h
        blank_center_r = 0.5 * (blank_inner_r + blank_outer_r)
        blank_xdim = blank_outer_r - blank_inner_r
        cutter = _create_solid_cylinder_for_boolean(ctx, interface_r, 0.0, length)
        boolean_operator = "DIFFERENCE"
    else:
        # Blank extends from the inward stiffener tip to slightly outside the
        # inside cylinder surface.  Intersection keeps only the part inside the
        # cylinder radius and leaves a curved root.
        blank_inner_r = max(interface_r - web_h, EPS)
        blank_outer_r = interface_r + root_overlap
        blank_center_r = 0.5 * (blank_inner_r + blank_outer_r)
        blank_xdim = blank_outer_r - blank_inner_r
        cutter = _create_solid_cylinder_for_boolean(ctx, interface_r, 0.0, length)
        boolean_operator = "INTERSECTION"

    blank = _create_oriented_rectangular_swept_solid(
        ctx,
        (blank_center_r * c, blank_center_r * s, 0.0),
        radial,
        (0.0, 0.0, 1.0),
        blank_xdim,
        web_t,
        length,
    )
    fitted = ctx.model.createIfcBooleanResult(boolean_operator, blank, cutter)
    shape = _product_shape_from_csg_solid(ctx, fitted)
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, "IfcMember", name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)

    props = {
        "model_type": "csg_cylinder_fitted_solid",
        "thickness_exported_as_geometry": True,
        "role": "longitudinal stiffener",
        "part": "web",
        "angle_rad": float(angle),
        "web_height_m": web_h,
        "nominal_web_thickness_m": web_t,
        "shell_interface_radius_m": float(radius),
        "member_base_radius_m": float(interface_r),
        "shell_thickness_m": float(shell_thk),
        "root_overlap_m": float(root_overlap),
        "root_fit_boolean": boolean_operator,
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    _track_solid_operand(ctx, element, fitted)
    return element


def _ifc_faces_from_points(ctx: IfcContext, faces: Iterable[Iterable[tuple[float, float, float]]]) -> list[Any]:
    ifc_faces = []
    for face_points in faces:
        pts = []
        for point in face_points:
            if len(point) != 3:
                raise ValueError("IFC surface point must have three coordinates.")
            pts.append(_cartesian_point(ctx, point))
        if len(pts) < 3:
            continue
        poly_loop = ctx.model.createIfcPolyLoop(pts)
        outer_bound = ctx.model.createIfcFaceOuterBound(poly_loop, True)
        ifc_faces.append(ctx.model.createIfcFace([outer_bound]))
    return ifc_faces


def _product_shape_from_closed_faces(ctx: IfcContext, faces: Iterable[Iterable[tuple[float, float, float]]]) -> Any:
    """Create a faceted B-rep solid from closed polygon faces."""
    brep = _faceted_brep_from_closed_faces(ctx, faces)
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "Brep",
        [brep],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])


def _faceted_brep_from_closed_faces(ctx: IfcContext, faces: Iterable[Iterable[tuple[float, float, float]]]) -> Any:
    ifc_faces = _ifc_faces_from_points(ctx, faces)
    if not ifc_faces:
        raise ValueError("No valid IFC solid faces were generated.")
    closed_shell = ctx.model.createIfcClosedShell(ifc_faces)
    return ctx.model.createIfcFacetedBrep(closed_shell)


def _product_shape_from_faces(ctx: IfcContext, faces: Iterable[Iterable[tuple[float, float, float]]]) -> Any:
    """Create an IFC surface model from rectangular/planar faces.

    This is the core of the shell export.  Each face is a zero-thickness IFC face.
    The result is a FaceBasedSurfaceModel, not a solid and not the preview mesh.
    """
    ifc_faces = _ifc_faces_from_points(ctx, faces)
    if not ifc_faces:
        raise ValueError("No valid IFC surface faces were generated.")

    face_set = ctx.model.createIfcConnectedFaceSet(ifc_faces)
    surface_model = ctx.model.createIfcFaceBasedSurfaceModel([face_set])
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "SurfaceModel",
        [surface_model],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])




def _ifc_parameter_value(ctx: IfcContext, value: float) -> Any:
    """Create an IFC parameter value for IfcTrimmedCurve trim arguments."""
    try:
        return ctx.model.create_entity("IfcParameterValue", float(value))
    except Exception:
        return float(value)


def _vertex_point(ctx: IfcContext, point: tuple[float, float, float]) -> Any:
    return ctx.model.createIfcVertexPoint(
        _cartesian_point(ctx, point)
    )


def _oriented_edge(ctx: IfcContext, edge: Any, orientation: bool = True) -> Any:
    """Create IfcOrientedEdge with derived start/end attributes omitted."""
    try:
        return ctx.model.createIfcOrientedEdge(None, None, edge, bool(orientation))
    except Exception:
        return ctx.model.create_entity("IfcOrientedEdge", None, None, edge, bool(orientation))


def _edge_curve(ctx: IfcContext, start_vertex: Any, end_vertex: Any, curve: Any,
                same_sense: bool = True) -> Any:
    return ctx.model.createIfcEdgeCurve(start_vertex, end_vertex, curve, bool(same_sense))


def _line_curve_between(ctx: IfcContext, p0: tuple[float, float, float],
                        p1: tuple[float, float, float]) -> Any:
    return ctx.model.createIfcPolyline([
        _cartesian_point(ctx, p0),
        _cartesian_point(ctx, p1),
    ])


def _trimmed_circle_curve_3d(ctx: IfcContext, radius: float, z: float,
                             theta_start: float, theta_end: float) -> Any:
    """Create a 3D circular arc curve at constant Z.

    This is used only as a boundary curve for the analytic cylindrical face.  The
    face geometry itself is carried by IfcCylindricalSurface.
    """
    circle = ctx.model.createIfcCircle(
        _axis2_placement_3d(
            ctx.model,
            location=(0.0, 0.0, float(z)),
            axis=(0.0, 0.0, 1.0),
            ref_direction=(1.0, 0.0, 0.0),
            length_scale=ctx.length_scale,
        ),
        _scale_value(max(float(radius), EPS), ctx.length_scale),
    )
    return ctx.model.createIfcTrimmedCurve(
        circle,
        [_ifc_parameter_value(ctx, theta_start)],
        [_ifc_parameter_value(ctx, theta_end)],
        True,
        "PARAMETER",
    )


def _advanced_cylindrical_face_with_curved_bounds(
    ctx: IfcContext,
    cylindrical_surface: Any,
    radius: float,
    z_min: float,
    z_max: float,
    theta_start: float,
    theta_end: float,
) -> Any:
    """Create one bounded analytic cylindrical face.

    The previous attempts used either profile sweeps or polygon loops.  Several
    IfcConvert/OpenCascade import paths treated those as caps or planar faces.
    Here the boundary loop itself contains circular arc edge curves, while the
    actual face is still an IfcCylindricalSurface.  This keeps the model as a
    zero-thickness shell but gives the converter enough information to build a
    cylindrical surface rather than a lid.
    """
    p00 = _cyl_point(radius, theta_start, z_min)
    p10 = _cyl_point(radius, theta_end, z_min)
    p11 = _cyl_point(radius, theta_end, z_max)
    p01 = _cyl_point(radius, theta_start, z_max)

    v00 = _vertex_point(ctx, p00)
    v10 = _vertex_point(ctx, p10)
    v11 = _vertex_point(ctx, p11)
    v01 = _vertex_point(ctx, p01)

    bottom_arc = _edge_curve(
        ctx, v00, v10,
        _trimmed_circle_curve_3d(ctx, radius, z_min, theta_start, theta_end),
        True,
    )
    side_at_end = _edge_curve(
        ctx, v10, v11,
        _line_curve_between(ctx, p10, p11),
        True,
    )
    top_arc = _edge_curve(
        ctx, v01, v11,
        _trimmed_circle_curve_3d(ctx, radius, z_max, theta_start, theta_end),
        True,
    )
    side_at_start = _edge_curve(
        ctx, v00, v01,
        _line_curve_between(ctx, p00, p01),
        True,
    )

    # Loop order: bottom a0->a1, vertical up, top a1->a0 using reversed top
    # edge, vertical down using reversed start edge.
    loop = ctx.model.createIfcEdgeLoop([
        _oriented_edge(ctx, bottom_arc, True),
        _oriented_edge(ctx, side_at_end, True),
        _oriented_edge(ctx, top_arc, False),
        _oriented_edge(ctx, side_at_start, False),
    ])
    outer = ctx.model.createIfcFaceOuterBound(loop, True)

    try:
        return ctx.model.createIfcAdvancedFace([outer], cylindrical_surface, True)
    except Exception:
        return ctx.model.createIfcFaceSurface([outer], cylindrical_surface, True)


def _product_shape_from_advanced_cylindrical_shell_with_arc_bounds(
    ctx: IfcContext,
    radius: float,
    z0: float,
    z1: float,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * math.pi,
    split_count: int = 2,
) -> Any:
    """Create analytic, zero-thickness cylindrical IFC shell faces.

    Representation used:
        IfcShellBasedSurfaceModel
          IfcOpenShell
            IfcAdvancedFace / IfcFaceSurface
              IfcCylindricalSurface
              IfcFaceOuterBound(IfcEdgeLoop with circular arc EdgeCurves)

    This is not a solid and not a thickness workaround.  The full cylinder is
    split into two analytic half-cylinder faces to avoid a coincident 360-degree
    seam while still avoiding the many-flat-plate representation.
    """
    radius = max(float(radius), EPS)
    z0 = float(z0)
    z1 = float(z1)
    z_min = min(z0, z1)
    z_max = max(z0, z1)
    theta_start = float(theta_start)
    theta_end = float(theta_end)

    if z_max - z_min <= EPS:
        raise ValueError("Cylindrical IFC shell surface has zero length.")
    if abs(theta_end - theta_start) <= EPS:
        raise ValueError("Cylindrical IFC shell surface has zero angular extent.")

    angular_extent = abs(theta_end - theta_start)
    is_full_circle = abs(angular_extent - 2.0 * math.pi) <= 1.0e-7
    face_count = max(1, int(split_count)) if is_full_circle else 1

    cylindrical_surface = ctx.model.createIfcCylindricalSurface(
        _axis2_placement_3d(
            ctx.model,
            location=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            ref_direction=(1.0, 0.0, 0.0),
            length_scale=ctx.length_scale,
        ),
        _scale_value(radius, ctx.length_scale),
    )

    ifc_faces = []
    for idx in range(face_count):
        a0 = theta_start + (theta_end - theta_start) * idx / face_count
        a1 = theta_start + (theta_end - theta_start) * (idx + 1) / face_count
        ifc_faces.append(
            _advanced_cylindrical_face_with_curved_bounds(
                ctx, cylindrical_surface, radius, z_min, z_max, a0, a1
            )
        )

    open_shell = ctx.model.createIfcOpenShell(ifc_faces)
    surface_model = ctx.model.createIfcShellBasedSurfaceModel([open_shell])
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "SurfaceModel",
        [surface_model],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])


def _add_analytic_swept_cylindrical_shell_surface(
    ctx: IfcContext,
    ifc_class: str,
    name: str,
    radius: float,
    z0: float,
    z1: float,
    predefined_type: str | None = None,
    extra_properties: dict[str, Any] | None = None,
) -> Any:
    """Add a full circular cylinder as analytic zero-thickness IFC shell faces."""
    shape = _product_shape_from_advanced_cylindrical_shell_with_arc_bounds(ctx, radius, z0, z1)
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)

    props = {
        "model_type": "analytic_zero_thickness_ifc_cylindrical_shell",
        "geometry_role": "full_cylindrical_shell_as_bounded_advanced_surface",
        "ifc_geometry": "IfcShellBasedSurfaceModel/IfcAdvancedFace/IfcCylindricalSurface/curved EdgeLoop bounds",
        "continuous_cylinder_export": True,
        "shell_export": True,
        "thickness_exported_as_geometry": False,
        "radius_m": float(radius),
        "z0_m": float(z0),
        "z1_m": float(z1),
        "theta_start_rad": 0.0,
        "theta_end_rad": float(2.0 * math.pi),
    }
    if extra_properties:
        props.update(extra_properties)
        props["model_type"] = "analytic_zero_thickness_ifc_cylindrical_shell"
        props["ifc_geometry"] = "IfcShellBasedSurfaceModel/IfcAdvancedFace/IfcCylindricalSurface/curved EdgeLoop bounds"
        props["continuous_cylinder_export"] = True
        props["shell_export"] = True
        props["thickness_exported_as_geometry"] = False

    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    return element


def _add_surface_element(ctx: IfcContext, ifc_class: str, name: str,
                         faces: Iterable[Iterable[tuple[float, float, float]]],
                         predefined_type: str | None = None,
                         extra_properties: dict[str, Any] | None = None) -> Any:
    """Add a zero-thickness shell/surface element.

    The placement is identity because all face coordinates are already global within
    the exported local ANYstructure model.  This avoids unintended offsets and makes
    web/flange/plate interfaces share exact coordinates.
    """
    shape = _product_shape_from_faces(ctx, faces)
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "model_type": "zero_thickness_shell_surface",
        "thickness_exported_as_geometry": False,
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    return element


def _add_faceted_solid_element(ctx: IfcContext, ifc_class: str, name: str,
                               faces: Iterable[Iterable[tuple[float, float, float]]],
                               predefined_type: str | None = None,
                               extra_properties: dict[str, Any] | None = None) -> Any:
    brep = _faceted_brep_from_closed_faces(ctx, faces)
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "Brep",
        [brep],
    )
    shape = ctx.model.createIfcProductDefinitionShape(None, None, [rep])
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "model_type": "faceted_brep_solid",
        "thickness_exported_as_geometry": True,
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    _track_solid_operand(ctx, element, brep)
    return element


def _rect_face_xy(x0: float, x1: float, y0: float, y1: float, z: float) -> list[tuple[float, float, float]]:
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def _rect_face_xz(x0: float, x1: float, y: float, z0: float, z1: float) -> list[tuple[float, float, float]]:
    return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]


def _rect_face_yz(x: float, y0: float, y1: float, z0: float, z1: float) -> list[tuple[float, float, float]]:
    return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]


def _cyl_point(radius: float, angle: float, z: float) -> tuple[float, float, float]:
    return (radius * math.cos(angle), radius * math.sin(angle), z)


def _cylindrical_faces(radius: float, z0: float, z1: float, theta_start: float,
                       theta_end: float, segments: int) -> list[list[tuple[float, float, float]]]:
    segments = max(1, int(segments))
    faces = []
    for i in range(segments):
        a0 = theta_start + (theta_end - theta_start) * i / segments
        a1 = theta_start + (theta_end - theta_start) * (i + 1) / segments
        faces.append([
            _cyl_point(radius, a0, z0),
            _cyl_point(radius, a1, z0),
            _cyl_point(radius, a1, z1),
            _cyl_point(radius, a0, z1),
        ])
    return faces


def _conical_faces(radius0: float, radius1: float, z0: float, z1: float, theta_start: float,
                   theta_end: float, segments: int) -> list[list[tuple[float, float, float]]]:
    segments = max(1, int(segments))
    faces = []
    for i in range(segments):
        a0 = theta_start + (theta_end - theta_start) * i / segments
        a1 = theta_start + (theta_end - theta_start) * (i + 1) / segments
        faces.append([
            _cyl_point(radius0, a0, z0),
            _cyl_point(radius0, a1, z0),
            _cyl_point(radius1, a1, z1),
            _cyl_point(radius1, a0, z1),
        ])
    return faces


def _annular_radial_faces(radius0: float, radius1: float, z: float, theta_start: float,
                          theta_end: float, segments: int) -> list[list[tuple[float, float, float]]]:
    segments = max(1, int(segments))
    faces = []
    for i in range(segments):
        a0 = theta_start + (theta_end - theta_start) * i / segments
        a1 = theta_start + (theta_end - theta_start) * (i + 1) / segments
        faces.append([
            _cyl_point(radius0, a0, z),
            _cyl_point(radius0, a1, z),
            _cyl_point(radius1, a1, z),
            _cyl_point(radius1, a0, z),
        ])
    return faces


def _ring_web_shell_faces(radius0: float, radius1: float, z: float, theta_start: float,
                          theta_end: float) -> list[list[tuple[float, float, float]]]:
    """Return one curved-footprint shell face per ring-web sector.

    Ring webs are physical plate fields, not FE mesh strips.  A full circular
    ring is represented by two 180-degree shell faces, matching the cylindrical
    shell convention.  A bounded cylinder-panel sector is represented by one
    shell face.  Each face is a single polygon loop with sampled arc boundaries,
    so the web remains visible as an annular plate instead of collapsing to a
    chord rectangle.
    """
    radius0, radius1 = sorted((max(float(radius0), EPS), max(float(radius1), EPS)))
    theta_start = float(theta_start)
    theta_end = float(theta_end)
    angular_extent = abs(theta_end - theta_start)
    is_full_circle = abs(angular_extent - 2.0 * math.pi) <= 1.0e-7
    face_count = 2 if is_full_circle else 1
    samples_per_face = max(8, int(math.ceil(angular_extent / face_count / (math.pi / 18.0))))
    faces: list[list[tuple[float, float, float]]] = []
    for face_index in range(face_count):
        a0 = theta_start + (theta_end - theta_start) * face_index / face_count
        a1 = theta_start + (theta_end - theta_start) * (face_index + 1) / face_count
        outer = [
            _cyl_point(radius1, a0 + (a1 - a0) * sample_index / samples_per_face, z)
            for sample_index in range(samples_per_face + 1)
        ]
        inner = [
            _cyl_point(radius0, a0 + (a1 - a0) * sample_index / samples_per_face, z)
            for sample_index in range(samples_per_face, -1, -1)
        ]
        faces.append(outer + inner)
    return faces


def _conical_wall_solid_faces(inner0: float, outer0: float, inner1: float, outer1: float,
                              z0: float, z1: float, theta_start: float, theta_end: float,
                              segments: int) -> list[list[tuple[float, float, float]]]:
    """Return closed faceted faces for a conical/frustum wall solid."""
    inner0, outer0 = sorted((max(float(inner0), EPS), max(float(outer0), EPS)))
    inner1, outer1 = sorted((max(float(inner1), EPS), max(float(outer1), EPS)))
    z0, z1 = float(z0), float(z1)
    segments = max(1, int(segments))
    faces: list[list[tuple[float, float, float]]] = []

    for i in range(segments):
        a0 = theta_start + (theta_end - theta_start) * i / segments
        a1 = theta_start + (theta_end - theta_start) * (i + 1) / segments
        faces.append([_cyl_point(outer0, a0, z0), _cyl_point(outer0, a1, z0),
                      _cyl_point(outer1, a1, z1), _cyl_point(outer1, a0, z1)])
        faces.append([_cyl_point(inner0, a1, z0), _cyl_point(inner0, a0, z0),
                      _cyl_point(inner1, a0, z1), _cyl_point(inner1, a1, z1)])
        faces.append([_cyl_point(inner0, a0, z0), _cyl_point(outer0, a0, z0),
                      _cyl_point(outer0, a1, z0), _cyl_point(inner0, a1, z0)])
        faces.append([_cyl_point(inner1, a0, z1), _cyl_point(inner1, a1, z1),
                      _cyl_point(outer1, a1, z1), _cyl_point(outer1, a0, z1)])

    is_full_circle = abs(abs(theta_end - theta_start) - 2.0 * math.pi) <= 1.0e-7
    if not is_full_circle:
        for angle in (theta_start, theta_end):
            faces.append([_cyl_point(inner0, angle, z0), _cyl_point(inner1, angle, z1),
                          _cyl_point(outer1, angle, z1), _cyl_point(outer0, angle, z0)])
    return faces


def _cylindrical_wall_solid_faces(radius0: float, radius1: float, z0: float, z1: float,
                                  theta_start: float, theta_end: float,
                                  segments: int) -> list[list[tuple[float, float, float]]]:
    """Return closed faceted faces for a cylindrical/sector wall solid."""
    r0, r1 = sorted((max(float(radius0), EPS), max(float(radius1), EPS)))
    z0, z1 = sorted((float(z0), float(z1)))
    segments = max(1, int(segments))
    faces: list[list[tuple[float, float, float]]] = []

    for i in range(segments):
        a0 = theta_start + (theta_end - theta_start) * i / segments
        a1 = theta_start + (theta_end - theta_start) * (i + 1) / segments
        faces.append([_cyl_point(r1, a0, z0), _cyl_point(r1, a1, z0),
                      _cyl_point(r1, a1, z1), _cyl_point(r1, a0, z1)])
        faces.append([_cyl_point(r0, a1, z0), _cyl_point(r0, a0, z0),
                      _cyl_point(r0, a0, z1), _cyl_point(r0, a1, z1)])
        faces.append([_cyl_point(r0, a0, z0), _cyl_point(r1, a0, z0),
                      _cyl_point(r1, a1, z0), _cyl_point(r0, a1, z0)])
        faces.append([_cyl_point(r0, a0, z1), _cyl_point(r0, a1, z1),
                      _cyl_point(r1, a1, z1), _cyl_point(r1, a0, z1)])

    is_full_circle = abs(abs(theta_end - theta_start) - 2.0 * math.pi) <= 1.0e-7
    if not is_full_circle:
        for angle in (theta_start, theta_end):
            faces.append([_cyl_point(r0, angle, z0), _cyl_point(r0, angle, z1),
                          _cyl_point(r1, angle, z1), _cyl_point(r1, angle, z0)])
    return faces


def _segment_count_for_arc(radius: float, theta_start: float, theta_end: float,
                           min_segments: int = 8, max_segments: int = 144) -> int:
    arc_length = abs(theta_end - theta_start) * max(radius, EPS)
    # About 0.25 m target facet length.  This is a geometric shell model, not the
    # Matplotlib preview mesh; the segmentation is only needed because portable IFC
    # surface models are face based.
    return max(min_segments, min(max_segments, int(math.ceil(arc_length / 0.25))))



def _create_rectangular_swept_solid(ctx: IfcContext, xdim: float, ydim: float, depth: float) -> Any:
    xdim = max(float(xdim), EPS)
    ydim = max(float(ydim), EPS)
    depth = max(float(depth), EPS)
    profile = ctx.model.createIfcRectangleProfileDef(
        "AREA",
        None,
        _axis2_placement_2d(ctx.model, length_scale=ctx.length_scale),
        _scale_value(xdim, ctx.length_scale),
        _scale_value(ydim, ctx.length_scale),
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model, length_scale=ctx.length_scale),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        _scale_value(depth, ctx.length_scale),
    )


def _create_circle_hollow_swept_solid(ctx: IfcContext, outer_radius: float, wall_thickness: float,
                                      depth: float) -> Any:
    outer_radius = max(float(outer_radius), EPS)
    wall_thickness = min(max(float(wall_thickness), EPS), outer_radius * 0.95)
    depth = max(float(depth), EPS)
    profile = ctx.model.createIfcCircleHollowProfileDef(
        "AREA",
        None,
        _axis2_placement_2d(ctx.model, length_scale=ctx.length_scale),
        _scale_value(outer_radius, ctx.length_scale),
        _scale_value(wall_thickness, ctx.length_scale),
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model, length_scale=ctx.length_scale),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        _scale_value(depth, ctx.length_scale),
    )


def _create_positioned_circle_hollow_swept_solid(ctx: IfcContext, outer_radius: float, wall_thickness: float,
                                                 z0: float, z1: float) -> Any:
    outer_radius = max(float(outer_radius), EPS)
    wall_thickness = min(max(float(wall_thickness), EPS), outer_radius * 0.95)
    depth = max(abs(float(z1) - float(z0)), EPS)
    profile = ctx.model.createIfcCircleHollowProfileDef(
        "AREA",
        None,
        _axis2_placement_2d(ctx.model, length_scale=ctx.length_scale),
        _scale_value(outer_radius, ctx.length_scale),
        _scale_value(wall_thickness, ctx.length_scale),
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model, (0.0, 0.0, min(float(z0), float(z1))),
                            length_scale=ctx.length_scale),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        _scale_value(depth, ctx.length_scale),
    )


def _add_box_element(ctx: IfcContext, ifc_class: str, name: str, x0: float, x1: float, y0: float, y1: float,
                     z0: float, z1: float, predefined_type: str | None = None,
                     extra_properties: dict[str, Any] | None = None) -> Any:
    """Add an axis-aligned rectangular swept solid as IFC element."""
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    z0, z1 = sorted((float(z0), float(z1)))
    xdim = max(x1 - x0, EPS)
    ydim = max(y1 - y0, EPS)
    depth = max(z1 - z0, EPS)
    solid = _create_rectangular_swept_solid(ctx, xdim, ydim, depth)
    shape = _product_shape_from_solid(ctx, solid)
    placement = _local_placement(ctx.model, ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0),
                                 length_scale=ctx.length_scale)
    element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "x0_m": x0, "x1_m": x1, "y0_m": y0, "y1_m": y1, "z0_m": z0, "z1_m": z1,
        "length_x_m": xdim, "length_y_m": ydim, "depth_z_m": depth,
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    _track_solid_operand(
        ctx,
        element,
        _create_oriented_rectangular_swept_solid(
            ctx,
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            xdim,
            ydim,
            depth,
        ),
    )
    return element


def _create_building_element(ctx: IfcContext, ifc_class: str, name: str, placement: Any, shape: Any,
                             predefined_type: str | None = None) -> Any:
    kwargs = {
        "GlobalId": _guid(),
        "OwnerHistory": None,
        "Name": name,
        "Description": "Exported by ANYstructure IfcOpenShell model exporter",
        "ObjectType": None,
        "ObjectPlacement": placement,
        "Representation": shape,
        "Tag": name,
    }
    try:
        element = ctx.model.create_entity(ifc_class, **kwargs)
    except TypeError:
        element = ctx.model.create_entity(ifc_class, _guid(), None, name, None, None, placement, shape, name)
    if predefined_type is not None:
        try:
            element.PredefinedType = predefined_type
        except Exception:
            pass
    return element


def _add_oriented_box_element(ctx: IfcContext, ifc_class: str, name: str, center: tuple[float, float, float],
                              local_x: tuple[float, float, float], local_z: tuple[float, float, float],
                              xdim: float, ydim: float, depth: float,
                              predefined_type: str | None = None,
                              extra_properties: dict[str, Any] | None = None) -> Any:
    """Add a rectangular swept solid with local X/ref direction and local Z/extrusion axis."""
    solid = _create_rectangular_swept_solid(ctx, xdim, ydim, depth)
    shape = _product_shape_from_solid(ctx, solid)
    placement = _local_placement(ctx.model, center, axis=local_z, ref_direction=local_x,
                                 length_scale=ctx.length_scale)
    element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "profile_x_m": float(xdim),
        "profile_y_m": float(ydim),
        "extrusion_depth_m": float(depth),
        "center_x_m": float(center[0]),
        "center_y_m": float(center[1]),
        "center_z_m": float(center[2]),
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    _track_solid_operand(
        ctx,
        element,
        _create_oriented_rectangular_swept_solid(ctx, center, local_x, local_z, xdim, ydim, depth),
    )
    return element


def _add_plate_box(ctx: IfcContext, name: str, x0: float, x1: float, y0: float, y1: float,
                   z0: float, z1: float, extra_properties: dict[str, Any] | None = None,
                   shell_export: bool = True) -> Any:
    """Add a plate either as a shell face or as a rectangular swept solid."""
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    z0, z1 = sorted((float(z0), float(z1)))
    nominal_thickness = abs(float(z1) - float(z0))
    if not shell_export:
        props = {
            "nominal_thickness_m": nominal_thickness,
            "shell_export": False,
            "role": "plate",
            "model_type": "swept_solid",
            "thickness_exported_as_geometry": True,
        }
        if extra_properties:
            props.update(extra_properties)
        return _add_box_element(
            ctx, "IfcPlate", name, x0, x1, y0, y1, z0, z1,
            predefined_type="SHEET",
            extra_properties=props,
        )

    props = {
        "nominal_thickness_m": nominal_thickness,
        "shell_export": True,
        "role": "plate",
    }
    if extra_properties:
        props.update(extra_properties)
    return _add_surface_element(
        ctx, "IfcPlate", name,
        [_rect_face_xy(x0, x1, y0, y1, 0.0)],
        predefined_type="SHEET",
        extra_properties=props,
    )


def _add_member_web_and_flange(ctx: IfcContext, base_name: str, orientation: str, x_center: float, y_center: float,
                               length: float, plate_thk: float, dims: SectionDimensions,
                               x_limits: tuple[float, float] | None = None,
                               y_limits: tuple[float, float] | None = None,
                               side_sign: float = 1.0,
                               member_role: str = "stiffener",
                               shell_export: bool = True) -> None:
    """Add member web/flange either as shell plates or swept rectangular solids."""
    web_h = max(dims.web_h, 0.0)
    web_t = max(dims.web_thk, 0.0)
    fl_w = max(dims.flange_w, 0.0)
    fl_t = max(dims.flange_thk, 0.0)
    sec_type = str(dims.type or "T")
    if length <= EPS or web_h <= EPS:
        return

    sign = 1.0 if side_sign >= 0.0 else -1.0
    z_base = 0.0
    z_tip = sign * web_h
    if shell_export:
        web_z = (z_base, z_tip)
        flange_z = z_tip
    elif sign >= 0.0:
        web_z = (plate_thk, plate_thk + web_h)
        flange_z = (plate_thk + web_h, plate_thk + web_h + fl_t)
    else:
        web_z = (-web_h, 0.0)
        flange_z = (-(web_h + fl_t), -web_h)

    if orientation == "x":
        x0 = x_center - length / 2.0
        x1 = x_center + length / 2.0
        if x_limits is not None:
            x0 = max(x0, x_limits[0])
            x1 = min(x1, x_limits[1])
        if x1 <= x0:
            return

        web_props = {
            "role": member_role,
            "part": "web",
            "section_type": sec_type,
            "nominal_web_thickness_m": web_t,
            "web_height_m": web_h,
            "plate_interface_z_m": 0.0,
        }
        if shell_export:
            _add_surface_element(
                ctx, "IfcMember", base_name + " Web",
                [_rect_face_xz(x0, x1, y_center, web_z[0], web_z[1])],
                predefined_type="STUD",
                extra_properties=web_props,
            )
        else:
            web_props.update({
                "model_type": "swept_solid",
                "thickness_exported_as_geometry": True,
            })
            _add_box_element(
                ctx, "IfcMember", base_name + " Web",
                x0, x1, y_center - web_t / 2.0, y_center + web_t / 2.0,
                web_z[0], web_z[1],
                predefined_type="STUD",
                extra_properties=web_props,
            )

        if fl_w > EPS and sec_type.upper() not in {"FB", "FLAT", "FLATBAR"}:
            if sec_type in ["L", "L-bulb"]:
                y0 = y_center
                y1 = y_center + fl_w
            else:
                y0 = y_center - fl_w / 2.0
                y1 = y_center + fl_w / 2.0
            flange_props = {
                "role": member_role,
                "part": "flange",
                "section_type": sec_type,
                "nominal_flange_thickness_m": fl_t,
                "flange_width_m": fl_w,
                "web_interface_z_m": z_tip,
            }
            if shell_export:
                _add_surface_element(
                    ctx, "IfcMember", base_name + " Flange",
                    [_rect_face_xy(x0, x1, y0, y1, flange_z)],
                    predefined_type="STUD",
                    extra_properties=flange_props,
                )
            elif fl_t > EPS:
                flange_props.update({
                    "model_type": "swept_solid",
                    "thickness_exported_as_geometry": True,
                })
                _add_box_element(
                    ctx, "IfcMember", base_name + " Flange",
                    x0, x1, y0, y1, flange_z[0], flange_z[1],
                    predefined_type="STUD",
                    extra_properties=flange_props,
                )
    else:
        y0 = y_center - length / 2.0
        y1 = y_center + length / 2.0
        if y_limits is not None:
            y0 = max(y0, y_limits[0])
            y1 = min(y1, y_limits[1])
        if y1 <= y0:
            return

        web_props = {
            "role": member_role,
            "part": "web",
            "section_type": sec_type,
            "nominal_web_thickness_m": web_t,
            "web_height_m": web_h,
            "plate_interface_z_m": 0.0,
        }
        if shell_export:
            _add_surface_element(
                ctx, "IfcMember", base_name + " Web",
                [_rect_face_yz(x_center, y0, y1, web_z[0], web_z[1])],
                predefined_type="STUD",
                extra_properties=web_props,
            )
        else:
            web_props.update({
                "model_type": "swept_solid",
                "thickness_exported_as_geometry": True,
            })
            _add_box_element(
                ctx, "IfcMember", base_name + " Web",
                x_center - web_t / 2.0, x_center + web_t / 2.0,
                y0, y1, web_z[0], web_z[1],
                predefined_type="STUD",
                extra_properties=web_props,
            )

        if fl_w > EPS and sec_type.upper() not in {"FB", "FLAT", "FLATBAR"}:
            if sec_type in ["L", "L-bulb"]:
                x0 = x_center
                x1 = x_center + fl_w
            else:
                x0 = x_center - fl_w / 2.0
                x1 = x_center + fl_w / 2.0
            flange_props = {
                "role": member_role,
                "part": "flange",
                "section_type": sec_type,
                "nominal_flange_thickness_m": fl_t,
                "flange_width_m": fl_w,
                "web_interface_z_m": z_tip,
            }
            if shell_export:
                _add_surface_element(
                    ctx, "IfcMember", base_name + " Flange",
                    [_rect_face_xy(x0, x1, y0, y1, flange_z)],
                    predefined_type="STUD",
                    extra_properties=flange_props,
                )
            elif fl_t > EPS:
                flange_props.update({
                    "model_type": "swept_solid",
                    "thickness_exported_as_geometry": True,
                })
                _add_box_element(
                    ctx, "IfcMember", base_name + " Flange",
                    x0, x1, y0, y1, flange_z[0], flange_z[1],
                    predefined_type="STUD",
                    extra_properties=flange_props,
                )


def _add_flat_structure(ctx: IfcContext, app: Any, all_obj: Any, active_line: str, side_sign: float,
                        shell_export: bool) -> None:
    plate = getattr(all_obj, "Plate", None)
    stiffener = getattr(all_obj, "Stiffener", None)
    girder = getattr(all_obj, "Girder", None)
    if plate is None:
        raise ValueError("The selected line has no plate object to export.")

    spacing = max(_safe_getter(plate, ("get_s",), ("spacing", "s"), 0.75), EPS)
    plate_thk = max(_safe_getter(plate, ("get_pl_thk",), ("plate_thk", "pl_thk", "thk"), 0.02), EPS)
    span = max(_safe_application_float(app, plate, ("get_span",), ("span",), 2.0), EPS)

    if girder is not None:
        width = _flat_lp_from_gui(app, span, spacing)
        length = _flat_lg_from_objects(app, girder, stiffener, spacing)
        girder_xs = _support_positions_from_length_and_span(width, span, max_count=80)
        gdims = _section_dimensions_from_app(app, girder)
        sdims = _section_dimensions_from_app(app, stiffener) if stiffener is not None else None
        _add_plate_box(
            ctx, f"{active_line} Plate field", 0.0, width, 0.0, length, 0.0, plate_thk,
            extra_properties={"active_line": active_line, "panel_type": "flat panel with girder"},
            shell_export=shell_export,
        )
        for index, x_pos in enumerate(girder_xs, start=1):
            _add_member_web_and_flange(
                ctx, f"{active_line} Girder {index:03d}", "y", x_pos, length / 2.0, length,
                plate_thk, gdims, y_limits=(0.0, length), side_sign=side_sign, member_role="girder",
                shell_export=shell_export,
            )
        if sdims is not None:
            stiffener_ys = _positions_from_length_and_spacing(length, spacing, include_ends=True, max_count=80)
            girder_gap = 0.0 if shell_export else max(gdims.web_thk, 0.0)
            bay_ranges = _bay_ranges_from_support_positions(width, girder_xs, girder_gap)
            for index, y in enumerate(stiffener_ys, start=1):
                for bay_index, (bay_x0, bay_x1) in enumerate(bay_ranges, start=1):
                    if bay_x1 <= bay_x0:
                        continue
                    _add_member_web_and_flange(
                        ctx, f"{active_line} Stiffener {index:03d} Bay {bay_index:03d}", "x",
                        (bay_x0 + bay_x1) / 2.0, y, bay_x1 - bay_x0, plate_thk,
                        sdims, x_limits=(bay_x0, bay_x1), side_sign=side_sign,
                        member_role="stiffener", shell_export=shell_export,
                    )
    else:
        if stiffener is not None:
            width = max(span, spacing, 0.8)
            length = _flat_lg_from_objects(app, None, stiffener, spacing)
        else:
            width = max(spacing, 0.8)
            length = max(span, 0.8)
        _add_plate_box(
            ctx, f"{active_line} Plate field", 0.0, width, 0.0, length, 0.0, plate_thk,
            extra_properties={"active_line": active_line, "panel_type": "flat stiffened panel" if stiffener else "flat plate"},
            shell_export=shell_export,
        )
        if stiffener is not None:
            sdims = _section_dimensions_from_app(app, stiffener)
            stiffener_ys = _positions_from_length_and_spacing(length, spacing, include_ends=True, max_count=80)
            for index, y in enumerate(stiffener_ys, start=1):
                _add_member_web_and_flange(
                    ctx, f"{active_line} Stiffener {index:03d}", "x", width / 2.0, y,
                    width, plate_thk, sdims, x_limits=(0.0, width), side_sign=side_sign,
                    member_role="stiffener", shell_export=shell_export,
                )


def _is_cylinder_panel(app: Any, cyl_obj: Any) -> bool:
    try:
        return bool(app._is_cylinder_panel_preview(cyl_obj))
    except Exception:
        try:
            domain = str(cyl_obj.geometry).lower()
        except Exception:
            return False
        return "panel" in domain and "shell" not in domain


def _cylinder_theta_range(app: Any, cyl_obj: Any) -> tuple[float, float]:
    try:
        return tuple(app._cylinder_preview_theta_range(cyl_obj))
    except Exception:
        if _is_cylinder_panel(app, cyl_obj):
            half_span = math.radians(60.0) / 2.0
            return -half_span, half_span
        return 0.0, 2.0 * math.pi


def _add_cylindrical_shell_surface(ctx: IfcContext, ifc_class: str, name: str, radius: float,
                                   z0: float, z1: float, theta_start: float, theta_end: float,
                                   predefined_type: str | None = None,
                                   extra_properties: dict[str, Any] | None = None) -> None:
    radius = max(float(radius), EPS)
    z0 = float(z0)
    z1 = float(z1)

    # Full circular cylinders/ring flanges should not be faceted into many flat
    # plates.  Use an analytic swept curve surface.  This remains a shell model:
    # no wall thickness is exported as geometry.
    is_full_circle = abs(abs(float(theta_end) - float(theta_start)) - 2.0 * math.pi) <= 1.0e-7
    if is_full_circle:
        _add_analytic_swept_cylindrical_shell_surface(
            ctx, ifc_class, name, radius, z0, z1,
            predefined_type=predefined_type,
            extra_properties=extra_properties,
        )
        return

    # Bounded cylindrical sectors are still faceted.  A proper analytic sector
    # would need IFC trimming curves and is intentionally kept separate from the
    # full-cylinder case.
    segments = _segment_count_for_arc(radius, theta_start, theta_end)
    props = {
        "radius_m": radius,
        "z0_m": z0,
        "z1_m": z1,
        "theta_start_rad": float(theta_start),
        "theta_end_rad": float(theta_end),
        "surface_segments": int(segments),
        "shell_export": True,
        "thickness_exported_as_geometry": False,
        "model_type": "faceted_zero_thickness_cylindrical_sector_surface",
    }
    if extra_properties:
        props.update(extra_properties)
        props["shell_export"] = True
        props["thickness_exported_as_geometry"] = False
    _add_surface_element(
        ctx, ifc_class, name,
        _cylindrical_faces(radius, z0, z1, theta_start, theta_end, segments),
        predefined_type=predefined_type,
        extra_properties=props,
    )


def _add_conical_shell_surface(ctx: IfcContext, ifc_class: str, name: str, radius0: float, radius1: float,
                               z0: float, z1: float, theta_start: float, theta_end: float,
                               predefined_type: str | None = None,
                               extra_properties: dict[str, Any] | None = None) -> None:
    radius0 = max(float(radius0), EPS)
    radius1 = max(float(radius1), EPS)
    z0 = float(z0)
    z1 = float(z1)
    segments = _segment_count_for_arc(max(radius0, radius1), theta_start, theta_end)
    props = {
        "radius0_m": radius0,
        "radius1_m": radius1,
        "z0_m": z0,
        "z1_m": z1,
        "theta_start_rad": float(theta_start),
        "theta_end_rad": float(theta_end),
        "surface_segments": int(segments),
        "shell_export": True,
        "geometry_role": "conical_shell_frustum",
    }
    if extra_properties:
        props.update(extra_properties)
    _add_surface_element(
        ctx, ifc_class, name,
        _conical_faces(radius0, radius1, z0, z1, theta_start, theta_end, segments),
        predefined_type=predefined_type,
        extra_properties=props,
    )


def _add_cylindrical_wall_solid(ctx: IfcContext, ifc_class: str, name: str, radius: float, thickness: float,
                                z0: float, z1: float, theta_start: float, theta_end: float,
                                side_sign: float, predefined_type: str | None = None,
                                extra_properties: dict[str, Any] | None = None) -> None:
    """Add a cylindrical wall as true thickness geometry.

    ``radius`` is the plate/member interface radius.  Positive side exports the
    wall outside that interface, negative side exports it inside.  This keeps the
    model connected at the same radius used by the shell export.
    """
    radius = max(float(radius), EPS)
    thickness = max(float(thickness), EPS)
    z0 = float(z0)
    z1 = float(z1)
    sign = 1.0 if side_sign >= 0.0 else -1.0
    if sign >= 0.0:
        inner_radius = radius
        outer_radius = radius + thickness
    else:
        inner_radius = max(radius - thickness, EPS)
        outer_radius = radius

    props = {
        "interface_radius_m": radius,
        "inner_radius_m": inner_radius,
        "outer_radius_m": outer_radius,
        "thickness_m": thickness,
        "z0_m": z0,
        "z1_m": z1,
        "theta_start_rad": float(theta_start),
        "theta_end_rad": float(theta_end),
        "shell_export": False,
    }
    if extra_properties:
        props.update(extra_properties)

    is_full_circle = abs(abs(theta_end - theta_start) - 2.0 * math.pi) <= 1.0e-7
    if is_full_circle:
        solid = _create_circle_hollow_swept_solid(ctx, outer_radius, outer_radius - inner_radius, abs(z1 - z0))
        joined_operand = _create_positioned_circle_hollow_swept_solid(
            ctx, outer_radius, outer_radius - inner_radius, z0, z1
        )
        shape = _product_shape_from_solid(ctx, solid)
        placement = _local_placement(ctx.model, (0.0, 0.0, min(z0, z1)), length_scale=ctx.length_scale)
        element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
        _assign_to_storey(ctx, element)
        _assign_material(ctx, element)
        props.update({
            "model_type": "hollow_swept_solid",
            "thickness_exported_as_geometry": True,
        })
        _add_property_set(ctx, element, "ANYstructureDimensions", props)
        ctx.summary.elements.append(name)
        _track_solid_operand(ctx, element, joined_operand)
        return

    segments = _segment_count_for_arc(radius, theta_start, theta_end)
    props.update({
        "model_type": "faceted_brep_solid",
        "surface_segments": int(segments),
        "thickness_exported_as_geometry": True,
    })
    _add_faceted_solid_element(
        ctx, ifc_class, name,
        _cylindrical_wall_solid_faces(inner_radius, outer_radius, z0, z1, theta_start, theta_end, segments),
        predefined_type=predefined_type,
        extra_properties=props,
    )


def _add_conical_wall_solid(ctx: IfcContext, ifc_class: str, name: str, radius0: float, radius1: float,
                            thickness: float, z0: float, z1: float, theta_start: float, theta_end: float,
                            side_sign: float, predefined_type: str | None = None,
                            extra_properties: dict[str, Any] | None = None) -> None:
    radius0 = max(float(radius0), EPS)
    radius1 = max(float(radius1), EPS)
    thickness = max(float(thickness), EPS)
    sign = 1.0 if side_sign >= 0.0 else -1.0
    if sign >= 0.0:
        inner0, inner1 = radius0, radius1
        outer0, outer1 = radius0 + thickness, radius1 + thickness
    else:
        inner0, inner1 = max(radius0 - thickness, EPS), max(radius1 - thickness, EPS)
        outer0, outer1 = radius0, radius1

    segments = _segment_count_for_arc(max(radius0, radius1), theta_start, theta_end)
    props = {
        "interface_radius0_m": radius0,
        "interface_radius1_m": radius1,
        "inner_radius0_m": inner0,
        "inner_radius1_m": inner1,
        "outer_radius0_m": outer0,
        "outer_radius1_m": outer1,
        "thickness_m": thickness,
        "z0_m": float(z0),
        "z1_m": float(z1),
        "theta_start_rad": float(theta_start),
        "theta_end_rad": float(theta_end),
        "surface_segments": int(segments),
        "model_type": "faceted_brep_solid",
        "shell_export": False,
        "thickness_exported_as_geometry": True,
        "geometry_role": "conical_shell_frustum",
    }
    if extra_properties:
        props.update(extra_properties)
    _add_faceted_solid_element(
        ctx, ifc_class, name,
        _conical_wall_solid_faces(inner0, outer0, inner1, outer1, z0, z1, theta_start, theta_end, segments),
        predefined_type=predefined_type,
        extra_properties=props,
    )


def _add_cylinder_longitudinal_members(ctx: IfcContext, active_line: str, radius: float, length: float,
                                       angles: Iterable[float], dims: SectionDimensions,
                                       shell_thk: float, side_sign: float,
                                       shell_export: bool) -> None:
    """Add longitudinal stiffeners as shell surfaces or true solids.

    The web begins exactly on the shell surface at radius R.  The flange is a
    tangential plate at radius R +/- web_h.  For solid export the web and flange
    touch at their interface dimensions without gaps.
    """
    sign = 1.0 if side_sign >= 0.0 else -1.0
    for index, angle in enumerate(angles, start=1):
        c = math.cos(angle)
        s = math.sin(angle)
        radial = (sign * c, sign * s, 0.0)
        tangential = (-s, c, 0.0)
        base_r = radius if shell_export else max(radius + sign * shell_thk, EPS)
        tip_r = max(radius + sign * dims.web_h, EPS)
        if not shell_export:
            tip_r = max(base_r + sign * dims.web_h, EPS)

        if dims.web_h > EPS:
            web_props = {
                "role": "longitudinal stiffener",
                "part": "web",
                "angle_rad": float(angle),
                "web_height_m": float(dims.web_h),
                "nominal_web_thickness_m": float(dims.web_thk),
                "shell_interface_radius_m": float(radius),
                "member_base_radius_m": float(base_r),
                "shell_thickness_m": float(shell_thk),
            }
            if shell_export:
                web_face = [
                    _cyl_point(base_r, angle, 0.0),
                    _cyl_point(base_r, angle, length),
                    _cyl_point(tip_r, angle, length),
                    _cyl_point(tip_r, angle, 0.0),
                ]
                _add_surface_element(
                    ctx, "IfcMember", f"{active_line} Longitudinal {index:03d} Web",
                    [web_face], predefined_type="STUD", extra_properties=web_props,
                )
            else:
                _add_cylinder_fitted_longitudinal_web_solid(
                    ctx,
                    f"{active_line} Longitudinal {index:03d} Web",
                    radius=radius,
                    shell_thk=shell_thk,
                    angle=angle,
                    length=length,
                    dims=dims,
                    side_sign=side_sign,
                    predefined_type="STUD",
                    extra_properties=web_props,
                )

        if dims.flange_w > EPS and str(dims.type or "T").upper() not in {"FB", "FLAT", "FLATBAR"}:
            half = dims.flange_w / 2.0
            flange_props = {
                "role": "longitudinal stiffener",
                "part": "flange",
                "angle_rad": float(angle),
                "flange_width_m": float(dims.flange_w),
                "nominal_flange_thickness_m": float(dims.flange_thk),
                "web_interface_radius_m": float(tip_r),
            }
            if shell_export:
                # Tangential strip at the web tip.  This shares its centreline with the web tip.
                p0 = (tip_r * c - half * tangential[0], tip_r * s - half * tangential[1], 0.0)
                p1 = (tip_r * c + half * tangential[0], tip_r * s + half * tangential[1], 0.0)
                p2 = (tip_r * c + half * tangential[0], tip_r * s + half * tangential[1], length)
                p3 = (tip_r * c - half * tangential[0], tip_r * s - half * tangential[1], length)
                _add_surface_element(
                    ctx, "IfcMember", f"{active_line} Longitudinal {index:03d} Flange",
                    [[p0, p1, p2, p3]], predefined_type="STUD", extra_properties=flange_props,
                )
            elif dims.flange_thk > EPS:
                flange_props.update({
                    "model_type": "swept_solid",
                    "thickness_exported_as_geometry": True,
                })
                flange_center_r = tip_r + sign * dims.flange_thk / 2.0
                _add_oriented_box_element(
                    ctx, "IfcMember", f"{active_line} Longitudinal {index:03d} Flange",
                    (flange_center_r * c, flange_center_r * s, 0.0),
                    tangential, (0.0, 0.0, 1.0),
                    dims.flange_w, dims.flange_thk, length,
                    predefined_type="STUD", extra_properties=flange_props,
                )


def _add_ring_set(ctx: IfcContext, active_line: str, role: str, radius: float, positions: Iterable[float],
                  dims: SectionDimensions, side_sign: float,
                  theta_start: float = 0.0, theta_end: float = 2.0 * math.pi,
                  shell_export: bool = True, shell_thk: float = 0.0) -> None:
    """Add ring stiffeners/frames as shell surfaces or true solids.

    Ring web: annular radial plate at z = ring position.\n
    Ring flange: cylindrical surface at the web tip with axial width = flange_w.\n
    Shell surfaces intersect at shared lines; solid boxes/walls share interface
    radii and axial faces.
    """
    sign = 1.0 if side_sign >= 0.0 else -1.0
    segments = _segment_count_for_arc(radius, theta_start, theta_end)
    base_radius = radius if shell_export else max(radius + sign * max(shell_thk, 0.0), EPS)
    for index, z_pos in enumerate(positions, start=1):
        z_pos = float(z_pos)
        tip_r = max(base_radius + sign * dims.web_h, EPS)
        r0, r1 = sorted((base_radius, tip_r))
        if dims.web_h > EPS:
            web_props = {
                "role": role,
                "part": "web",
                "z_position_m": z_pos,
                "web_height_m": float(dims.web_h),
                "nominal_web_thickness_m": float(dims.web_thk),
                "shell_interface_radius_m": float(radius),
                "member_base_radius_m": float(base_radius),
                "theta_start_rad": float(theta_start),
                "theta_end_rad": float(theta_end),
            }
            if shell_export:
                _add_surface_element(
                    ctx, "IfcMember", f"{active_line} {role} {index:03d} Web",
                    _ring_web_shell_faces(r0, r1, z_pos, theta_start, theta_end),
                    predefined_type="MEMBER",
                    extra_properties=web_props,
                )
            else:
                web_props.update({
                    "model_type": "faceted_brep_solid",
                    "surface_segments": int(segments),
                    "thickness_exported_as_geometry": True,
                })
                _add_faceted_solid_element(
                    ctx, "IfcMember", f"{active_line} {role} {index:03d} Web",
                    _cylindrical_wall_solid_faces(r0, r1,
                                                  z_pos - max(dims.web_thk, EPS) / 2.0,
                                                  z_pos + max(dims.web_thk, EPS) / 2.0,
                                                  theta_start, theta_end, segments),
                    predefined_type="MEMBER",
                    extra_properties=web_props,
                )
        if dims.flange_w > EPS and str(dims.type or "T").upper() not in {"FB", "FLAT", "FLATBAR"}:
            flange_props = {
                "role": role,
                "part": "flange",
                "z_position_m": z_pos,
                "flange_width_m": float(dims.flange_w),
                "nominal_flange_thickness_m": float(dims.flange_thk),
                "web_interface_radius_m": float(tip_r),
            }
            if shell_export:
                _add_cylindrical_shell_surface(
                    ctx, "IfcMember", f"{active_line} {role} {index:03d} Flange",
                    radius=tip_r,
                    z0=z_pos - dims.flange_w / 2.0,
                    z1=z_pos + dims.flange_w / 2.0,
                    theta_start=theta_start,
                    theta_end=theta_end,
                    predefined_type="MEMBER",
                    extra_properties=flange_props,
                )
            elif dims.flange_thk > EPS:
                _add_cylindrical_wall_solid(
                    ctx, "IfcMember", f"{active_line} {role} {index:03d} Flange",
                    radius=tip_r, thickness=dims.flange_thk,
                    z0=z_pos - dims.flange_w / 2.0,
                    z1=z_pos + dims.flange_w / 2.0,
                    theta_start=theta_start,
                    theta_end=theta_end,
                    side_sign=side_sign,
                    predefined_type="MEMBER",
                    extra_properties=flange_props,
                )


def _add_cylinder_structure(ctx: IfcContext, app: Any, cyl_obj: Any, active_line: str, side_sign: float,
                            shell_export: bool) -> None:
    shell = getattr(cyl_obj, "ShellObj", None)
    if shell is None:
        raise ValueError("The selected cylinder line has no ShellObj to export.")
    is_conical = getattr(cyl_obj, "geometry", None) == 9
    if is_conical:
        radius0 = _normalise_length_to_m(getattr(shell, "cone_r1", None), 0.0)
        radius1 = _normalise_length_to_m(getattr(shell, "cone_r2", None), 0.0)
        length = _normalise_length_to_m(getattr(shell, "cone_length", None), 1.0)
        if radius0 <= EPS or radius1 <= EPS:
            radius0 = radius1 = max(float(getattr(shell, "radius")), EPS)
    else:
        radius = max(float(getattr(shell, "radius")), EPS)
        length = max(float(getattr(shell, "length_of_shell")), EPS)
    thk = max(float(getattr(shell, "thk")), EPS)
    is_panel = _is_cylinder_panel(app, cyl_obj)
    theta_start, theta_end = _cylinder_theta_range(app, cyl_obj)

    if is_conical:
        shell_name = f"{active_line} Unstiffened conical shell"
        shell_props = {
            "active_line": active_line,
            "panel_type": "unstiffened conical shell",
            "nominal_shell_thickness_m": thk,
            "cone_radius_1_m": float(radius0),
            "cone_radius_2_m": float(radius1),
            "cone_length_m": float(length),
            "cone_alpha_deg": float(getattr(shell, "cone_alpha", 0.0) or 0.0),
        }
        if shell_export:
            _add_conical_shell_surface(
                ctx, "IfcPlate", shell_name,
                radius0=radius0, radius1=radius1, z0=0.0, z1=length,
                theta_start=0.0, theta_end=2.0 * math.pi,
                predefined_type="SHEET", extra_properties=shell_props,
            )
        else:
            _add_conical_wall_solid(
                ctx, "IfcPlate", shell_name,
                radius0=radius0, radius1=radius1, thickness=thk, z0=0.0, z1=length,
                theta_start=0.0, theta_end=2.0 * math.pi,
                side_sign=side_sign, predefined_type="SHEET", extra_properties=shell_props,
            )
        if getattr(cyl_obj, "LongStfObj", None) is not None or getattr(cyl_obj, "RingStfObj", None) is not None or \
                getattr(cyl_obj, "RingFrameObj", None) is not None:
            ctx.summary.warnings.append("Conical shell CAD export v1 exports the unstiffened cone only.")
        return

    shell_name = f"{active_line} Cylindrical panel shell" if is_panel else f"{active_line} Cylindrical shell"
    shell_props = {
        "active_line": active_line,
        "panel_type": "cylindrical panel shell" if is_panel else "full cylindrical shell",
        "nominal_shell_thickness_m": thk,
    }
    if shell_export:
        _add_cylindrical_shell_surface(
            ctx, "IfcPlate", shell_name,
            radius=radius, z0=0.0, z1=length,
            theta_start=theta_start if is_panel else 0.0,
            theta_end=theta_end if is_panel else 2.0 * math.pi,
            predefined_type="SHEET", extra_properties=shell_props,
        )
    else:
        _add_cylindrical_wall_solid(
            ctx, "IfcPlate", shell_name,
            radius=radius, thickness=thk, z0=0.0, z1=length,
            theta_start=theta_start if is_panel else 0.0,
            theta_end=theta_end if is_panel else 2.0 * math.pi,
            side_sign=side_sign, predefined_type="SHEET", extra_properties=shell_props,
        )

    if getattr(cyl_obj, "LongStfObj", None) is not None:
        long_dims = _section_dimensions_from_app(app, cyl_obj.LongStfObj)
        spacing = max(long_dims.spacing, EPS)
        if is_panel:
            arc_length = abs(theta_end - theta_start) * radius
            num_stf = max(2, min(72, int(round(arc_length / spacing)) + 1))
            angles = [theta_start + (theta_end - theta_start) * i / (num_stf - 1) for i in range(num_stf)]
        else:
            num_stf = max(4, min(144, int(round(2.0 * math.pi * radius / spacing))))
            angles = [2.0 * math.pi * idx / num_stf for idx in range(num_stf)]
        _add_cylinder_longitudinal_members(ctx, active_line, radius, length, angles, long_dims, thk, side_sign,
                                           shell_export=shell_export)

    frame_dims = None
    frame_positions: list[float] = []
    if getattr(cyl_obj, "RingFrameObj", None) is not None:
        frame_dims = _section_dimensions_from_app(app, cyl_obj.RingFrameObj)
        try:
            frame_spacing = _normalise_length_to_m(cyl_obj.length_between_girders, 0.0)
        except Exception:
            frame_spacing = _normalise_length_to_m(_safe_getter(cyl_obj, (), ("length_between_girders",), 0.0), 0.0)
        if frame_spacing <= EPS:
            try:
                frame_spacing = _normalise_length_to_m(app._new_shell_ring_frame_length_between_girders.get(), 0.0)
            except Exception:
                frame_spacing = 0.0
        frame_positions = [length / 2.0] if frame_spacing <= EPS else _positions_from_length_and_spacing(
            length, frame_spacing, include_ends=False, max_count=40
        )

    if getattr(cyl_obj, "RingStfObj", None) is not None:
        ring_dims = _section_dimensions_from_app(app, cyl_obj.RingStfObj)
        try:
            ring_spacing = _normalise_length_to_m(shell._dist_between_rings, 0.0)
        except Exception:
            ring_spacing = _normalise_length_to_m(_safe_getter(shell, (), ("dist_between_rings",), 0.0), 0.0)
        if ring_spacing <= EPS:
            try:
                ring_spacing = _normalise_length_to_m(app._new_shell_dist_rings.get(), 0.0)
            except Exception:
                ring_spacing = 0.0
        ring_positions = _positions_from_length_and_spacing(length, ring_spacing, include_ends=False, max_count=80)
        ring_positions = _ring_positions_without_heavy_frame_overlap(
            ring_positions,
            frame_positions,
            _ring_member_half_width(ring_dims),
            _ring_member_half_width(frame_dims),
        )
        _add_ring_set(ctx, active_line, "Ring stiffener", radius, ring_positions, ring_dims, side_sign,
                      theta_start=theta_start if is_panel else 0.0,
                      theta_end=theta_end if is_panel else 2.0 * math.pi,
                      shell_export=shell_export, shell_thk=thk)

    if frame_dims is not None:
        _add_ring_set(ctx, active_line, "Ring frame", radius, frame_positions, frame_dims, side_sign,
                      theta_start=theta_start if is_panel else 0.0,
                      theta_end=theta_end if is_panel else 2.0 * math.pi,
                      shell_export=shell_export, shell_thk=thk)


def export_selected_structure_from_application(
    app: Any,
    filename: str,
    output_format: str | None = "ifc",
    ifcconvert_path: str | None = None,
    keep_intermediate_ifc: bool = True,
    shell_export: bool = True,
    boolean_join_all_solids: bool = False,
    length_unit: str = "m",
    transformation_scale: float | str | None = 1.0,
) -> ExportSummary:
    """Export the active ANYstructure line as an IFC solid or shell model.

    The exporter reads the selected line's real structural objects.  It does not use
    _prop_3d_export_mesh, _prop_3d_shell_export_mesh, STL, UNV, meshio,
    numpy-stl, or any Matplotlib preview geometry.  Shell export creates
    zero-thickness plates with shared interface lines.  Solid export creates true
    thickness geometry with swept solids where possible and closed BReps where a
    cylindrical sector requires faceting.

    Parameters
    ----------
    filename:
        Final requested output file.  If output_format is not ``ifc``, an IFC file
        is written first and then IfcConvert is called to create the requested
        output.
    output_format:
        One of supported_export_formats().  ``ifc`` is native.  OBJ/DAE/GLB/STP/
        IGS/XML/SVG/H5/TTL/RDB/JSON require IfcConvert.
    ifcconvert_path:
        Optional internal override. Normal GUI use leaves this as None; the exporter resolves
        the bundled/package IfcConvert executable automatically.
    keep_intermediate_ifc:
        Keep the generated IFC beside converted output.  Recommended for audit and
        for re-conversion later.
    shell_export:
        When True, export zero-thickness shell/surface geometry with nominal
        thickness stored as metadata.  When False, export plate/web/flange
        thickness as geometry.
    boolean_join_all_solids:
        When True for solid export, replace the separate plate/member products
        with one IFC product containing all solid bodies for downstream meshing
        workflows.  The exporter intentionally avoids a global IFC boolean UNION
        because that can make converters and FE importers hang.
    length_unit:
        Export geometry length unit.  ``m`` writes metre coordinates; ``mm``
        writes millimetre coordinates and declares IFC length units as millimetres.
    transformation_scale:
        Positive scale factor applied to exported geometry coordinates and profile
        dimensions.  ``1.0`` exports the model at its native size.
    """
    output_format = _normalise_export_format(output_format, filename)
    length_unit, _length_scale, _si_prefix = _normalise_export_length_unit(length_unit)
    transformation_scale = _normalise_transformation_scale(transformation_scale)
    requested_filename = filename

    if output_format == "ifc":
        native_ifc_filename = filename
    else:
        root, _ext = os.path.splitext(filename)
        native_ifc_filename = root + ".ifc" if keep_intermediate_ifc else _temporary_filename_near(
            os.path.join(tempfile.gettempdir(), os.path.basename(root) + "_anystructure_export.ifc"),
            ".ifc",
        )
    if not getattr(app, "_line_is_active", False):
        raise ValueError("No active line selected. Select a line before exporting IFC.")
    active_line = getattr(app, "_active_line", "")
    if active_line not in getattr(app, "_line_to_struc", {}):
        raise ValueError("The active line has no assigned structure properties.")

    if not getattr(app, "_simplified_calculation_mode", False):
        try:
            app.set_selected_variables(active_line)
        except Exception:
            pass

    try:
        material_yield = float(app._new_material.get())
    except Exception:
        material_yield = 355.0
    material_name = f"Steel S{int(round(material_yield))}"
    project_name = "ANYstructure IFC model - " + str(active_line)
    ctx = _create_basic_context(
        native_ifc_filename,
        project_name,
        material_name=material_name,
        length_unit=length_unit,
        transformation_scale=transformation_scale,
    )
    ctx.boolean_join_all_solids = bool(boolean_join_all_solids) and not bool(shell_export)

    try:
        side_sign = -1.0 if bool(getattr(app, "_new_prop_3d_opposite_side").get()) else 1.0
    except Exception:
        side_sign = 1.0
    line_data = app._line_to_struc[active_line]

    try:
        cylinder_obj = line_data[5]
    except Exception:
        cylinder_obj = None

    if cylinder_obj is not None:
        _add_cylinder_structure(ctx, app, cylinder_obj, active_line, side_sign, shell_export=shell_export)
    else:
        try:
            all_obj = line_data[0]
        except Exception as error:
            raise ValueError("Could not read flat panel structure object from selected line.") from error
        _add_flat_structure(ctx, app, all_obj, active_line, side_sign, shell_export=shell_export)

    _replace_solid_parts_with_single_product(ctx, active_line)

    _add_property_set(ctx, ctx.project, "ANYstructureExport", {
        "active_line": active_line,
        "source": "ANYstructure",
        "geometry_source": "model parameters; not preview mesh",
        "material_yield_MPa": material_yield,
        "opposite_side": side_sign < 0.0,
        "export_model_type": "zero_thickness_shell_surface" if shell_export else "model_based_cad",
        "shell_export": bool(shell_export),
        "thickness_exported_as_geometry": not bool(shell_export),
        "boolean_join_all_solids": bool(ctx.boolean_join_all_solids),
        "export_length_unit": ctx.length_unit,
        "export_length_scale_from_m": ctx.length_scale,
        "transformation_scale": ctx.transformation_scale,
    })

    _write_ifc_atomic(ctx.model, native_ifc_filename)

    ctx.summary.output_format = output_format
    ctx.summary.native_ifc_filename = native_ifc_filename
    ctx.summary.filename = requested_filename

    if output_format != "ifc":
        _convert_ifc_atomic(native_ifc_filename, requested_filename, ifcconvert_path=ifcconvert_path)
        ctx.summary.converted_filename = requested_filename
        if not keep_intermediate_ifc:
            try:
                os.remove(native_ifc_filename)
            except OSError:
                ctx.summary.warnings.append("Temporary IFC file could not be removed: " + native_ifc_filename)
    else:
        ctx.summary.converted_filename = native_ifc_filename

    return ctx.summary
