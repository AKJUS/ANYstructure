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


def _axis2_placement_3d(model: Any, location=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                        ref_direction=(1.0, 0.0, 0.0)) -> Any:
    return model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint(tuple(float(v) for v in location)),
        model.createIfcDirection(tuple(float(v) for v in axis)),
        model.createIfcDirection(tuple(float(v) for v in ref_direction)),
    )


def _axis2_placement_2d(model: Any, location=(0.0, 0.0), ref_direction=(1.0, 0.0)) -> Any:
    return model.createIfcAxis2Placement2D(
        model.createIfcCartesianPoint(tuple(float(v) for v in location)),
        model.createIfcDirection(tuple(float(v) for v in ref_direction)),
    )


def _local_placement(model: Any, location=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                     ref_direction=(1.0, 0.0, 0.0), relative_to=None) -> Any:
    return model.createIfcLocalPlacement(
        relative_to,
        _axis2_placement_3d(model, location, axis, ref_direction),
    )


def _create_basic_context(filename: str, project_name: str, material_name: str = "Steel") -> IfcContext:
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

    length_unit = model.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE")
    area_unit = model.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE")
    volume_unit = model.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE")
    unit_assignment = model.createIfcUnitAssignment([length_unit, area_unit, volume_unit])

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
    )


def _assign_to_storey(ctx: IfcContext, element: Any) -> None:
    ctx.model.createIfcRelContainedInSpatialStructure(
        _guid(), None, "Contained in ANYstructure storey", None, [element], ctx.storey
    )


def _assign_material(ctx: IfcContext, element: Any) -> None:
    ctx.model.createIfcRelAssociatesMaterial(
        _guid(), None, "Material", None, [element], ctx.material
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


def _ifc_faces_from_points(ctx: IfcContext, faces: Iterable[Iterable[tuple[float, float, float]]]) -> list[Any]:
    ifc_faces = []
    for face_points in faces:
        pts = []
        for point in face_points:
            if len(point) != 3:
                raise ValueError("IFC surface point must have three coordinates.")
            pts.append(ctx.model.createIfcCartesianPoint(tuple(float(v) for v in point)))
        if len(pts) < 3:
            continue
        poly_loop = ctx.model.createIfcPolyLoop(pts)
        outer_bound = ctx.model.createIfcFaceOuterBound(poly_loop, True)
        ifc_faces.append(ctx.model.createIfcFace([outer_bound]))
    return ifc_faces


def _product_shape_from_closed_faces(ctx: IfcContext, faces: Iterable[Iterable[tuple[float, float, float]]]) -> Any:
    """Create a faceted B-rep solid from closed polygon faces."""
    ifc_faces = _ifc_faces_from_points(ctx, faces)
    if not ifc_faces:
        raise ValueError("No valid IFC solid faces were generated.")
    closed_shell = ctx.model.createIfcClosedShell(ifc_faces)
    brep = ctx.model.createIfcFacetedBrep(closed_shell)
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "Brep",
        [brep],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])


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
    shape = _product_shape_from_closed_faces(ctx, faces)
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
        _axis2_placement_2d(ctx.model),
        xdim,
        ydim,
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        depth,
    )


def _create_circle_hollow_swept_solid(ctx: IfcContext, outer_radius: float, wall_thickness: float,
                                      depth: float) -> Any:
    outer_radius = max(float(outer_radius), EPS)
    wall_thickness = min(max(float(wall_thickness), EPS), outer_radius * 0.95)
    depth = max(float(depth), EPS)
    profile = ctx.model.createIfcCircleHollowProfileDef(
        "AREA",
        None,
        _axis2_placement_2d(ctx.model),
        outer_radius,
        wall_thickness,
    )
    return ctx.model.createIfcExtrudedAreaSolid(
        profile,
        _axis2_placement_3d(ctx.model),
        ctx.model.createIfcDirection((0.0, 0.0, 1.0)),
        depth,
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
    placement = _local_placement(ctx.model, ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0))
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
    placement = _local_placement(ctx.model, center, axis=local_z, ref_direction=local_x)
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
        x_mid = width / 2.0
        gdims = _section_dimensions_from_app(app, girder)
        sdims = _section_dimensions_from_app(app, stiffener) if stiffener is not None else None
        _add_plate_box(
            ctx, f"{active_line} Plate field", 0.0, width, 0.0, length, 0.0, plate_thk,
            extra_properties={"active_line": active_line, "panel_type": "flat panel with girder"},
            shell_export=shell_export,
        )
        _add_member_web_and_flange(
            ctx, f"{active_line} Girder", "y", x_mid, length / 2.0, length,
            plate_thk, gdims, y_limits=(0.0, length), side_sign=side_sign, member_role="girder",
            shell_export=shell_export,
        )
        if sdims is not None:
            stiffener_ys = _positions_from_length_and_spacing(length, spacing, include_ends=True, max_count=80)
            # Shell model: no web thickness is exported, so stiffeners meet the girder
            # exactly at the girder web centreline.  This avoids both gaps and overlaps.
            left_x0, left_x1 = 0.0, x_mid
            right_x0, right_x1 = x_mid, width
            for index, y in enumerate(stiffener_ys, start=1):
                if left_x1 > left_x0:
                    _add_member_web_and_flange(
                        ctx, f"{active_line} Stiffener {index:03d} Left", "x",
                        (left_x0 + left_x1) / 2.0, y, left_x1 - left_x0, plate_thk,
                        sdims, x_limits=(left_x0, left_x1), side_sign=side_sign,
                        member_role="stiffener", shell_export=shell_export,
                    )
                if right_x1 > right_x0:
                    _add_member_web_and_flange(
                        ctx, f"{active_line} Stiffener {index:03d} Right", "x",
                        (right_x0 + right_x1) / 2.0, y, right_x1 - right_x0, plate_thk,
                        sdims, x_limits=(right_x0, right_x1), side_sign=side_sign,
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
    segments = _segment_count_for_arc(radius, theta_start, theta_end)
    props = {
        "radius_m": radius,
        "z0_m": z0,
        "z1_m": z1,
        "theta_start_rad": float(theta_start),
        "theta_end_rad": float(theta_end),
        "surface_segments": int(segments),
        "shell_export": True,
    }
    if extra_properties:
        props.update(extra_properties)
    _add_surface_element(
        ctx, ifc_class, name,
        _cylindrical_faces(radius, z0, z1, theta_start, theta_end, segments),
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
        shape = _product_shape_from_solid(ctx, solid)
        placement = _local_placement(ctx.model, (0.0, 0.0, min(z0, z1)))
        element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
        _assign_to_storey(ctx, element)
        _assign_material(ctx, element)
        props.update({
            "model_type": "hollow_swept_solid",
            "thickness_exported_as_geometry": True,
        })
        _add_property_set(ctx, element, "ANYstructureDimensions", props)
        ctx.summary.elements.append(name)
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
                web_props.update({
                    "model_type": "swept_solid",
                    "thickness_exported_as_geometry": True,
                })
                web_center_r = base_r + sign * dims.web_h / 2.0
                _add_oriented_box_element(
                    ctx, "IfcMember", f"{active_line} Longitudinal {index:03d} Web",
                    (web_center_r * c, web_center_r * s, 0.0),
                    radial, (0.0, 0.0, 1.0),
                    dims.web_h, max(dims.web_thk, EPS), length,
                    predefined_type="STUD", extra_properties=web_props,
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
                    _annular_radial_faces(r0, r1, z_pos, theta_start, theta_end, segments),
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
    radius = max(float(getattr(shell, "radius")), EPS)
    length = max(float(getattr(shell, "length_of_shell")), EPS)
    thk = max(float(getattr(shell, "thk")), EPS)
    is_panel = _is_cylinder_panel(app, cyl_obj)
    theta_start, theta_end = _cylinder_theta_range(app, cyl_obj)

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
        _add_ring_set(ctx, active_line, "Ring stiffener", radius, ring_positions, ring_dims, side_sign,
                      theta_start=theta_start if is_panel else 0.0,
                      theta_end=theta_end if is_panel else 2.0 * math.pi,
                      shell_export=shell_export, shell_thk=thk)

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
    """
    output_format = _normalise_export_format(output_format, filename)
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
    ctx = _create_basic_context(native_ifc_filename, project_name, material_name=material_name)

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

    _add_property_set(ctx, ctx.project, "ANYstructureExport", {
        "active_line": active_line,
        "source": "ANYstructure",
        "geometry_source": "model parameters; not preview mesh",
        "material_yield_MPa": material_yield,
        "opposite_side": side_sign < 0.0,
        "export_model_type": "zero_thickness_shell_surface" if shell_export else "model_based_cad",
        "shell_export": bool(shell_export),
        "thickness_exported_as_geometry": not bool(shell_export),
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
