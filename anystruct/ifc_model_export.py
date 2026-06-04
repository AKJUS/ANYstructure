# -*- coding: utf-8 -*-
"""Proper IFC solid model export for ANYstructure.

This module intentionally does not export the Matplotlib preview mesh.  It rebuilds
plate, stiffener, girder, shell, longitudinal stiffener and ring stiffener objects
as IFC swept solids from the active ANYstructure line data.

Install dependency:
    pip install ifcopenshell

Typical use from Application:
    ifc_model_export.export_selected_structure_from_application(self, filename)
"""

from __future__ import annotations

import math
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
    length = _pos_float(length, 0.0)
    spacing = _pos_float(spacing, 0.0)
    if length <= EPS:
        return [0.0]
    if spacing <= EPS:
        return [0.0, length] if include_ends else [length / 2.0]
    positions = [0.0] if include_ends else []
    next_pos = spacing
    count_guard = 0
    while next_pos < length - EPS and count_guard < max_count:
        positions.append(float(next_pos))
        next_pos += spacing
        count_guard += 1
    if include_ends and abs(positions[-1] - length) > EPS:
        positions.append(float(length))
    if not positions:
        return [length / 2.0]
    return positions


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
    rep = ctx.model.createIfcShapeRepresentation(
        ctx.body_context,
        "Body",
        "SweptSolid",
        [solid],
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [rep])


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
                   z0: float, z1: float, extra_properties: dict[str, Any] | None = None) -> Any:
    return _add_box_element(ctx, "IfcPlate", name, x0, x1, y0, y1, z0, z1,
                            predefined_type="SHEET", extra_properties=extra_properties)


def _add_member_web_and_flange(ctx: IfcContext, base_name: str, orientation: str, x_center: float, y_center: float,
                               length: float, plate_thk: float, dims: SectionDimensions,
                               x_limits: tuple[float, float] | None = None,
                               y_limits: tuple[float, float] | None = None,
                               side_sign: float = 1.0,
                               member_role: str = "stiffener") -> None:
    """Add web and flange as separate proper swept-solid IfcMember objects."""
    web_h = max(dims.web_h, 0.0)
    web_t = max(dims.web_thk, 0.0)
    fl_w = max(dims.flange_w, 0.0)
    fl_t = max(dims.flange_thk, 0.0)
    sec_type = str(dims.type or "T")
    if length <= EPS or (web_h <= EPS and fl_t <= EPS):
        return

    if side_sign >= 0.0:
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
        if web_h > EPS and web_t > EPS:
            _add_box_element(
                ctx, "IfcMember", base_name + " Web", x0, x1,
                y_center - web_t / 2.0, y_center + web_t / 2.0,
                web_z[0], web_z[1], predefined_type="STUD",
                extra_properties={"role": member_role, "section_type": sec_type},
            )
        if fl_w > EPS and fl_t > EPS:
            if sec_type in ["L", "L-bulb"]:
                y0 = y_center - web_t / 2.0
                y1 = y0 + fl_w
            else:
                y0 = y_center - fl_w / 2.0
                y1 = y_center + fl_w / 2.0
            _add_box_element(
                ctx, "IfcMember", base_name + " Flange", x0, x1,
                y0, y1, flange_z[0], flange_z[1], predefined_type="STUD",
                extra_properties={"role": member_role, "section_type": sec_type},
            )
    else:
        y0 = y_center - length / 2.0
        y1 = y_center + length / 2.0
        if y_limits is not None:
            y0 = max(y0, y_limits[0])
            y1 = min(y1, y_limits[1])
        if y1 <= y0:
            return
        if web_h > EPS and web_t > EPS:
            _add_box_element(
                ctx, "IfcMember", base_name + " Web",
                x_center - web_t / 2.0, x_center + web_t / 2.0,
                y0, y1, web_z[0], web_z[1], predefined_type="STUD",
                extra_properties={"role": member_role, "section_type": sec_type},
            )
        if fl_w > EPS and fl_t > EPS:
            if sec_type in ["L", "L-bulb"]:
                x0 = x_center - web_t / 2.0
                x1 = x0 + fl_w
            else:
                x0 = x_center - fl_w / 2.0
                x1 = x_center + fl_w / 2.0
            _add_box_element(
                ctx, "IfcMember", base_name + " Flange",
                x0, x1, y0, y1, flange_z[0], flange_z[1], predefined_type="STUD",
                extra_properties={"role": member_role, "section_type": sec_type},
            )


def _add_flat_structure(ctx: IfcContext, app: Any, all_obj: Any, active_line: str, side_sign: float) -> None:
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
        girder_gap = max(gdims.web_thk, 0.0)
        _add_plate_box(
            ctx, f"{active_line} Plate field", 0.0, width, 0.0, length, 0.0, plate_thk,
            extra_properties={"active_line": active_line, "panel_type": "flat panel with girder"},
        )
        _add_member_web_and_flange(
            ctx, f"{active_line} Girder", "y", x_mid, length / 2.0, length,
            plate_thk, gdims, y_limits=(0.0, length), side_sign=side_sign, member_role="girder",
        )
        if sdims is not None:
            stiffener_ys = _positions_from_length_and_spacing(length, spacing, include_ends=True, max_count=80)
            left_x0, left_x1 = 0.0, max(x_mid - girder_gap / 2.0, 0.0)
            right_x0, right_x1 = min(x_mid + girder_gap / 2.0, width), width
            for index, y in enumerate(stiffener_ys, start=1):
                if left_x1 > left_x0:
                    _add_member_web_and_flange(
                        ctx, f"{active_line} Stiffener {index:03d} Left", "x",
                        (left_x0 + left_x1) / 2.0, y, left_x1 - left_x0, plate_thk,
                        sdims, x_limits=(left_x0, left_x1), side_sign=side_sign,
                        member_role="stiffener",
                    )
                if right_x1 > right_x0:
                    _add_member_web_and_flange(
                        ctx, f"{active_line} Stiffener {index:03d} Right", "x",
                        (right_x0 + right_x1) / 2.0, y, right_x1 - right_x0, plate_thk,
                        sdims, x_limits=(right_x0, right_x1), side_sign=side_sign,
                        member_role="stiffener",
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
        )
        if stiffener is not None:
            sdims = _section_dimensions_from_app(app, stiffener)
            stiffener_ys = _positions_from_length_and_spacing(length, spacing, include_ends=True, max_count=80)
            for index, y in enumerate(stiffener_ys, start=1):
                _add_member_web_and_flange(
                    ctx, f"{active_line} Stiffener {index:03d}", "x", width / 2.0, y,
                    width, plate_thk, sdims, x_limits=(0.0, width), side_sign=side_sign,
                    member_role="stiffener",
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


def _add_circular_hollow_element(ctx: IfcContext, ifc_class: str, name: str, outer_radius: float,
                                 wall_thickness: float, z0: float, depth: float,
                                 predefined_type: str | None = None,
                                 extra_properties: dict[str, Any] | None = None) -> None:
    solid = _create_circle_hollow_swept_solid(ctx, outer_radius, wall_thickness, depth)
    shape = _product_shape_from_solid(ctx, solid)
    placement = _local_placement(ctx.model, (0.0, 0.0, z0))
    element = _create_building_element(ctx, ifc_class, name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "outer_radius_m": float(outer_radius),
        "wall_thickness_m": float(wall_thickness),
        "z0_m": float(z0),
        "depth_m": float(depth),
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)


def _add_cylinder_longitudinal_members(ctx: IfcContext, active_line: str, radius: float, length: float,
                                       angles: Iterable[float], dims: SectionDimensions,
                                       shell_thk: float, side_sign: float) -> None:
    radial_extension_sign = 1.0 if side_sign >= 0.0 else -1.0
    base_radius = radius + radial_extension_sign * (shell_thk if side_sign < 0.0 else 0.0)
    for index, angle in enumerate(angles, start=1):
        radial = (math.cos(angle), math.sin(angle), 0.0)
        tangential = (-math.sin(angle), math.cos(angle), 0.0)
        if dims.web_h > EPS and dims.web_thk > EPS:
            center_radius = base_radius + radial_extension_sign * dims.web_h / 2.0
            center = (center_radius * math.cos(angle), center_radius * math.sin(angle), length / 2.0)
            _add_oriented_box_element(
                ctx, "IfcMember", f"{active_line} Longitudinal {index:03d} Web",
                center=center,
                local_x=tangential,
                local_z=(0.0, 0.0, 1.0),
                xdim=dims.web_thk,
                ydim=dims.web_h,
                depth=length,
                predefined_type="STUD",
                extra_properties={"role": "longitudinal stiffener", "angle_rad": float(angle)},
            )
        if dims.flange_w > EPS and dims.flange_thk > EPS:
            center_radius = base_radius + radial_extension_sign * (dims.web_h + dims.flange_thk / 2.0)
            center = (center_radius * math.cos(angle), center_radius * math.sin(angle), length / 2.0)
            _add_oriented_box_element(
                ctx, "IfcMember", f"{active_line} Longitudinal {index:03d} Flange",
                center=center,
                local_x=tangential,
                local_z=(0.0, 0.0, 1.0),
                xdim=dims.flange_w,
                ydim=dims.flange_thk,
                depth=length,
                predefined_type="STUD",
                extra_properties={"role": "longitudinal stiffener", "angle_rad": float(angle)},
            )


def _add_ring_set(ctx: IfcContext, active_line: str, role: str, radius: float, positions: Iterable[float],
                  dims: SectionDimensions, side_sign: float) -> None:
    radial_sign = 1.0 if side_sign >= 0.0 else -1.0
    for index, z_pos in enumerate(positions, start=1):
        outer_radius = radius + radial_sign * dims.web_h if radial_sign > 0.0 else radius
        if dims.web_h > EPS and dims.web_thk > EPS:
            _add_circular_hollow_element(
                ctx, "IfcMember", f"{active_line} {role} {index:03d} Web",
                outer_radius=max(outer_radius, EPS), wall_thickness=max(dims.web_h, EPS),
                z0=z_pos - dims.web_thk / 2.0, depth=dims.web_thk,
                predefined_type="RING",
                extra_properties={"role": role, "z_position_m": float(z_pos)},
            )
        if dims.flange_w > EPS and dims.flange_thk > EPS:
            flange_outer_radius = radius + radial_sign * (dims.web_h + dims.flange_thk) if radial_sign > 0.0 else radius
            _add_circular_hollow_element(
                ctx, "IfcMember", f"{active_line} {role} {index:03d} Flange",
                outer_radius=max(flange_outer_radius, EPS), wall_thickness=max(dims.flange_thk, EPS),
                z0=z_pos - dims.flange_w / 2.0, depth=dims.flange_w,
                predefined_type="RING",
                extra_properties={"role": role, "z_position_m": float(z_pos)},
            )


def _add_cylinder_structure(ctx: IfcContext, app: Any, cyl_obj: Any, active_line: str, side_sign: float) -> None:
    shell = getattr(cyl_obj, "ShellObj", None)
    if shell is None:
        raise ValueError("The selected cylinder line has no ShellObj to export.")
    radius = max(float(getattr(shell, "radius")), EPS)
    length = max(float(getattr(shell, "length_of_shell")), EPS)
    thk = max(float(getattr(shell, "thk")), EPS)
    is_panel = _is_cylinder_panel(app, cyl_obj)
    theta_start, theta_end = _cylinder_theta_range(app, cyl_obj)

    if is_panel:
        # IFC has no universally robust partial hollow circle profile.  For panel exports,
        # use a rectangular curved-panel-equivalent plate field, but still export all stiffeners
        # as proper swept solids.  This is model geometry, not the preview mesh.
        arc_length = abs(theta_end - theta_start) * radius
        _add_plate_box(
            ctx, f"{active_line} Curved panel equivalent plate", -arc_length / 2.0, arc_length / 2.0,
            0.0, length, 0.0, thk,
            extra_properties={
                "active_line": active_line,
                "panel_type": "cylinder panel equivalent",
                "radius_m": radius,
                "theta_start_rad": theta_start,
                "theta_end_rad": theta_end,
            },
        )
        ctx.summary.warnings.append(
            "Cylinder panel shell exported as a curved-panel-equivalent plate solid because IFC partial hollow "
            "circle profiles are not handled robustly by all viewers. Full cylinders are exported as hollow "
            "circular swept solids."
        )
    else:
        _add_circular_hollow_element(
            ctx, "IfcPlate", f"{active_line} Cylindrical shell",
            outer_radius=radius, wall_thickness=thk, z0=0.0, depth=length,
            predefined_type="SHEET",
            extra_properties={"active_line": active_line, "panel_type": "full cylindrical shell"},
        )

    if getattr(cyl_obj, "LongStfObj", None) is not None:
        long_dims = _section_dimensions_from_app(app, cyl_obj.LongStfObj)
        spacing = max(long_dims.spacing, EPS)
        if is_panel:
            arc_length = abs(theta_end - theta_start) * radius
            num_stf = max(2, min(72, int(round(arc_length / spacing)) + 1))
            if num_stf == 1:
                angles = [(theta_start + theta_end) / 2.0]
            else:
                angles = [theta_start + (theta_end - theta_start) * i / (num_stf - 1) for i in range(num_stf)]
        else:
            num_stf = max(4, min(144, int(round(2.0 * math.pi * radius / spacing))))
            angles = [2.0 * math.pi * idx / num_stf for idx in range(num_stf)]
        _add_cylinder_longitudinal_members(ctx, active_line, radius, length, angles, long_dims, thk, side_sign)

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
        _add_ring_set(ctx, active_line, "Ring stiffener", radius, ring_positions, ring_dims, side_sign)

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
        _add_ring_set(ctx, active_line, "Ring frame", radius, frame_positions, frame_dims, side_sign)


def export_selected_structure_from_application(app: Any, filename: str) -> ExportSummary:
    """Export the active ANYstructure line as a proper IFC model.

    The exporter reads the selected line's real structural objects.  It does not use
    _prop_3d_export_mesh, _prop_3d_shell_export_mesh, STL, meshio, numpy-stl, or any
    Matplotlib preview geometry.
    """
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
    ctx = _create_basic_context(filename, project_name, material_name=material_name)

    side_sign = -1.0 if bool(getattr(app, "_new_prop_3d_opposite_side", False).get()) else 1.0
    line_data = app._line_to_struc[active_line]

    try:
        cylinder_obj = line_data[5]
    except Exception:
        cylinder_obj = None

    if cylinder_obj is not None:
        _add_cylinder_structure(ctx, app, cylinder_obj, active_line, side_sign)
    else:
        try:
            all_obj = line_data[0]
        except Exception as error:
            raise ValueError("Could not read flat panel structure object from selected line.") from error
        _add_flat_structure(ctx, app, all_obj, active_line, side_sign)

    _add_property_set(ctx, ctx.project, "ANYstructureExport", {
        "active_line": active_line,
        "source": "ANYstructure",
        "geometry_source": "model parameters; not preview mesh",
        "material_yield_MPa": material_yield,
        "opposite_side": side_sign < 0.0,
    })

    ctx.model.write(filename)
    return ctx.summary
