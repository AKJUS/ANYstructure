#!/usr/bin/env python3
"""Patch ANYstructure IFC ring export to use continuous annular geometry."""
from __future__ import annotations

import argparse
from pathlib import Path

HELPERS = '''

def _is_full_circle_range(theta_start: float, theta_end: float, tolerance: float = 1.0e-7) -> bool:
    """Return True when the angular range represents one complete circumference."""
    return abs(abs(float(theta_end) - float(theta_start)) - 2.0 * math.pi) <= tolerance


def _product_shape_from_advanced_annular_face(
    ctx: IfcContext,
    inner_radius: float,
    outer_radius: float,
    z: float,
) -> Any:
    """Create one zero-thickness analytic annular plate."""
    inner_radius = max(float(inner_radius), EPS)
    outer_radius = max(float(outer_radius), inner_radius + EPS)
    z = float(z)

    def circle_loop(radius: float, reverse: bool) -> Any:
        p0 = _cyl_point(radius, 0.0, z)
        p1 = _cyl_point(radius, math.pi, z)
        v0 = _vertex_point(ctx, p0)
        v1 = _vertex_point(ctx, p1)
        first_half = _edge_curve(
            ctx, v0, v1,
            _trimmed_circle_curve_3d(ctx, radius, z, 0.0, math.pi),
            True,
        )
        second_half = _edge_curve(
            ctx, v1, v0,
            _trimmed_circle_curve_3d(ctx, radius, z, math.pi, 2.0 * math.pi),
            True,
        )
        if reverse:
            edges = [
                _oriented_edge(ctx, second_half, False),
                _oriented_edge(ctx, first_half, False),
            ]
        else:
            edges = [
                _oriented_edge(ctx, first_half, True),
                _oriented_edge(ctx, second_half, True),
            ]
        return ctx.model.createIfcEdgeLoop(edges)

    outer_loop = circle_loop(outer_radius, reverse=False)
    inner_loop = circle_loop(inner_radius, reverse=True)
    outer_bound = ctx.model.createIfcFaceOuterBound(outer_loop, True)
    inner_bound = ctx.model.createIfcFaceBound(inner_loop, True)
    plane = ctx.model.createIfcPlane(
        _axis2_placement_3d(
            ctx.model,
            location=(0.0, 0.0, z),
            axis=(0.0, 0.0, 1.0),
            ref_direction=(1.0, 0.0, 0.0),
            length_scale=ctx.length_scale,
        )
    )
    try:
        face = ctx.model.createIfcAdvancedFace([outer_bound, inner_bound], plane, True)
    except Exception:
        face = ctx.model.createIfcFaceSurface([outer_bound, inner_bound], plane, True)
    shell = ctx.model.createIfcOpenShell([face])
    surface_model = ctx.model.createIfcShellBasedSurfaceModel([shell])
    representation = ctx.model.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SurfaceModel", [surface_model]
    )
    return ctx.model.createIfcProductDefinitionShape(None, None, [representation])


def _add_full_annular_ring_web_surface(
    ctx: IfcContext,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z: float,
    predefined_type: str | None = "MEMBER",
    extra_properties: dict[str, Any] | None = None,
) -> Any:
    """Add a full ring web as one analytic annular shell face."""
    shape = _product_shape_from_advanced_annular_face(ctx, inner_radius, outer_radius, z)
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, "IfcMember", name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "model_type": "analytic_annular_surface",
        "shell_export": True,
        "thickness_exported_as_geometry": False,
        "continuous_shell_interface": True,
        "inner_radius_m": float(inner_radius),
        "outer_radius_m": float(outer_radius),
        "z_position_m": float(z),
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    return element


def _add_full_annular_ring_web_solid(
    ctx: IfcContext,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z0: float,
    z1: float,
    predefined_type: str | None = "MEMBER",
    extra_properties: dict[str, Any] | None = None,
) -> Any:
    """Add a full ring web as one exact annular extrusion."""
    inner_radius = max(float(inner_radius), EPS)
    outer_radius = max(float(outer_radius), inner_radius + EPS)
    wall_thickness = outer_radius - inner_radius
    solid = _create_positioned_circle_hollow_swept_solid(
        ctx,
        outer_radius=outer_radius,
        wall_thickness=wall_thickness,
        z0=z0,
        z1=z1,
    )
    shape = _product_shape_from_solid(ctx, solid)
    placement = _local_placement(ctx.model)
    element = _create_building_element(ctx, "IfcMember", name, placement, shape, predefined_type)
    _assign_to_storey(ctx, element)
    _assign_material(ctx, element)
    props = {
        "model_type": "analytic_annular_swept_solid",
        "shell_export": False,
        "thickness_exported_as_geometry": True,
        "continuous_shell_interface": True,
        "inner_radius_m": inner_radius,
        "outer_radius_m": outer_radius,
        "axial_thickness_m": abs(float(z1) - float(z0)),
    }
    if extra_properties:
        props.update(extra_properties)
    _add_property_set(ctx, element, "ANYstructureDimensions", props)
    ctx.summary.elements.append(name)
    _track_solid_operand(ctx, element, solid)
    return element
'''

OLD_BLOCK = '''            if shell_export:
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
                )'''

NEW_BLOCK = '''            full_circle = _is_full_circle_range(theta_start, theta_end)
            web_name = f"{active_line} {role} {index:03d} Web"
            if shell_export:
                if full_circle:
                    _add_full_annular_ring_web_surface(
                        ctx,
                        web_name,
                        inner_radius=r0,
                        outer_radius=r1,
                        z=z_pos,
                        predefined_type="MEMBER",
                        extra_properties=web_props,
                    )
                else:
                    _add_surface_element(
                        ctx,
                        "IfcMember",
                        web_name,
                        _ring_web_shell_faces(r0, r1, z_pos, theta_start, theta_end),
                        predefined_type="MEMBER",
                        extra_properties=web_props,
                    )
            else:
                z0 = z_pos - max(dims.web_thk, EPS) / 2.0
                z1 = z_pos + max(dims.web_thk, EPS) / 2.0
                if full_circle:
                    _add_full_annular_ring_web_solid(
                        ctx,
                        web_name,
                        inner_radius=r0,
                        outer_radius=r1,
                        z0=z0,
                        z1=z1,
                        predefined_type="MEMBER",
                        extra_properties=web_props,
                    )
                else:
                    web_props.update({
                        "model_type": "faceted_brep_solid",
                        "surface_segments": int(segments),
                        "thickness_exported_as_geometry": True,
                        "continuous_shell_interface": False,
                    })
                    _add_faceted_solid_element(
                        ctx,
                        "IfcMember",
                        web_name,
                        _cylindrical_wall_solid_faces(
                            r0, r1, z0, z1, theta_start, theta_end, segments
                        ),
                        predefined_type="MEMBER",
                        extra_properties=web_props,
                    )'''


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "_add_full_annular_ring_web_solid" not in text:
        marker = "\ndef _add_ring_set("
        if marker not in text:
            raise RuntimeError("Could not locate _add_ring_set() insertion point.")
        text = text.replace(marker, HELPERS + marker, 1)
    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    elif NEW_BLOCK not in text:
        raise RuntimeError("Could not locate the expected ring web branch.")
    path.write_text(text, encoding="utf-8")
    print(f"Updated: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    patch_file(args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
