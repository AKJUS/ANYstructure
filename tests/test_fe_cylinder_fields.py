"""Geometry tests for cylindrical FE field extraction."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from anystruct.fe_plate_fields import (
    CylinderStress,
    FeShellModel,
    ShellElement,
    ShellSection,
    anystructure_input_for_cylinder_field,
    calculate_cylinder_buckling,
    create_fea_buckling_session,
    detect_cylinder_geometry,
    infer_cylinder_fields,
)


CYLINDER_SAMPLE_INP = Path(r"C:\PrePoMax v2.5.1 dev\Temp\Analysis-cylinder-simple.inp")
CYLINDER_SAMPLE_FRD = Path(r"C:\PrePoMax v2.5.1 dev\Temp\Analysis-cylinder-simple.frd")


def _orthogonally_stiffened_cylinder() -> FeShellModel:
    nodes = {}
    elements = {}
    node_id = 1
    element_id = 1

    radius = 2.0
    length = 6.0
    circumferential_divisions = 64
    axial_divisions = 6

    skin_grid = {}
    for axial_index in range(axial_divisions + 1):
        z = length * axial_index / axial_divisions
        for angular_index in range(circumferential_divisions):
            angle = 2.0 * math.pi * angular_index / circumferential_divisions
            skin_grid[(axial_index, angular_index)] = node_id
            nodes[node_id] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
                z,
            )
            node_id += 1

    for axial_index in range(axial_divisions):
        for angular_index in range(circumferential_divisions):
            next_angle = (angular_index + 1) % circumferential_divisions
            elements[element_id] = ShellElement(
                element_id=element_id,
                node_ids=(
                    skin_grid[(axial_index, angular_index)],
                    skin_grid[(axial_index, next_angle)],
                    skin_grid[(axial_index + 1, next_angle)],
                    skin_grid[(axial_index + 1, angular_index)],
                ),
                element_type="S4",
                elset="SKIN",
            )
            element_id += 1

    # Four longitudinal flat-bar webs.
    for angular_index in (0, 16, 32, 48):
        angle = 2.0 * math.pi * angular_index / circumferential_divisions
        for axial_index in range(axial_divisions):
            z0 = length * axial_index / axial_divisions
            z1 = length * (axial_index + 1) / axial_divisions
            web_nodes = []
            for local_radius, z in (
                (radius, z0),
                (radius - 0.2, z0),
                (radius - 0.2, z1),
                (radius, z1),
            ):
                nodes[node_id] = (
                    local_radius * math.cos(angle),
                    local_radius * math.sin(angle),
                    z,
                )
                web_nodes.append(node_id)
                node_id += 1
            elements[element_id] = ShellElement(
                element_id=element_id,
                node_ids=tuple(web_nodes),
                element_type="S4",
                elset="LONGITUDINALS",
            )
            element_id += 1

    # Two ring webs.
    for z in (2.0, 4.0):
        for angular_index in range(circumferential_divisions):
            angle0 = 2.0 * math.pi * angular_index / circumferential_divisions
            angle1 = 2.0 * math.pi * ((angular_index + 1) % circumferential_divisions) / circumferential_divisions
            ring_nodes = []
            for local_radius, angle in (
                (radius, angle0),
                (radius - 0.3, angle0),
                (radius - 0.3, angle1),
                (radius, angle1),
            ):
                nodes[node_id] = (
                    local_radius * math.cos(angle),
                    local_radius * math.sin(angle),
                    z,
                )
                ring_nodes.append(node_id)
                node_id += 1
            elements[element_id] = ShellElement(
                element_id=element_id,
                node_ids=tuple(ring_nodes),
                element_type="S4",
                elset="RINGS",
            )
            element_id += 1

    elsets = {
        "ALL": tuple(sorted(elements)),
        "SKIN": tuple(element_id for element_id, element in elements.items() if element.elset == "SKIN"),
        "LONGITUDINALS": tuple(
            element_id for element_id, element in elements.items() if element.elset == "LONGITUDINALS"
        ),
        "RINGS": tuple(element_id for element_id, element in elements.items() if element.elset == "RINGS"),
    }
    return FeShellModel(
        nodes=nodes,
        shell_elements=elements,
        elsets=elsets,
        shell_sections=(ShellSection(elset="ALL", material="S355", thickness_m=0.01),),
    )


def test_detects_cylinder_axis_and_radius() -> None:
    model = _orthogonally_stiffened_cylinder()
    geometry = detect_cylinder_geometry(model)

    assert geometry.radius_m == pytest.approx(2.0, abs=1.0e-8)
    assert abs(geometry.axis_direction[2]) == pytest.approx(1.0, abs=1.0e-8)
    assert len(geometry.skin_element_ids) == 64 * 6


def test_extracts_orthogonal_cylinder_bays() -> None:
    model = _orthogonally_stiffened_cylinder()
    geometry = detect_cylinder_geometry(model)
    fields = infer_cylinder_fields(model, geometry)

    # 4 angular bays x 3 axial bays.
    assert len(fields) == 12
    assert all(field.element_ids for field in fields)
    assert all(field.radius_m == pytest.approx(2.0) for field in fields)

    members = {
        member.member_id: member
        for field in fields
        for member in field.members
    }
    roles = [member.role for member in members.values()]
    assert roles.count("longitudinal_stiffener") == 4
    assert roles.count("ring_stiffener") == 2


def test_orthogonal_cylinder_uses_ring_stiffener_as_frame_for_calculation() -> None:
    model = _orthogonally_stiffened_cylinder()
    geometry = detect_cylinder_geometry(model)
    fields = infer_cylinder_fields(model, geometry)
    field = fields[0]
    stress = CylinderStress(
        field_id=field.field_id,
        axial_stress_mpa=-0.2,
        hoop_stress_mpa=-20.0,
        torsional_shear_mpa=0.1,
        transverse_shear_mpa=0.0,
        sample_count=1,
        reduction="test",
    )

    input_data = anystructure_input_for_cylinder_field(field, stress)
    results = calculate_cylinder_buckling([field], [stress])

    assert input_data["calculation_domain"] == "Orthogonally Stiffened shell"
    assert input_data["ring_stiffener"] is None
    assert input_data["ring_frame"]["source_member_id"].startswith("ring_stiffener_")
    assert results[0]["available"] is True
    assert results[0].get("error") is None
    assert results[0]["result"]["Heavy ring frame"] is not None


@pytest.mark.skipif(
    not (CYLINDER_SAMPLE_INP.exists() and CYLINDER_SAMPLE_FRD.exists()),
    reason="Provided PrePoMax cylinder sample is not available",
)
def test_provided_prepomax_cylinder_import_calculates_panel_ufs() -> None:
    session = create_fea_buckling_session(CYLINDER_SAMPLE_INP, CYLINDER_SAMPLE_FRD, run_buckling=True)

    assert session.summary()["geometry_type"] == "cylinder"
    assert session.panel_count == 84
    assert session.field_count == 84
    assert session.geometry.radius_m == pytest.approx(5.0, abs=5.0e-5)
    assert session.geometry.skin_thickness_m == pytest.approx(0.015)
    assert len(session.usage_factors()) == session.panel_count
    assert all(panel.buckling_result["available"] is True for panel in session.panels)
    assert session.panels[0].usage_factor == pytest.approx(0.511746358910791)
