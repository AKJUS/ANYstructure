from pathlib import Path

import pytest

from anystruct import fe_plate_fields
from anystruct.sesam_import import (
    build_fe_model_from_sesam_fem,
    read_sesam_model,
    read_sesam_sif_stress,
    sesam_model_to_generated_geometry,
)
from anystruct.fe_solver_backend import assemble_load_vector


REF_CASES = Path(__file__).resolve().parents[1] / "ref_CasesCyl"

pytestmark = pytest.mark.skipif(not REF_CASES.exists(), reason="SESAM reference cases are not available")


@pytest.mark.parametrize(
    ("filename", "node_count", "shell_counts", "beam_counts"),
    [
        (
            "allQuadLinear_pressure_force_gravity_girder_stiffners.FEM",
            689,
            {"S4": 640},
            {"B2": 212},
        ),
        (
            "allTriLinear_pressure_force_gravity_girder_stiffners.FEM",
            755,
            {"S3": 1412},
            {"B2": 212},
        ),
        (
            "allQuad2ndorder_pressure_force_gravity_girder_stiffners.FEM",
            2021,
            {"S8": 640},
            {"B3": 212},
        ),
        (
            "allTri2ndorder_pressure_force_gravity_girder_stiffners.FEM",
            2925,
            {"S6": 1412},
            {"B3": 212},
        ),
        (
            "mixedLinear_pressure_force_gravity_girder_stiffners.FEM",
            675,
            {"S4": 626, "S3": 4},
            {"B2": 212},
        ),
        (
            "mixed2ndOrder_pressure_force_gravity_girder_stiffner.FEM",
            1983,
            {"S8": 626, "S6": 4},
            {"B3": 212},
        ),
    ],
)
def test_read_sesam_reference_fem_topologies(filename, node_count, shell_counts, beam_counts):
    model = read_sesam_model(REF_CASES / filename)

    assert len(model.nodes) == node_count
    assert _type_counts(model.shell_elements.values()) == shell_counts
    assert _type_counts(model.beam_elements.values()) == beam_counts
    assert model.shell_thicknesses
    assert model.beam_sections
    assert model.boundaries
    assert model.pressure_loads
    assert model.gravity == pytest.approx((0.0, 0.0, -9.80665016))


def test_sesam_generated_geometry_preserves_mixed_shell_and_beam_topology():
    model = read_sesam_model(REF_CASES / "mixedLinear_pressure_force_gravity_girder_stiffners.FEM")
    generated = sesam_model_to_generated_geometry(model)

    assert len(generated["nodes"]) == 675
    assert _node_count_histogram(generated["shells"]) == {4: 626, 3: 4}
    assert _node_count_histogram(generated["beams"]) == {2: 212}
    assert len(generated["supports"]) == len(model.boundaries)
    assert generated["materials"][0]["elastic_modulus"] == pytest.approx(210000003000.0)


def test_sesam_fem_import_builds_backend_model_and_load_vector():
    imported = build_fe_model_from_sesam_fem(
        REF_CASES / "mixedLinear_pressure_force_gravity_girder_stiffners.FEM"
    )

    assert imported.fe_model.mesh.num_nodes == 675
    assert imported.fe_model.mesh.num_elements == 842
    assert imported.load_case is not None
    assert len(imported.load_case.pressure_loads) == 630
    load_vector, load_info = assemble_load_vector(imported.fe_model, imported.load_case)
    assert load_vector.shape[0] == imported.fe_model.mesh.dof_manager.total_dofs
    assert load_info["load_norm"] > 0.0


def test_sesam_sif_stress_is_frd_like_global_tensor_result():
    stress = read_sesam_sif_stress(REF_CASES / "allTriLinear_pressure_force_gravity_girder_stiffners.SIF")

    assert stress.components == ("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX")
    assert len(stress.element_stress) == 1412
    assert len(stress.element_nodes) == 1412
    assert stress.nodal_stress
    assert max(abs(component) for values in stress.nodal_stress.values() for component in values) > 0.0


def test_sesam_sif_imports_directly_into_cylinder_buckling_interpreter():
    sif_path = REF_CASES / "allTriLinear_pressure_force_gravity_girder_stiffners.SIF"
    session = fe_plate_fields.create_fea_buckling_session(
        sif_path,
        geometry_type="cylinder",
        run_buckling=True,
    )

    assert session.inp_path == str(sif_path)
    assert session.frd_path == str(sif_path)
    assert session.frd_summary["format"] == "SESAM SIF/FEM"
    assert len(session.model.shell_elements) == 1412
    assert session.panels
    assert any(panel.stress is not None and panel.stress.sample_count > 0 for panel in session.panels)
    assert any(panel.usage_factor is not None for panel in session.panels)


def test_sesam_fem_auto_pairs_sibling_sif_for_interpreter_stresses():
    fem_path = REF_CASES / "allTriLinear_pressure_force_gravity_girder_stiffners.FEM"
    session = fe_plate_fields.create_fea_buckling_session(
        fem_path,
        geometry_type="cylinder",
        run_buckling=False,
    )

    assert session.inp_path == str(fem_path)
    assert session.frd_path == str(fem_path.with_suffix(".SIF"))
    assert any(panel.stress is not None and panel.stress.sample_count > 0 for panel in session.panels)


def _type_counts(elements):
    counts = {}
    for element in elements:
        counts[element.element_type] = counts.get(element.element_type, 0) + 1
    return counts


def _node_count_histogram(items):
    counts = {}
    for item in items:
        count = len(item["node_ids"])
        counts[count] = counts.get(count, 0) + 1
    return counts
