import numpy as np

from anystruct import optimize as opt
from anystruct.optimize_cylinder import CreateOptimizeCylinderWindow
from anystruct.optimize_multiple_window import CreateOptimizeMultipleWindow
from anystruct.optimize_window import CreateOptimizeWindow


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_flat_runtime_step_counter_matches_optimizer_ranges():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    window._is_unstiffened_plate = False
    lower = np.array([0.6, 0.015, 0.3, 0.01, 0.1, 0.015, 3.5, 10.0])
    upper = np.array([0.8, 0.025, 0.5, 0.02, 0.2, 0.03, 3.5, 10.0])
    deltas = np.array([0.1, 0.005, 0.1, 0.005, 0.05, 0.005, 0.1, 1.0])

    estimated = 1
    for low, up, delta in zip(lower[:6], upper[:6], deltas[:6]):
        estimated *= window._count_steps(low, up, delta)

    assert estimated == len(opt.any_get_all_combs(lower, upper, deltas))


def test_unstiffened_flat_runtime_counter_uses_only_spacing_and_plate_thickness():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    window._is_unstiffened_plate = True
    window._new_algorithm = Var("anysmart")
    window._new_spacing_lower = Var(600.0)
    window._new_spacing_upper = Var(800.0)
    window._new_pl_thk_lower = Var(15.0)
    window._new_pl_thk_upper = Var(25.0)
    window._new_web_h_lower = Var(0.0)
    window._new_web_h_upper = Var(500.0)
    window._new_web_thk_lower = Var(0.0)
    window._new_web_thk_upper = Var(25.0)
    window._new_fl_w_lower = Var(0.0)
    window._new_fl_w_upper = Var(300.0)
    window._new_fl_thk_lower = Var(0.0)
    window._new_fl_thk_upper = Var(25.0)
    window._new_span = Var(3.5)
    window._new_width_lg = Var(10.0)
    window._new_delta_spacing = Var(100.0)
    window._new_delta_pl_thk = Var(5.0)
    window._new_delta_web_h = Var(100.0)
    window._new_delta_web_thk = Var(5.0)
    window._new_delta_fl_w = Var(50.0)
    window._new_delta_fl_thk = Var(5.0)
    window._new_use_weight_filter = Var(True)
    window.running_time_per_item = {"RP": 1.0}

    _, combinations = window.get_running_time()

    assert combinations == 9


def test_stiffened_girder_runtime_counter_includes_girder_bounds():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    window._is_unstiffened_plate = False
    window._has_girder = True
    window._predefined_stiffener_iter = None
    window._new_algorithm = Var("anysmart")
    window._new_spacing_lower = Var(600.0)
    window._new_spacing_upper = Var(700.0)
    window._new_pl_thk_lower = Var(15.0)
    window._new_pl_thk_upper = Var(15.0)
    window._new_web_h_lower = Var(300.0)
    window._new_web_h_upper = Var(300.0)
    window._new_web_thk_lower = Var(10.0)
    window._new_web_thk_upper = Var(10.0)
    window._new_fl_w_lower = Var(100.0)
    window._new_fl_w_upper = Var(100.0)
    window._new_fl_thk_lower = Var(15.0)
    window._new_fl_thk_upper = Var(15.0)
    window._new_girder_web_h_lower = Var(400.0)
    window._new_girder_web_h_upper = Var(500.0)
    window._new_girder_web_thk_lower = Var(12.0)
    window._new_girder_web_thk_upper = Var(12.0)
    window._new_girder_fl_w_lower = Var(120.0)
    window._new_girder_fl_w_upper = Var(120.0)
    window._new_girder_fl_thk_lower = Var(12.0)
    window._new_girder_fl_thk_upper = Var(12.0)
    window._new_span = Var(3.5)
    window._new_width_lg = Var(10.0)
    window._new_delta_spacing = Var(100.0)
    window._new_delta_pl_thk = Var(5.0)
    window._new_delta_web_h = Var(100.0)
    window._new_delta_web_thk = Var(5.0)
    window._new_delta_fl_w = Var(50.0)
    window._new_delta_fl_thk = Var(5.0)
    window._new_delta_girder_web_h = Var(100.0)
    window._new_delta_girder_web_thk = Var(5.0)
    window._new_delta_girder_fl_w = Var(50.0)
    window._new_delta_girder_fl_thk = Var(5.0)
    window._new_use_weight_filter = Var(True)
    window.running_time_per_item = {"RP": 1.0}

    _, combinations = window.get_running_time()

    assert combinations == 4


def test_stiffened_girder_default_runtime_counter_includes_full_girder_range():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    window._is_unstiffened_plate = False
    window._has_girder = True
    window._predefined_stiffener_iter = None
    window._new_algorithm = Var("anysmart")
    window._new_spacing_lower = Var(750.0)
    window._new_spacing_upper = Var(750.0)
    window._new_pl_thk_lower = Var(10.0)
    window._new_pl_thk_upper = Var(30.0)
    window._new_web_h_lower = Var(200.0)
    window._new_web_h_upper = Var(500.0)
    window._new_web_thk_lower = Var(10.0)
    window._new_web_thk_upper = Var(30.0)
    window._new_fl_w_lower = Var(100.0)
    window._new_fl_w_upper = Var(300.0)
    window._new_fl_thk_lower = Var(10.0)
    window._new_fl_thk_upper = Var(30.0)
    window._new_girder_web_h_lower = Var(500.0)
    window._new_girder_web_h_upper = Var(1000.0)
    window._new_girder_web_thk_lower = Var(10.0)
    window._new_girder_web_thk_upper = Var(30.0)
    window._new_girder_fl_w_lower = Var(100.0)
    window._new_girder_fl_w_upper = Var(300.0)
    window._new_girder_fl_thk_lower = Var(10.0)
    window._new_girder_fl_thk_upper = Var(30.0)
    window._new_span = Var(4.0)
    window._new_width_lg = Var(10.0)
    window._new_delta_spacing = Var(5.0)
    window._new_delta_pl_thk = Var(5.0)
    window._new_delta_web_h = Var(50.0)
    window._new_delta_web_thk = Var(5.0)
    window._new_delta_fl_w = Var(50.0)
    window._new_delta_fl_thk = Var(5.0)
    window._new_delta_girder_web_h = Var(100.0)
    window._new_delta_girder_web_thk = Var(5.0)
    window._new_delta_girder_fl_w = Var(50.0)
    window._new_delta_girder_fl_thk = Var(5.0)
    window._new_use_weight_filter = Var(True)
    window.running_time_per_item = {"RP": 1.0}

    _, combinations = window.get_running_time()

    panel_and_stiffener_combinations = 1 * 5 * 7 * 5 * 5 * 5
    girder_combinations = 6 * 5 * 5 * 5
    assert combinations == panel_and_stiffener_combinations * girder_combinations
    assert combinations > panel_and_stiffener_combinations


def test_unstiffened_flat_constraint_tuple_suppresses_stiffener_only_checks():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    window._is_unstiffened_plate = True
    window._new_check_sec_mod = Var(True)
    window._new_check_min_pl_thk = Var(True)
    window._new_check_shear_area = Var(True)
    window._new_check_buckling = Var(True)
    window._new_check_fatigue = Var(True)
    window._new_check_slamming = Var(True)
    window._new_check_local_buckling = Var(True)
    window._new_check_buckling_semi_analytical = Var(False)
    window._new_check_buckling_ml_numeric = Var(False)

    assert window._get_constraint_tuple() == (False, True, False, True, True, False, False, False, False, False)


def test_girder_constraint_tuple_uses_prescriptive_buckling_only():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    window._is_unstiffened_plate = False
    window._has_girder = True
    window._new_check_sec_mod = Var(True)
    window._new_check_min_pl_thk = Var(True)
    window._new_check_shear_area = Var(True)
    window._new_check_buckling = Var(True)
    window._new_check_fatigue = Var(False)
    window._new_check_slamming = Var(False)
    window._new_check_local_buckling = Var(True)
    window._new_check_buckling_semi_analytical = Var(True)
    window._new_check_buckling_ml_numeric = Var(True)

    assert window._get_constraint_tuple() == (True, True, True, True, False, False, True, False, False, False)


def test_multiple_runtime_step_counter_matches_optimizer_ranges():
    window = CreateOptimizeMultipleWindow.__new__(CreateOptimizeMultipleWindow)
    lower = np.array([0.6, 0.015, 0.3, 0.01, 0.1, 0.015, 3.5, 10.0])
    upper = np.array([0.8, 0.025, 0.5, 0.02, 0.2, 0.03, 3.5, 10.0])
    deltas = np.array([0.1, 0.005, 0.1, 0.005, 0.05, 0.005, 0.1, 1.0])

    estimated = 1
    for low, up, delta in zip(lower[:6], upper[:6], deltas[:6]):
        estimated *= window._count_steps(low, up, delta)

    assert estimated == len(opt.any_get_all_combs(lower, upper, deltas))


def test_cylinder_runtime_component_counter_matches_optimizer_ranges():
    window = CreateOptimizeCylinderWindow.__new__(CreateOptimizeCylinderWindow)
    window._map_type_idx = {0: True, 1: True, 2: True, 3: True}
    window._predefined_stiffener_iter = None

    lower = [0.02, 2.5, 5.0, 0.6, 4.0, 0.0, 0.0, 0.0]
    upper = [0.03, 2.5, 5.0, 0.8, 6.0, 0.0, 0.0, 0.0]
    deltas = [0.01, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0]

    estimated = window._count_cylinder_component_combinations(0, lower, upper, deltas)

    assert estimated == len(opt.any_get_all_combs(lower, upper, deltas))


def test_cylinder_runtime_counter_matches_inactive_component_rule():
    window = CreateOptimizeCylinderWindow.__new__(CreateOptimizeCylinderWindow)
    window._map_type_idx = {0: True, 1: False, 2: False, 3: False}
    window._predefined_stiffener_iter = None

    assert window._count_cylinder_component_combinations(
        1,
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ) == 1
