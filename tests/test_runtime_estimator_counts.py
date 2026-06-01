import numpy as np

from anystruct import optimize as opt
from anystruct.optimize_cylinder import CreateOptimizeCylinderWindow
from anystruct.optimize_multiple_window import CreateOptimizeMultipleWindow
from anystruct.optimize_window import CreateOptimizeWindow


def test_flat_runtime_step_counter_matches_optimizer_ranges():
    window = CreateOptimizeWindow.__new__(CreateOptimizeWindow)
    lower = np.array([0.6, 0.015, 0.3, 0.01, 0.1, 0.015, 3.5, 10.0])
    upper = np.array([0.8, 0.025, 0.5, 0.02, 0.2, 0.03, 3.5, 10.0])
    deltas = np.array([0.1, 0.005, 0.1, 0.005, 0.05, 0.005, 0.1, 1.0])

    estimated = 1
    for low, up, delta in zip(lower[:6], upper[:6], deltas[:6]):
        estimated *= window._count_steps(low, up, delta)

    assert estimated == len(opt.any_get_all_combs(lower, upper, deltas))


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
