from copy import deepcopy

from anystruct import optimize as opt, example_data as ex, calc_structure as calc
import numpy as np
import pytest

# Testing the Structure class

def _with_panel_default(obj_dict):
    obj_dict = deepcopy(obj_dict)
    obj_dict.setdefault('panel or shell', ['panel', ''])
    return obj_dict

@pytest.fixture
def opt_input():
    obj_dict = _with_panel_default(ex.obj_dict)
    fat_obj = ex.get_fatigue_object()
    fp = ex.get_fatigue_pressures()
    fat_press = ((fp['p_ext']['loaded'],fp['p_ext']['ballast'],fp['p_ext']['part']),
                 (fp['p_int']['loaded'],fp['p_int']['ballast'],fp['p_int']['part']))
    x0 = [obj_dict['spacing'][0], obj_dict['plate_thk'][0], obj_dict['stf_web_height'][0], obj_dict['stf_web_thk'][0],
          obj_dict['stf_flange_width'][0], obj_dict['stf_flange_thk'][0], obj_dict['span'][0], 10]
    scantlings = calc.CalcScantlings(obj_dict)
    obj = calc.AllStructure(Plate=scantlings, Stiffener=scantlings, Girder=None, main_dict=ex.prescriptive_main_dict)
    lat_press = 271.124
    upper_bounds = np.array([0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10])
    lower_bounds = np.array([0.8, 0.02, 0.5, 0.02, 0.22, 0.03, 3.5, 10])
    deltas = np.array([0.05, 0.005, 0.05, 0.005, 0.05, 0.005])
    return obj, upper_bounds, lower_bounds, lat_press, deltas, fat_obj, fat_press, x0

def test_optimization(opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, fat_obj, fat_press, x0 = opt_input
    results = opt.run_optmizataion(obj, upper_bounds, lower_bounds, lat_press, deltas, algorithm='anysmart',
                                   fatigue_obj=fat_obj, fat_press_ext_int=fat_press)

    assert len(results) == 5
    assert results[3] in (True, False)

def test_weight_calc(opt_input):
    assert opt.calc_weight(opt_input[-1]) == pytest.approx(8243.530164606)


def test_weld_bias_filter_does_not_use_panel_weight(opt_input):
    obj, _, _, lat_press, _, _, _, _ = opt_input
    x = [0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10]
    weld_limit = opt.calc_weld_consumable(x) * 1.01

    result = opt.any_constraints_all(
        x=x,
        obj=obj,
        lat_press=lat_press,
        init_weight=weld_limit,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
    )

    assert result[0] is True
    assert opt.calc_weight(x) > weld_limit


def test_weld_bias_filter_rejects_by_weld_consumables(opt_input):
    obj, _, _, lat_press, _, _, _, _ = opt_input
    x = [0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10]
    weld_limit = opt.calc_weld_consumable(x) * 0.99

    result = opt.any_constraints_all(
        x=x,
        obj=obj,
        lat_press=lat_press,
        init_weight=weld_limit,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
    )

    assert result[0] is False
    assert result[1] == 'Weld filter'


def test_weld_length_filter_does_not_use_weld_consumables(monkeypatch, opt_input):
    obj, _, _, lat_press, _, _, _, _ = opt_input
    x = [0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10]

    monkeypatch.setattr(opt, 'calc_weld_length', lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable', lambda *args, **kwargs: 1000.0)

    result = opt.any_constraints_all(
        x=x,
        obj=obj,
        lat_press=lat_press,
        init_weight=11.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
        weld_metric='weld_length',
    )

    assert result[0] is True


def test_weld_length_filter_rejects_by_length(monkeypatch, opt_input):
    obj, _, _, lat_press, _, _, _, _ = opt_input
    x = [0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10]

    monkeypatch.setattr(opt, 'calc_weld_length', lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable', lambda *args, **kwargs: 1.0)

    result = opt.any_constraints_all(
        x=x,
        obj=obj,
        lat_press=lat_press,
        init_weight=9.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
        weld_metric='Weld length',
    )

    assert result[0] is False
    assert result[1] == 'Weld filter'


def test_cost_filter_uses_weight_and_selected_weld_metric(monkeypatch, opt_input):
    obj, _, _, lat_press, _, _, _, _ = opt_input
    x = [0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10]

    monkeypatch.setattr(opt, 'calc_weight', lambda candidate: 100.0)
    monkeypatch.setattr(opt, 'calc_weld_length', lambda *args, **kwargs: 20.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable', lambda *args, **kwargs: 2000.0)

    result = opt.any_constraints_all(
        x=x,
        obj=obj,
        lat_press=lat_press,
        init_weight=139.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_metric='Weld length',
        cost_factors={'steel': 1.0, 'weld': 2.0},
    )

    assert result[0] is False
    assert result[1] == 'Cost filter'


def test_cost_objective_calculates_flat_panel_cost(monkeypatch):
    monkeypatch.setattr(opt, 'calc_weight', lambda candidate: 100.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable', lambda *args, **kwargs: 7.0)

    assert opt.calc_flat_objective_value(
        x='candidate',
        cost_factors=(2.0, 3.0),
    ) == pytest.approx(221.0)


def test_cylinder_weld_bias_filter_does_not_use_panel_weight(monkeypatch):
    class FakeCylinder:
        RingStfObj = None
        RingFrameObj = None

    monkeypatch.setattr(opt, 'create_new_cylinder_obj', lambda obj, x: FakeCylinder())
    monkeypatch.setattr(opt, 'calc_weight_cylinder', lambda x: 1000.0)
    monkeypatch.setattr(
        opt,
        'calc_weld_consumable_cylinder',
        lambda x, include_web_to_flange=False: 10.0,
    )

    result = opt.any_constraints_cylinder(
        x='candidate',
        obj=object(),
        init_weight=11.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
    )

    assert result is None


def test_cylinder_weld_bias_filter_rejects_by_weld_consumables(monkeypatch):
    class FakeCylinder:
        RingStfObj = None
        RingFrameObj = None

        def get_utilization_factors(self, optimizing=False, empty_result_dict=False):
            return {}

    monkeypatch.setattr(opt, 'create_new_cylinder_obj', lambda obj, x: FakeCylinder())
    monkeypatch.setattr(opt, 'calc_weight_cylinder', lambda x: 1000.0)
    monkeypatch.setattr(
        opt,
        'calc_weld_consumable_cylinder',
        lambda x, include_web_to_flange=False: 10.0,
    )

    result = opt.any_constraints_cylinder(
        x='candidate',
        obj=object(),
        init_weight=9.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
    )

    assert result[0] is False
    assert result[1] == 'Weld filter'


def test_cylinder_weld_length_filter_does_not_use_weld_consumables(monkeypatch):
    class FakeCylinder:
        RingStfObj = None
        RingFrameObj = None

    monkeypatch.setattr(opt, 'create_new_cylinder_obj', lambda obj, x: FakeCylinder())
    monkeypatch.setattr(opt, 'calc_weight_cylinder', lambda x: 1000.0)
    monkeypatch.setattr(opt, 'calc_weld_length_cylinder', lambda x, include_web_to_flange=False: 10.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable_cylinder', lambda x, include_web_to_flange=False: 1000.0)

    result = opt.any_constraints_cylinder(
        x='candidate',
        obj=object(),
        init_weight=11.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
        weld_metric='weld_length',
    )

    assert result is None


def test_cylinder_weld_length_filter_rejects_by_length(monkeypatch):
    class FakeCylinder:
        RingStfObj = None
        RingFrameObj = None

        def get_utilization_factors(self, optimizing=False, empty_result_dict=False):
            return {}

    monkeypatch.setattr(opt, 'create_new_cylinder_obj', lambda obj, x: FakeCylinder())
    monkeypatch.setattr(opt, 'calc_weight_cylinder', lambda x: 1000.0)
    monkeypatch.setattr(opt, 'calc_weld_length_cylinder', lambda x, include_web_to_flange=False: 10.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable_cylinder', lambda x, include_web_to_flange=False: 1.0)

    result = opt.any_constraints_cylinder(
        x='candidate',
        obj=object(),
        init_weight=9.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_bias=1.0,
        weld_metric='Weld length',
    )

    assert result[0] is False
    assert result[1] == 'Weld filter'


def test_cylinder_cost_filter_uses_weight_and_selected_weld_metric(monkeypatch):
    class FakeCylinder:
        RingStfObj = None
        RingFrameObj = None

        def get_utilization_factors(self, optimizing=False, empty_result_dict=False):
            return {}

    monkeypatch.setattr(opt, 'create_new_cylinder_obj', lambda obj, x: FakeCylinder())
    monkeypatch.setattr(opt, 'calc_weight_cylinder', lambda x: 100.0)
    monkeypatch.setattr(opt, 'calc_weld_length_cylinder', lambda x, include_web_to_flange=False: 20.0)
    monkeypatch.setattr(opt, 'calc_weld_consumable_cylinder', lambda x, include_web_to_flange=False: 2000.0)

    result = opt.any_constraints_cylinder(
        x='candidate',
        obj=object(),
        init_weight=139.0,
        chk=(False, False, False, False, False, False, False, False, False, False),
        weld_metric='Weld length',
        cost_factors={'steel': 1.0, 'weld': 2.0},
    )

    assert result[0] is False
    assert result[1] == 'Cost filter'


def test_cylinder_constraints_request_optimizer_tuple_for_full_cylinder(monkeypatch):
    class FakeCylinder:
        RingStfObj = object()
        RingFrameObj = object()

        def __init__(self):
            self.optimizing_values = []

        def get_utilization_factors(self, optimizing=False, empty_result_dict=False):
            self.optimizing_values.append(optimizing)
            if empty_result_dict:
                return {}
            if optimizing:
                return True, 'Check OK', {}
            return {'Check OK': True}

    fake_cylinder = FakeCylinder()
    monkeypatch.setattr(opt, 'create_new_cylinder_obj', lambda obj, x: fake_cylinder)
    monkeypatch.setattr(opt, 'calc_weight_cylinder', lambda x: 100.0)

    result = opt.any_constraints_cylinder(
        x='candidate',
        obj=object(),
        init_weight=False,
        chk=(True, False, False, False, False, False, False, False, False, False),
    )

    assert result[0] is True
    assert result[1] == 'Check OK'
    assert fake_cylinder.optimizing_values == [True]


def test_mixed_weld_bias_skips_initial_filter(monkeypatch, opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, _, _, _ = opt_input
    captured = {}

    def fail_get_initial_weight(**kwargs):
        raise AssertionError('mixed weld bias should not use the initial filter')

    def fake_any_smart_loop(*args, **kwargs):
        captured['init_filter'] = args[5]
        captured['processes'] = kwargs['processes']
        captured['weld_bias'] = kwargs['weld_bias']
        return None, None, None, False, []

    monkeypatch.setattr(opt, 'get_initial_weight', fail_get_initial_weight)
    monkeypatch.setattr(opt, 'any_smart_loop', fake_any_smart_loop)

    opt.run_optmizataion(
        obj,
        upper_bounds,
        lower_bounds,
        lat_press,
        deltas,
        algorithm='anysmart',
        use_weight_filter=True,
        weld_bias=0.5,
        processes=1,
    )

    assert captured == {
        'init_filter': float('inf'),
        'processes': 1,
        'weld_bias': 0.5,
    }


def test_run_optimization_forwards_weld_metric_to_flat_algorithms(monkeypatch, opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, _, _, _ = opt_input
    captured = {}

    def fake_any_optimize_loop(*args, **kwargs):
        captured['anydetail'] = kwargs['weld_metric']
        return None, None, None, False, []

    def fake_get_random_result(*args, **kwargs):
        captured['random'] = kwargs['weld_metric']
        return None

    def fake_get_random_result_no_bounds(*args, **kwargs):
        captured['random_no_delta'] = kwargs['weld_metric']
        return None

    monkeypatch.setattr(opt, 'any_optimize_loop', fake_any_optimize_loop)
    monkeypatch.setattr(opt, 'get_random_result', fake_get_random_result)
    monkeypatch.setattr(opt, 'get_random_result_no_bounds', fake_get_random_result_no_bounds)

    for algorithm in ('anydetail', 'random', 'random_no_delta'):
        opt.run_optmizataion(
            obj,
            upper_bounds,
            lower_bounds,
            lat_press,
            deltas,
            algorithm=algorithm,
            use_weight_filter=False,
            weld_bias=1.0,
            weld_metric='Weld length',
        )

    assert captured == {
        'anydetail': 'weld_length',
        'random': 'weld_length',
        'random_no_delta': 'weld_length',
    }


def test_run_optimization_forwards_weld_metric_to_cylinder(monkeypatch, opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, _, _, _ = opt_input
    captured = {}

    def fake_any_smart_loop_cylinder(*args, **kwargs):
        captured['weld_metric'] = kwargs['weld_metric']
        return None, False

    monkeypatch.setattr(opt, 'any_smart_loop_cylinder', fake_any_smart_loop_cylinder)

    opt.run_optmizataion(
        obj,
        upper_bounds,
        lower_bounds,
        lat_press,
        deltas,
        algorithm='anysmart cylinder',
        cylinder=True,
        weld_bias=1.0,
        weld_metric='Weld length',
    )

    assert captured['weld_metric'] == 'weld_length'


def test_run_optimization_forwards_cost_factors_to_flat_algorithm(monkeypatch, opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, _, _, _ = opt_input
    captured = {}

    def fake_any_smart_loop(*args, **kwargs):
        captured['cost_factors'] = kwargs['cost_factors']
        return None, None, None, False, []

    monkeypatch.setattr(opt, 'get_initial_weight', lambda **kwargs: 1.0)
    monkeypatch.setattr(opt, 'any_smart_loop', fake_any_smart_loop)

    opt.run_optmizataion(
        obj,
        upper_bounds,
        lower_bounds,
        lat_press,
        deltas,
        algorithm='anysmart',
        cost_factors={'steel': 2.0, 'weld': 3.0},
    )

    assert captured['cost_factors'] == {'steel': 2.0, 'weld': 3.0}


def test_external_excel_puls_sheet_argument_is_removed(opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, fat_obj, fat_press, _ = opt_input

    with pytest.raises(NotImplementedError, match="External Excel-sheet PULS optimization was removed"):
        opt.run_optmizataion(
            obj,
            upper_bounds,
            lower_bounds,
            lat_press,
            deltas,
            puls_sheet="removed-puls.xlsm",
            fatigue_obj=fat_obj,
            fat_press_ext_int=fat_press,
        )


def test_semi_analytical_optimizer_replacement_is_available(opt_input):
    obj, _, _, lat_press, _, _, _, x0 = opt_input

    calc_object_stf = opt.create_new_calc_obj(obj.Stiffener, x0, None)
    calc_object_pl = opt.create_new_calc_obj(obj.Plate, x0, None)
    calc_object = [
        calc.AllStructure(
            Plate=calc_object_pl[0],
            Stiffener=calc_object_stf[0],
            Girder=None,
            main_dict=obj.get_main_properties()['main dict'],
        ),
        calc_object_pl[1],
    ]

    semi_analytical_result = opt._predict_semi_analytical_uf(calc_object, lat_press)

    assert semi_analytical_result[2] == 1
    assert np.isfinite(semi_analytical_result[0])
    assert np.isfinite(semi_analytical_result[1])

    check_ok, check_not_ok = opt.get_filtered_results(
        [x0],
        obj,
        lat_press,
        float('inf'),
        chk=(False, False, False, False, False, False, False, True, False, False),
        processes=1,
    )

    assert len(check_ok) + len(check_not_ok) == 1
    assert (check_ok or check_not_ok)[0][1] in ('Check OK', 'SemiAnalytical')
