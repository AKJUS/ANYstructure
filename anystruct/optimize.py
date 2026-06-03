# Optimize structure
import numpy as np

import itertools as it
import time
import random
import copy
from multiprocessing import Pool, cpu_count
import math
from math import floor
from matplotlib import pyplot as plt
from tkinter.filedialog import asksaveasfilename
import csv

try:
    import anystruct.calc_structure as calc
    import anystruct.helper as hlp
    import anystruct.calculate_semianalytical as semi_analytical
except ModuleNotFoundError:
    import ANYstructure.anystruct.calc_structure as calc
    import ANYstructure.anystruct.helper as hlp
    import ANYstructure.anystruct.calculate_semianalytical as semi_analytical


DEFAULT_USE_SEMIANALYTICAL_SPEED_STAGING = True


def _set_material_factor_on_structure(obj, material_factor):
    """
    Apply selected material factor to Plate/Stiffener/Girder objects.

    This is important for the ML-Numeric optimizer because the numeric UF
    is converted as:
        UF_checked = UF_predicted * material_factor
    and get_filtered_results reads the material factor from init_stuc_obj.Plate.

    obj may be a single AllStructure-like object or a list/tuple of such objects
    for geometric optimization.
    """
    if material_factor is None:
        return obj

    try:
        mat_fac = float(material_factor)
    except Exception:
        return obj

    def _apply(one_obj):
        for attr_name in ('Plate', 'Stiffener', 'Girder'):
            try:
                part = getattr(one_obj, attr_name)
            except Exception:
                part = None
            if part is not None:
                try:
                    part.mat_factor = mat_fac
                except Exception:
                    pass
        return one_obj

    if isinstance(obj, (list, tuple)):
        for one_obj in obj:
            _apply(one_obj)
    else:
        _apply(obj)

    return obj


def _deactivate_non_prescriptive_girder_buckling(const_chk):
    checks = list(const_chk)
    while len(checks) < 10:
        checks.append(False)
    checks[7] = False
    checks[8] = False
    checks[9] = False
    return tuple(checks)


def run_optmizataion(initial_structure_obj=None, min_var=None, max_var=None, lateral_pressure=None,
                     deltas=None, algorithm='anysmart', trials=30000, side='p',
                     const_chk=(True, True, True, True, True, True, True, False, False, False),
                     pso_options=(100, 0.5, 0.5, 0.5, 100, 1e-8, 1e-8), is_geometric=False, fatigue_obj=None,
                     fat_press_ext_int=None,
                     min_max_span=(2, 6), tot_len=None, frame_height=2.5, frame_distance=None,
                     slamming_press=0, predefined_stiffener_iter=None, processes=None, use_weight_filter=True,
                     load_pre=False, opt_girder_prop=None, puls_sheet=None, puls_acceptance=0.87,
                     fdwn=1, fup=0.5, ml_algo=None, cylinder=False, material_factor=None,
                     weld_bias=0.0, builtup_stiffener=False, weld_metric='weld_consumables',
                     cost_factors=None):
    """
    The optimization is initiated here. It is called from optimize_window / optimize_cylinder.

    weld_bias:
        0.0 = pure weight optimization. Existing behaviour is preserved and no weld
              consumable objective is used downstream.
        1.0 = pure estimated weld consumable optimization.
        Between 0 and 1 = mixed normalized objective.

    builtup_stiffener:
        If True, include web-to-flange weld consumables for built-up stiffeners.
    """
    if puls_sheet is not None:
        raise NotImplementedError(
            "External Excel-sheet PULS optimization was removed. Use the built-in SemiAnalytical replacement, "
            "prescriptive, or ML-Numeric buckling checks."
        )

    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0

    builtup_stiffener = bool(builtup_stiffener)
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    # Make material factor explicit for all optimizer variants.
    # Single optimization receives one structure object; geometric optimization receives a list.
    initial_structure_obj = _set_material_factor_on_structure(initial_structure_obj, material_factor)
    if not cylinder and not is_geometric and getattr(initial_structure_obj, 'Girder', None) is not None:
        const_chk = _deactivate_non_prescriptive_girder_buckling(const_chk)

    init_filter_weight = float('inf')

    if is_geometric:
        fat_dict = [None if this_fat is None else this_fat.get_fatigue_properties() for this_fat in fatigue_obj]
    else:
        fat_dict = None if fatigue_obj is None else fatigue_obj.get_fatigue_properties()

    # Initial filtering is only safe when the objective is a single monotonic
    # quantity: pure weight or pure weld consumables. Mixed normalized
    # objectives need the full valid set before a candidate can be rejected.
    if use_weight_filter and not cylinder and (
            cost_factors is not None or weld_bias <= 0.0 or weld_bias >= 1.0):

        if is_geometric or algorithm == 'pso':
            init_filter_weight = float('inf')
        else:
            predefined_stiffener_iter = None if predefined_stiffener_iter is None else predefined_stiffener_iter

            init_filter_weight = get_initial_weight(obj=initial_structure_obj,
                                                    lat_press=lateral_pressure,
                                                    min_var=min_var, max_var=max_var, deltas=deltas,
                                                    trials=30000 if predefined_stiffener_iter is None else
                                                    len(predefined_stiffener_iter),
                                                    fat_dict=fat_dict,
                                                    fat_press=None if fat_press_ext_int is None else fat_press_ext_int,
                                                    predefined_stiffener_iter=predefined_stiffener_iter,
                                                    slamming_press=slamming_press, fdwn=fdwn, fup=fup,
                                                    ml_algo=ml_algo, weld_bias=weld_bias,
                                                    builtup_stiffener=builtup_stiffener,
                                                    weld_metric=weld_metric,
                                                    cost_factors=cost_factors)

    if cylinder:
        to_return = any_smart_loop_cylinder(
            min_var=min_var,
            max_var=max_var,
            deltas=deltas,
            initial_structure_obj=initial_structure_obj,
            use_weight_filter=use_weight_filter,
            predefiened_stiffener_iter=predefined_stiffener_iter,
            processes=processes,
            weld_bias=weld_bias,
            builtup_stiffener=builtup_stiffener,
            weld_metric=weld_metric,
            cost_factors=cost_factors,
        )
        return to_return

    elif algorithm == 'anysmart' and not is_geometric:
        to_return = any_smart_loop(min_var, max_var, deltas, initial_structure_obj, lateral_pressure,
                                   init_filter_weight, side=side, const_chk=const_chk, fat_dict=fat_dict,
                                   fat_press=fat_press_ext_int, slamming_press=slamming_press,
                                   predefiened_stiffener_iter=predefined_stiffener_iter, puls_sheet=puls_sheet,
                                   puls_acceptance=puls_acceptance, fdwn=fdwn, fup=fup, ml_algo=ml_algo,
                                   weld_bias=weld_bias,
                                   builtup_stiffener=builtup_stiffener,
                                   weld_metric=weld_metric, processes=processes,
                                   cost_factors=cost_factors)
        return to_return
    elif algorithm == 'anysmart' and is_geometric:
        return geometric_summary_search(min_var=min_var, max_var=max_var, deltas=deltas,
                                        initial_structure_obj=initial_structure_obj, lateral_pressure=lateral_pressure,
                                        init_filter=init_filter_weight, side=side, const_chk=const_chk,
                                        fat_obj=fatigue_obj, fat_press=fat_press_ext_int, min_max_span=min_max_span,
                                        tot_len=tot_len, frame_distance=frame_distance,
                                        algorithm='anysmart', predefiened_stiffener_iter=predefined_stiffener_iter,
                                        slamming_press=slamming_press, load_pre=load_pre,
                                        opt_girder_prop=opt_girder_prop, processes=processes, ml_algo=ml_algo,
                                        weld_bias=weld_bias, builtup_stiffener=builtup_stiffener,
                                        weld_metric=weld_metric, cost_factors=cost_factors)
    elif algorithm == 'anydetail' and not is_geometric:
        return any_optimize_loop(min_var, max_var, deltas, initial_structure_obj, lateral_pressure, init_filter_weight,
                                 side=side, const_chk=const_chk, fat_dict=fat_dict, fat_press=fat_press_ext_int,
                                 slamming_press=slamming_press, weld_bias=weld_bias,
                                 builtup_stiffener=builtup_stiffener, weld_metric=weld_metric,
                                 cost_factors=cost_factors)
    elif algorithm == 'random' and not is_geometric:
        return get_random_result(initial_structure_obj, lateral_pressure, min_var, max_var, deltas, trials=trials,
                                 side=side, const_chk=const_chk, fat_dict=fat_dict, fat_press=fat_press_ext_int,
                                 weld_bias=weld_bias, builtup_stiffener=builtup_stiffener,
                                 weld_metric=weld_metric, cost_factors=cost_factors)
    elif algorithm == 'random_no_delta' and not is_geometric:
        return get_random_result_no_bounds(initial_structure_obj, lateral_pressure, min_var, max_var, trials=trials,
                                           side=side, const_chk=const_chk, fat_dict=fat_dict,
                                           fat_press=fat_press_ext_int, weld_bias=weld_bias,
                                           builtup_stiffener=builtup_stiffener, weld_metric=weld_metric,
                                           cost_factors=cost_factors)

    # elif algorithm == 'pso' and is_geometric:
    #     return geometric_summary_search(min_var,max_var,deltas, initial_structure_obj,lateral_pressure,
    #                                     init_filter_weight,side,const_chk,pso_options,fatigue_obj,fat_press_ext_int,
    #                                     min_max_span,tot_len,frame_height,frame_cross_a, 'pso')
    else:
        return None

def any_optimize_loop(min_var, max_var, deltas, initial_structure_obj, lateral_pressure, init_filter=float('inf'),
                      side='p', const_chk=(True, True, True, True, True, False), fat_dict=None, fat_press=None,
                      slamming_press=0, weld_bias=0.0, builtup_stiffener=False,
                      weld_metric='weld_consumables', cost_factors=None):
    '''
    Calulating initial values.
    :param min:
    :param max:
    :return:
    '''
    ass_var = []
    plot_x, plot_y = [], []
    plt.xlabel('#')
    plt.ylabel('weigth [kg]')
    plt.title('ANYdetail brute force results')
    plt.grid(True)
    plt.draw()
    iter_count = 0
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    builtup_stiffener = bool(builtup_stiffener)
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)
    try:
        stiffener_type = (
            initial_structure_obj.Stiffener.get_stiffener_type()
            if initial_structure_obj.Stiffener is not None
            else 'T'
        )
    except Exception:
        stiffener_type = 'T'

    min_objective = init_filter
    main_fail = list()
    for spacing in np.arange(min_var[0], max_var[0] + deltas[0], deltas[0]):
        for plate_thk in np.arange(min_var[1], max_var[1] + deltas[1], deltas[1]):
            for stf_web_h in np.arange(min_var[2], max_var[2] + deltas[2], deltas[2]):
                for stf_web_thk in np.arange(min_var[3], max_var[3] + deltas[3], deltas[3]):
                    for stf_flange_width in np.arange(min_var[4], max_var[4] + deltas[4], deltas[4]):
                        for stf_flange_thk in np.arange(min_var[5], max_var[5] + deltas[5], deltas[5]):
                            var_x = np.array([spacing, plate_thk, stf_web_h, stf_web_thk, stf_flange_width,
                                              stf_flange_thk, min_var[6], min_var[7]])
                            check = any_constraints_all(var_x, initial_structure_obj, lat_press=lateral_pressure,
                                                        init_weight=min_objective, side=side, chk=const_chk,
                                                        fat_dict=fat_dict, fat_press=fat_press,
                                                        slamming_press=slamming_press, weld_bias=weld_bias,
                                                        builtup_stiffener=builtup_stiffener,
                                                        weld_metric=weld_metric,
                                                        cost_factors=cost_factors)
                            if check[0] is not False:
                                current_objective = calc_flat_objective_value(
                                    var_x,
                                    stiffener_type=stiffener_type,
                                    weld_bias=weld_bias,
                                    include_web_to_flange=builtup_stiffener,
                                    weld_metric=weld_metric,
                                    cost_factors=cost_factors,
                                )

                                if current_objective <= min_objective:
                                    iter_count += 1
                                    min_objective = current_objective
                                    ass_var = var_x
                                main_fail.append(check)
                            else:
                                main_fail.append(check)
    if ass_var is None:
        return None, None, None, False, main_fail

    new_struc_obj = create_new_structure_obj(initial_structure_obj, [item for item in ass_var])
    new_calc_obj = create_new_calc_obj(initial_structure_obj, [item for item in ass_var])[0]

    return new_struc_obj, new_calc_obj, fat_dict, True, main_fail


def any_smart_loop(min_var, max_var, deltas, initial_structure_obj, lateral_pressure, init_filter=float('inf'),
                   side='p', const_chk=(True, True, True, True, True, True, True, False, False, False),
                   fat_dict=None, fat_press=None,
                   slamming_press=0, predefiened_stiffener_iter=None, processes=None,
                   puls_sheet=None, puls_acceptance=0.87, fdwn=1, fup=0.5, ml_algo=None,
                   weld_bias=0.0, builtup_stiffener=False, weld_metric='weld_consumables',
                   cost_factors=None):
    """
    Trying to be smart.

    weld_bias:
        0.0 = pure weight optimization.
              IMPORTANT: no weld consumable calculations are performed.
        1.0 = pure weld consumable optimization.
        Between 0 and 1 = normalized mixed objective.

    builtup_stiffener:
        If True, include web-to-flange weld consumables in addition to
        plate-to-stiffener weld consumables.
    """
    initial_structure_obj.lat_press = lateral_pressure

    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    cost_factors = normalize_cost_factors(cost_factors)

    if predefiened_stiffener_iter is None:
        structure_to_check = any_get_all_combs(min_var, max_var, deltas)
    else:
        structure_to_check = any_get_all_combs(
            min_var,
            max_var,
            deltas,
            predef_stiffeners=[item.get_tuple() for item in predefiened_stiffener_iter],
        )

    main_result = get_filtered_results(structure_to_check, initial_structure_obj, lateral_pressure,
                                       init_filter_weight=init_filter, side=side, chk=const_chk, fat_dict=fat_dict,
                                       fat_press=fat_press, slamming_press=slamming_press, processes=processes,
                                       puls_sheet=puls_sheet, puls_acceptance=puls_acceptance, ml_algo=ml_algo,
                                       weld_bias=weld_bias, builtup_stiffener=builtup_stiffener,
                                       weld_metric=weld_metric, cost_factors=cost_factors)

    main_iter = main_result[0]
    main_fail = main_result[1]

    ass_var = None

    if cost_factors is not None:
        current_score = float('inf')
        try:
            stiffener_type = (
                initial_structure_obj.Stiffener.get_stiffener_type()
                if initial_structure_obj.Stiffener is not None
                else 'T'
            )
        except Exception:
            stiffener_type = 'T'

        for item in main_iter:
            main_fail.append(item)
            objective_score = calc_flat_objective_value(
                item[2],
                stiffener_type=stiffener_type,
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )
            if objective_score < current_score:
                ass_var = item[2]
                current_score = objective_score

    elif weld_bias <= 0.0:
        current_weight = float('inf')
        for item in main_iter:
            main_fail.append(item)
            item_weight = calc_weight(item[2])
            if item_weight < current_weight:
                ass_var = item[2]
                current_weight = item_weight
    else:
        if len(main_iter) == 0:
            return None, None, None, False, main_fail

        try:
            stiffener_type = (
                initial_structure_obj.Stiffener.get_stiffener_type()
                if initial_structure_obj.Stiffener is not None
                else 'T'
            )
        except Exception:
            stiffener_type = 'T'

        valid_x = [item[2] for item in main_iter]
        weight_values = [calc_weight(x) for x in valid_x]
        weld_values = [
            calc_weld_objective(
                x,
                stiffener_type=stiffener_type,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
            )
            for x in valid_x
        ]

        positive_weight_values = [val for val in weight_values if val > 0.0]
        positive_weld_values = [val for val in weld_values if val > 0.0]

        weight_ref = min(positive_weight_values) if len(positive_weight_values) > 0 else 1.0
        weld_ref = min(positive_weld_values) if len(positive_weld_values) > 0 else 1.0

        weight_bias = 1.0 - weld_bias
        current_score = float('inf')

        for item, item_weight, item_weld in zip(main_iter, weight_values, weld_values):
            main_fail.append(item)
            weight_score = item_weight / weight_ref
            weld_score = item_weld / weld_ref
            objective_score = weight_bias * weight_score + weld_bias * weld_score

            if objective_score < current_score:
                ass_var = item[2]
                current_score = objective_score

    if ass_var == None:
        return None, None, None, False, main_fail

    if len(ass_var) >= 12:
        ass_var = [round(item, 10) for item in ass_var[0:12]]
    elif len(ass_var) == 8:
        ass_var = [round(item, 10) for item in ass_var[0:8]]
    else:
        ass_var = [round(item, 10) for item in ass_var[0:8]] + [ass_var[8]]

    calc_object = create_new_calc_obj(initial_structure_obj, ass_var, fat_dict, fdwn=fdwn, fup=fup)[0]
    calc_object.lat_press = lateral_pressure

    return calc_object, fat_dict, True, main_fail

def any_smart_loop_cylinder(min_var, max_var, deltas, initial_structure_obj, lateral_pressure=None,
                            init_filter=float('inf'),
                            side='p', const_chk=(True, True, True, True, True, True, True, False, False, False),
                            fat_dict=None,
                            fat_press=None, slamming_press=0, predefiened_stiffener_iter=None, processes=None,
                            fdwn=1, fup=0.5, ml_algo=None, use_weight_filter=True,
                            weld_bias=0.0, builtup_stiffener=False, weld_metric='weld_consumables',
                            cost_factors=None):
    """
    Cylinder optimization.

    weld_bias:
        0.0 = pure weight optimization. Existing behaviour is preserved and no
              weld consumable calculations are performed.
        1.0 = pure estimated weld consumable optimization.
        Between 0 and 1 = mixed normalized objective.

    builtup_stiffener:
        If True, include web-to-flange welds for built-up stiffeners.
    """
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0

    builtup_stiffener = bool(builtup_stiffener)
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    combs = list()

    for idx, str_type in enumerate(range(len(min_var))):
        if sum(min_var[idx]) == 0:
            structure_to_check = [(0, 0, 0, 0, 0, 0, 0, 0), ]
        else:
            if any([predefiened_stiffener_iter is None, idx == 0]):
                if initial_structure_obj.LongStfObj is not None:
                    initial_structure_obj.LongStfObj.stiffener_type = 'T'
                structure_to_check = any_get_all_combs(min_var[idx], max_var[idx], deltas[idx])
            else:
                structure_to_check = any_get_all_combs(
                    min_var[idx],
                    max_var[idx],
                    deltas[idx],
                    predef_stiffeners=[item.get_tuple() for item in predefiened_stiffener_iter],
                )
        combs.append(structure_to_check)

    final_comb = list()
    for shell in combs[0]:
        for long in combs[1]:
            for ring_stf in combs[2]:
                for ring_frame in combs[3]:
                    final_comb.append([[shell, long, ring_stf, ring_frame], initial_structure_obj])

    min_filter_value = float('inf')
    use_initial_filter = use_weight_filter and (
            cost_factors is not None or weld_bias <= 0.0 or weld_bias >= 1.0)
    if use_initial_filter:
        to_check = [
            tuple(random.choice(final_comb) + [float('inf'), None, 'p',
                                               (True, True, True, True, True, True, True, False, False, False),
                                               None, None, 0, 1, 0.5, None, weld_bias, builtup_stiffener,
                                               weld_metric, cost_factors])
            for dummy in range(10000)
        ]
        pool_processes = max(cpu_count() - 1, 1) if processes is None else int(processes)
        if pool_processes == 1:
            res_pre = [any_constraints_cylinder(*args) for args in to_check]
        else:
            with Pool(processes=pool_processes) as my_process:
                res_pre = my_process.starmap(any_constraints_cylinder, to_check)

        for chk_res in res_pre:
            if chk_res[0]:
                current_value = calc_cylinder_objective_value(
                    chk_res[2],
                    weld_bias=weld_bias,
                    include_web_to_flange=builtup_stiffener,
                    weld_metric=weld_metric,
                    cost_factors=cost_factors,
                )
                if current_value < min_filter_value:
                    min_filter_value = current_value
    else:
        min_filter_value = False

    final_comb_inc_weight = list()
    for val in final_comb:
        final_comb_inc_weight.append(tuple(val + [min_filter_value, None, 'p',
                                                  (True, True, True, True, True, True, True, False, False, False),
                                                  None, None, 0, 1, 0.5, None, weld_bias, builtup_stiffener,
                                                  weld_metric, cost_factors]))

    pool_processes = max(cpu_count() - 1, 1) if processes is None else int(processes)
    if pool_processes == 1:
        res_pre = [any_constraints_cylinder(*args) for args in final_comb_inc_weight]
    else:
        with Pool(processes=pool_processes) as my_process:
            res_pre = my_process.starmap(any_constraints_cylinder, final_comb_inc_weight)

    check_ok, check_not_ok = list(), list()
    for item in res_pre:
        if item[0] is False:
            check_not_ok.append(item)
        else:
            check_ok.append(item)

    main_iter = check_ok
    main_fail = check_not_ok
    ass_var = None

    if cost_factors is not None:
        current_score = float('inf')
        for item in main_iter:
            main_fail.append(item)
            objective_score = calc_cylinder_objective_value(
                item[2],
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )
            if objective_score < current_score:
                ass_var = item[2]
                current_score = objective_score

    elif weld_bias <= 0.0:
        current_weight = float('inf')
        for item in main_iter:
            main_fail.append(item)
            item_weight = calc_weight_cylinder(item[2])
            if item_weight < current_weight:
                ass_var = item[2]
                current_weight = item_weight
    else:
        if len(main_iter) == 0:
            return None, None, None, False, main_fail

        valid_x = [item[2] for item in main_iter]
        weight_values = [calc_weight_cylinder(x) for x in valid_x]
        weld_values = [
            calc_weld_objective_cylinder(
                x,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
            )
            for x in valid_x
        ]

        positive_weight_values = [val for val in weight_values if val > 0.0]
        positive_weld_values = [val for val in weld_values if val > 0.0]
        weight_ref = min(positive_weight_values) if len(positive_weight_values) > 0 else 1.0
        weld_ref = min(positive_weld_values) if len(positive_weld_values) > 0 else 1.0

        weight_bias = 1.0 - weld_bias
        current_score = float('inf')

        for item, item_weight, item_weld in zip(main_iter, weight_values, weld_values):
            main_fail.append(item)
            objective_score = weight_bias * (item_weight / weight_ref) + weld_bias * (item_weld / weld_ref)
            if objective_score < current_score:
                ass_var = item[2]
                current_score = objective_score

    if ass_var == None:
        return None, None, None, False, main_fail

    new_cylinder_obj = create_new_cylinder_obj(initial_structure_obj, ass_var)
    return new_cylinder_obj, main_fail

def any_smart_loop_geometric(min_var, max_var, deltas, initial_structure_obj, lateral_pressure,
                             init_filter=float('inf'),
                             side='p', const_chk=(True, True, True, True, True, True), fat_obj=None, fat_press=None,
                             slamming_press=None, predefiened_stiffener_iter=None, processes=None, ml_algo=None,
                             weld_bias=0.0, builtup_stiffener=False, weld_metric='weld_consumables',
                             cost_factors=None):
    ''' Searching multiple sections using the smart loop. '''

    all_obj = []
    idx = 0
    for struc_obj, lat_press, fatigue_obj, fatigue_press, slam_press in zip(initial_structure_obj, lateral_pressure,
                                                                            fat_obj, fat_press, slamming_press):
        # print(predefiened_stiffener_iter)
        if predefiened_stiffener_iter is not None:
            this_predefiened_objects = hlp.helper_read_section_file(predefiened_stiffener_iter, struc_obj.Stiffener)
        else:
            this_predefiened_objects = None

        opt_obj = any_smart_loop(min_var=min_var, max_var=max_var, deltas=deltas, initial_structure_obj=struc_obj,
                                 lateral_pressure=lat_press, init_filter=init_filter, side=side,
                                 const_chk=const_chk,
                                 fat_dict=None if fatigue_obj is None else fatigue_obj.get_fatigue_properties(),
                                 fat_press=None if fatigue_press is None else fatigue_press,
                                 slamming_press=0 if slam_press is None else slam_press,
                                 predefiened_stiffener_iter=this_predefiened_objects, processes=processes,
                                 ml_algo=ml_algo, weld_bias=weld_bias,
                                 builtup_stiffener=builtup_stiffener, weld_metric=weld_metric,
                                 cost_factors=cost_factors)

        all_obj.append(opt_obj)
        idx += 1

    return all_obj


def geometric_summary_search(min_var=None, max_var=None, deltas=None, initial_structure_obj=None, lateral_pressure=None,
                             init_filter=float('inf'), side='p', const_chk=(True, True, True, True, True, True),
                             pso_options=(100, 0.5, 0.5, 0.5, 100, 1e-8, 1e-8), fat_obj=None, fat_press=None,
                             min_max_span=(2, 6), tot_len=None, frame_distance=None,
                             algorithm='anysmart', predefiened_stiffener_iter=None, reiterate=True,
                             processes=None, slamming_press=None, load_pre=False, opt_girder_prop=None,
                             ml_algo=None, weld_bias=0.0, builtup_stiffener=False,
                             weld_metric='weld_consumables', cost_factors=None):
    '''Geometric optimization of all relevant sections. '''
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0

    builtup_stiffener = bool(builtup_stiffener)
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    # Checking the number of initial objects and adding if number of fraction is to be changed.
    # print('Min/max span is', min_max_span)
    found_max, found_min = False, False
    for frames in range(1, 100):
        frame_count = frames
        if tot_len / frames <= min_max_span[1] and found_min is False:
            min_frame_count = frame_count - 1
            found_min = True
        if tot_len / frames <= min_max_span[0] and found_max is False:
            max_frame_count = frame_count - 1
            found_max = True
        if found_min and found_max:
            break

    results = {}
    # print('Frame count min/max: ', min_frame_count, max_frame_count)
    # print('Initial objects: ', [print(type(obj)) for obj in initial_structure_obj])
    # print('Initial lateral: ', lateral_pressure)
    working_objects = {}
    working_lateral = {}
    working_fatigue = {}
    working_fatigue_press = {}
    working_slamming = {}

    for no_of_fractions in range(min_frame_count + 1, max_frame_count + 1):
        # Create fraction varables
        frac_var, min_frac, max_frac = [], [], []

        for var in range(no_of_fractions):
            # Frame height is a interpolation between height at start and end.
            frac_var.append(1 / no_of_fractions)
            working_objects[no_of_fractions] = list(initial_structure_obj)
            working_lateral[no_of_fractions] = list(lateral_pressure)
            working_fatigue[no_of_fractions] = list(fat_obj)
            working_fatigue_press[no_of_fractions] = list(fat_press)
            working_slamming[no_of_fractions] = list(slamming_press)
            similar_count = len(working_objects[no_of_fractions])
            tick_tock = True

            while similar_count != no_of_fractions * 2:

                if similar_count > no_of_fractions * 2:
                    for var_dict in [working_objects, working_lateral, working_fatigue,
                                     working_fatigue_press, working_slamming]:
                        if tick_tock:
                            lower_idx = 0
                            upper_idx = int(floor(len(working_objects[no_of_fractions]) / 2))
                            tick_tock = False
                        else:
                            lower_idx = int(len(working_objects[no_of_fractions]) / 2) - 1
                            upper_idx = -1
                            tick_tock = True

                        var_dict[no_of_fractions].pop(lower_idx)
                        var_dict[no_of_fractions].pop(upper_idx)
                    similar_count -= 2
                else:
                    if tick_tock:
                        lower_idx = 0
                        upper_idx = int(len(working_objects[no_of_fractions]) / 2)
                        tick_tock = False
                    else:
                        lower_idx = int(len(working_objects[no_of_fractions]) / 2) - 1
                        upper_idx = -1
                        tick_tock = True
                    # print(no_of_fractions, int(ceil(len(working_objects[no_of_fractions])/2)))

                    obj_start, obj_stop = copy.deepcopy(working_objects[no_of_fractions][lower_idx]), \
                        copy.deepcopy(working_objects[no_of_fractions][upper_idx])

                    fat_obj_start, fat_obj_stop = copy.deepcopy(working_fatigue[no_of_fractions][lower_idx]), \
                        copy.deepcopy(working_fatigue[no_of_fractions][upper_idx])
                    lat_start, lat_stop = working_lateral[no_of_fractions][lower_idx], \
                        working_lateral[no_of_fractions][upper_idx]
                    fat_press_start, fat_press_stop = working_fatigue_press[no_of_fractions][lower_idx], \
                        working_fatigue_press[no_of_fractions][upper_idx]
                    slam_start, slam_stop = working_slamming[no_of_fractions][lower_idx], \
                        working_slamming[no_of_fractions][upper_idx]
                    # if no_of_fractions == 11:
                    #     print('Tick/tock', tick_tock, 'lower/opper idx', lower_idx, upper_idx)

                    for work, work_input in zip([working_objects[no_of_fractions], working_lateral[no_of_fractions],
                                                 working_fatigue[no_of_fractions],
                                                 working_fatigue_press[no_of_fractions],
                                                 working_slamming[no_of_fractions]],
                                                [(obj_start, obj_stop), (lat_start, lat_stop),
                                                 (fat_obj_start, fat_obj_stop), (fat_press_start, fat_press_stop),
                                                 (slam_start, slam_stop)]):
                        # First iteration tick_tock true, second tick_tock false
                        if not tick_tock:
                            lower_idx = lower_idx
                            upper_idx = upper_idx + 1
                        else:
                            lower_idx = lower_idx + 1
                            upper_idx = -1
                        work.insert(lower_idx, work_input[0])
                        work.insert(upper_idx, work_input[1])
                    similar_count += 2
                # if no_of_fractions == 11:
                #     [print(item.get_structure_type()) for item in working_objects[no_of_fractions]]
                #     print('')
        for no_of_fractions, struc_objects in working_objects.items():
            for struc_obj in struc_objects:
                struc_obj.Plate.set_span(tot_len / no_of_fractions)
                struc_obj.Stiffener.set_span(tot_len / no_of_fractions)

        solution_found, iterations = False, 0

        while not solution_found:
            iterations += 1
            if iterations != 1:
                min_var[0:6] += deltas / 2
                max_var[0:6] -= deltas / 2

            if algorithm == 'anysmart':
                if load_pre:
                    import pickle
                    with open('geo_opt_2.pickle', 'rb') as file:
                        opt_objects = pickle.load(file)[no_of_fractions][1]
                else:

                    opt_objects = any_smart_loop_geometric(min_var=min_var, max_var=max_var, deltas=deltas,
                                                           initial_structure_obj=working_objects[no_of_fractions],
                                                           lateral_pressure=working_lateral[no_of_fractions],
                                                           init_filter=init_filter, side=side, const_chk=const_chk,
                                                           fat_obj=working_fatigue[no_of_fractions],
                                                           slamming_press=working_slamming[no_of_fractions],
                                                           fat_press=working_fatigue_press[no_of_fractions],
                                                           predefiened_stiffener_iter=predefiened_stiffener_iter,
                                                           processes=processes,
                                                           ml_algo=ml_algo,
                                                           weld_bias=weld_bias,
                                                           builtup_stiffener=builtup_stiffener,
                                                           weld_metric=weld_metric,
                                                           cost_factors=cost_factors)

            # Finding weight of this solution.

            tot_weight, tot_weld, frame_spacings, valid, width, weight_details = \
                0, 0, [None for dummy in range(len(opt_objects))], \
                True, 10, {'frames': list(), 'objects': list(),
                           'scales': list(), 'weld_objects': list(), 'weld_frames': list()}

            # print('Weight for', no_of_fractions)
            for count, opt in enumerate(opt_objects):
                obj = opt[0]

                if opt[3]:
                    weigth_to_add = calc_weight((obj.Plate.get_s(), obj.Plate.get_pl_thk(), obj.Stiffener.get_web_h(),
                                                 obj.Stiffener.get_web_thk(),
                                                 obj.Stiffener.get_fl_w(), obj.Stiffener.get_fl_thk(),
                                                 obj.Plate.span, width), prt=False)
                    tot_weight += weigth_to_add
                    weight_details['objects'].append(weigth_to_add)

                    if weld_bias > 0.0 or cost_factors is not None:
                        panel_x = (obj.Plate.get_s(), obj.Plate.get_pl_thk(), obj.Stiffener.get_web_h(),
                                   obj.Stiffener.get_web_thk(), obj.Stiffener.get_fl_w(),
                                   obj.Stiffener.get_fl_thk(), obj.Plate.span, width)
                        weld_to_add = calc_weld_objective(
                            panel_x,
                            stiffener_type=obj.Stiffener.get_stiffener_type(),
                            include_web_to_flange=builtup_stiffener,
                            weld_metric=weld_metric,
                        )
                        tot_weld += weld_to_add
                        weight_details['weld_objects'].append(weld_to_add)

                    if frame_spacings[count // 2] is None:
                        frame_spacings[count // 2] = obj.Plate.get_s()
                    # print('added normal weight', weigth_to_add)

                else:
                    # In this case there are no applicable solutions found in the specified dimension ranges.
                    tot_weight += float('inf')
                    valid = False
            if valid:
                # print(frame_distance)
                for frame in range(no_of_fractions - 1):
                    frame_height = 2.5 if frame_distance is None else frame_distance['start_dist'] + \
                                                                      (frame_distance['stop_dist'] -
                                                                       frame_distance['start_dist']) * \
                                                                      ((frame + 1) / no_of_fractions)

                    # pl_area, stf_area = 0.018 * width, 0.25 * 0.015 * (width//frame_spacings[frame])
                    this_x = (frame_spacings[frame], opt_girder_prop[0], opt_girder_prop[1], opt_girder_prop[2],
                              opt_girder_prop[3], opt_girder_prop[4], None, width)
                    this_weight = sum(get_field_tot_area(this_x)) * frame_height * 7850
                    scale_max, scale_min = opt_girder_prop[5], opt_girder_prop[6]

                    this_scale = scale_min + (scale_max - scale_min) * (abs((max_frame_count - (count + 1) / 2)) /
                                                                        (max_frame_count - min_frame_count))
                    # print('Number of fractions', no_of_fractions, 'Scale', this_scale)
                    tot_weight += this_weight * this_scale
                    solution_found = True
                    # print('added frame weight', this_weight * this_scale)
                    weight_details['frames'].append(this_weight * this_scale)
                    weight_details['scales'].append(this_scale)

                    if weld_bias > 0.0 or cost_factors is not None:
                        frame_weld = calc_weld_objective(
                            this_x,
                            include_web_to_flange=builtup_stiffener,
                            weld_metric=weld_metric,
                        ) * this_scale
                        tot_weld += frame_weld
                        weight_details['weld_frames'].append(frame_weld)
            elif iterations == 2:
                solution_found = True  # Only iterate once.

            if predefiened_stiffener_iter is not None or not reiterate:
                solution_found = True  # Noe solution may be found, but in this case no more iteations.

        if cost_factors is not None:
            total_objective = cost_factors['steel'] * tot_weight + cost_factors['weld'] * tot_weld
            weight_details['total_weight'] = tot_weight
            weight_details['total_weld_consumables'] = tot_weld
            weight_details['objective'] = total_objective
            results[no_of_fractions] = total_objective, opt_objects, weight_details
        elif weld_bias > 0.0:
            total_objective = (1.0 - weld_bias) * tot_weight + weld_bias * tot_weld
            weight_details['total_weight'] = tot_weight
            weight_details['total_weld_consumables'] = tot_weld
            weight_details['objective'] = total_objective
            results[no_of_fractions] = total_objective, opt_objects, weight_details
        else:
            results[no_of_fractions] = tot_weight, opt_objects, weight_details
    # for key, val in results.items():
    #     print(key)
    #     print(val)
    return results


def any_find_min_weight_var(var):
    '''
    Find the minimum weight of the inpu
    :param min:
    :param max:
    :return:
    '''

    return min(map(calc_weight))


def any_constraints_cylinder(x, obj: calc.CylinderAndCurvedPlate, init_weight, lat_press=None, side='p',
                             chk=(True, True, True, True, True, True, True, False, False, False),
                             fat_dict=None, fat_press=None, slamming_press=0, fdwn=1, fup=0.5,
                             ml_results=None, weld_bias=0.0, builtup_stiffener=False,
                             weld_metric='weld_consumables', cost_factors=None):
    '''
    Checking all constraints defined.

        iter_var = ((item,init_stuc_obj,lat_press,init_filter_weight,side,chk,fat_dict,fat_press,slamming_press, PULSrun)
                for item in iterable_all)
    :param x:
    :return:
    '''

    all_checks = [0, 0, 0, 0, 0, 0, 0, 0]
    check_map = {'weight': 0, 'UF unstiffened': 1, 'Column stability': 2, 'UF longitudinal stiffeners': 3,
                 'Stiffener check': 4, 'UF ring stiffeners': 5, 'UF ring frame': 6, 'Check OK': 7}

    calc_obj = create_new_cylinder_obj(obj, x)

    optimizing = True
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    cost_factors = normalize_cost_factors(cost_factors)
    weld_metric = normalize_weld_metric(weld_metric)

    # Initial objective filter.
    if init_weight != False:
        if cost_factors is not None:
            filter_value = calc_cylinder_objective_value(
                x,
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )
            filter_name = 'Cost filter'
        elif weld_bias >= 1.0:
            filter_value = calc_weld_objective_cylinder(
                x,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
            )
            filter_name = 'Weld filter'
        else:
            filter_value = calc_weight_cylinder(x)
            filter_name = 'Weight filter'

        if filter_value > init_weight:
            results = calc_obj.get_utilization_factors(optimizing=optimizing, empty_result_dict=True)
            results['Weight'] = calc_weight_cylinder(x)
            if init_weight == 0:
                all_checks[0] = float('inf')
            else:
                all_checks[0] = filter_value / init_weight
            return False, filter_name, x, all_checks, calc_obj

    if chk[0]:
        results = calc_obj.get_utilization_factors(optimizing=optimizing)
        if results[0]:
            all_checks[check_map[results[1]]] += 1
            return True, results[1], x, all_checks, calc_obj
        else:
            all_checks[check_map[results[1]]] += 1
            return False, results[1], x, all_checks, calc_obj


def _get_ml_input_for_optimization(calc_object, lat_press):
    """
    Return the correct ML input row for an optimization candidate.

    UP models use the plate object input.
    SP models use the stiffener object input, because SP numeric/regression models
    were trained with stiffener geometry included.
    """
    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object

    if all_structure.Plate.get_puls_sp_or_up() == 'UP':
        return all_structure.Plate.get_buckling_ml_input(lat_press, alone=False)

    if all_structure.Stiffener is not None:
        return all_structure.Stiffener.get_buckling_ml_input(lat_press, alone=False)

    # Defensive fallback. SP without stiffener should normally not occur.
    return all_structure.Plate.get_buckling_ml_input(lat_press, alone=False)


def _is_integrated_puls_boundary(boundary):
    return str(boundary).strip().lower() in {'int', 'integrated'}


def _get_numeric_pipeline_prefix(calc_object):
    """Return numeric ML pipeline prefix for this optimization candidate."""
    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object

    sp_or_up = all_structure.Plate.get_puls_sp_or_up()

    if sp_or_up == 'UP':
        boundary = all_structure.Plate.get_puls_boundary()
        return 'num UP int' if _is_integrated_puls_boundary(boundary) else 'num UP GLGT'

    if all_structure.Stiffener is not None:
        boundary = all_structure.Stiffener.get_puls_boundary()
    else:
        boundary = all_structure.Plate.get_puls_boundary()

    return 'num SP int' if _is_integrated_puls_boundary(boundary) else 'num SP GLGT'


def _predict_numeric_uf_group(ml_algo, input_rows, prefix, mat_fac):
    """
    Batch-predict numeric UF for optimization.

    Returns an array with columns:
        [buckling_uf_material_factored, ultimate_uf_material_factored, valid_prediction]

    The numeric regressor predicts UF at material factor = 1.0.
    For optimization checks, the values are converted using:
        UF = predicted_UF * mat_fac
    and then compared against 1.0.
    """
    result = np.full((len(input_rows), 3), np.inf, dtype=float)

    if len(input_rows) == 0:
        return result

    result[:, 2] = 0.0

    required_keys = [
        f'{prefix} validity predictor',
        f'{prefix} validity xscaler',
        f'{prefix} UF reg predictor',
        f'{prefix} UF reg xscaler',
        f'{prefix} UF reg yscaler',
    ]

    for key in required_keys:
        if ml_algo is None or key not in ml_algo or ml_algo[key] is None:
            return result

    try:
        x_valid = ml_algo[f'{prefix} validity xscaler'].transform(input_rows)
        valid_pred = ml_algo[f'{prefix} validity predictor'].predict(x_valid)
        valid_pred = np.asarray(valid_pred, dtype=float).ravel()
        result[:, 2] = valid_pred

        valid_idx = np.where(valid_pred == 1)[0]
        if len(valid_idx) == 0:
            return result

        valid_inputs = [input_rows[idx] for idx in valid_idx]
        x_reg = ml_algo[f'{prefix} UF reg xscaler'].transform(valid_inputs)
        y_scaled = ml_algo[f'{prefix} UF reg predictor'].predict(x_reg)
        y_raw = ml_algo[f'{prefix} UF reg yscaler'].inverse_transform(y_scaled)

        # Material factor correction. The regressor output is for mat. factor = 1.0.
        y_factored = y_raw * float(mat_fac)

        result[valid_idx, 0] = y_factored[:, 0]
        result[valid_idx, 1] = y_factored[:, 1]

        return result

    except Exception:
        # If numeric prediction fails, mark all as invalid so optimization rejects them.
        return result


def _semi_analytical_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _semi_analytical_stiffener_type(value):
    return {
        'T': 'T-bar',
        'T-bar': 'T-bar',
        'L': 'Angle',
        'Angle': 'Angle',
        'L-bulb': 'L-bulb',
        'FB': 'Flatbar',
        'F': 'Flatbar',
        'Flatbar': 'Flatbar',
    }.get(value, value)


def _semi_analytical_stiffener_boundary(value):
    return {
        'C': 'Cont',
        'Cont': 'Cont',
        'Continuous': 'Cont',
        'S': 'Sniped',
        'Sniped': 'Sniped',
    }.get(value, value)


def _semi_analytical_in_plane_support(value):
    return {
        'Int': 'Integrated',
        'Integrated': 'Integrated',
        'GL': 'Girder - long',
        'Girder - long': 'Girder - long',
        'GT': 'Girder - trans',
        'Girder - trans': 'Girder - trans',
    }.get(value, value)


def _puls_selected_method(value):
    text = str(value).strip().lower()
    if text in ('1', 'buckling'):
        return 'buckling'
    if text in ('2', 'ultimate'):
        return 'ultimate'
    return text


def _get_semi_analytical_input_for_optimization(calc_object, lat_press):
    """
    Build the reduced SemiAnalytical input from an optimization candidate.

    The replacement currently covers regular stiffened SP panels.  It follows
    the same candidate object and pressure convention as the ML-Numeric SP
    path: geometry/stresses are in the candidate object, and lateral pressure
    is converted from kPa to MPa for the PULS-style input surface.
    """
    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object

    if all_structure.Plate.get_puls_sp_or_up() != 'SP':
        return None
    if all_structure.Stiffener is None:
        return None

    stiffener = all_structure.Stiffener
    puls_boundary = stiffener.get_puls_boundary()
    sig_x1 = stiffener.sigma_x1
    sig_x2 = stiffener.sigma_x2
    if sig_x1 * sig_x2 >= 0:
        sigxd = sig_x1 if abs(sig_x1) > abs(sig_x2) else sig_x2
    else:
        sigxd = max(sig_x1, sig_x2)

    pressure_mpa = _semi_analytical_float(lat_press) / 1000.0
    return semi_analytical.S3PanelInput(
        length=stiffener.span * 1000.0,
        stiffener_spacing=stiffener.spacing,
        plate_thickness=stiffener.t,
        stiffener_type=_semi_analytical_stiffener_type(stiffener.get_stiffener_type()),
        stiffener_boundary=_semi_analytical_stiffener_boundary(stiffener.get_puls_stf_end()),
        stiffener_height=stiffener.hw,
        web_thickness=stiffener.tw,
        flange_width=stiffener.b,
        flange_thickness=stiffener.tf,
        yield_stress_plate=stiffener.mat_yield / 1e6,
        yield_stress_stiffener=stiffener.mat_yield / 1e6,
        axial_stress=0.0 if puls_boundary == 'GT' else sigxd,
        transverse_stress_1=0.0 if puls_boundary == 'GL' else stiffener.sigma_y1,
        transverse_stress_2=0.0 if puls_boundary == 'GL' else stiffener.sigma_y2,
        shear_stress=stiffener.tau_xy,
        pressure=pressure_mpa,
        in_plane_support=_semi_analytical_in_plane_support(puls_boundary),
        elastic_modulus=210000.0,
        poisson_ratio=0.3,
    )


def _predict_semi_analytical_uf(calc_object, lat_press, default_acceptance=0.87, selected_method=None, cache=None):
    """
    Return [buckling_uf, ultimate_uf, valid_prediction, acceptance_limit].
    """
    result = np.array([float('inf'), float('inf'), 0.0, float(default_acceptance)], dtype=float)

    try:
        if selected_method is None:
            try:
                selected_method = _puls_selected_method(calc_object[0].Plate.get_puls_method())
            except Exception:
                selected_method = None
        if hasattr(semi_analytical, 'predict_anystructure_uf_with_acceptance'):
            return semi_analytical.predict_anystructure_uf_with_acceptance(
                calc_object,
                lat_press,
                default_acceptance=default_acceptance,
                selected_method=selected_method,
                cache=cache,
            )
        if hasattr(semi_analytical, 'predict_anystructure_uf'):
            result[0:3] = semi_analytical.predict_anystructure_uf(
                calc_object,
                lat_press,
                selected_method=selected_method,
                cache=cache,
            )
            return result

        panel = _get_semi_analytical_input_for_optimization(calc_object, lat_press)
        if panel is None:
            return result

        solved = semi_analytical.solve_s3_panel(panel)
        if (
            solved.valid
            and solved.buckling_usage_factor is not None
            and solved.ultimate_usage_factor is not None
        ):
            result[0] = float(solved.buckling_usage_factor)
            result[1] = float(solved.ultimate_usage_factor)
            result[2] = 1.0
    except Exception:
        pass

    return result


def any_constraints_all(x, obj, lat_press, init_weight, side='p', chk=(True, True, True, True, True, True, True, False,
                                                                       False, False),
                        fat_dict=None, fat_press=None, slamming_press=0, PULSrun: calc.PULSpanel = None,
                        print_result=False, fdwn=1, fup=0.5, ml_results=None, random_result_return=False,
                        weld_bias=0.0, builtup_stiffener=False, weld_metric='weld_consumables',
                        cost_factors=None):
    '''
    Checking all constraints defined.

    ml_results is used for both ML pipelines:
        chk[7]  SemiAnalytical replacement:
            ml_results[0] = buckling UF
            ml_results[1] = ultimate UF
            ml_results[2] = valid prediction flag, 1 = valid, 0 = invalid/unsupported
            ml_results[3] = SemiAnalytical acceptance limit
            accepted if selected UF / acceptance < 1.0

        chk[8]  ML-CL:
            deactivated; ML-Numeric has replaced this classifier path

        chk[9]  ML-Numeric:
            ml_results[0] = material-factored buckling UF
            ml_results[1] = material-factored ultimate UF
            ml_results[2] = valid numeric prediction flag, 1 = valid, 0 = invalid/NaN
            accepted if selected UF <= 1.0
    '''
    if random_result_return:
        # Skip all calculations
        if random.choice([True, False, False, False, False, False, False]):
            return True, 'Check OK', x, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        else:
            return False, 'Random result', x, [1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    all_checks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print_result = False
    if getattr(obj, 'Girder', None) is not None:
        chk = _deactivate_non_prescriptive_girder_buckling(chk)

    calc_object = list(create_new_calc_obj(obj, x, fat_dict, fdwn=fdwn, fup=fup))
    calc_object[0].lat_press = lat_press

    # SemiAnalytical buckling check
    if chk[7]:
        if ml_results is not None:
            try:
                valid_prediction = int(ml_results[2]) if len(ml_results) > 2 else 0
                puls_acceptance = float(ml_results[3]) if len(ml_results) > 3 else 0.87

                if valid_prediction != 1:
                    return False, 'SemiAnalytical', x, all_checks

                puls_method = _puls_selected_method(calc_object[0].Plate.get_puls_method())
                if puls_method == 'buckling':
                    puls_uf = float(ml_results[0])
                elif puls_method == 'ultimate':
                    puls_uf = float(ml_results[1])
                else:
                    puls_uf = None
            except Exception:
                return False, 'SemiAnalytical', x, all_checks
        elif PULSrun is not None:
            x_id = x_to_string(x)
            puls_acceptance = PULSrun.puls_acceptance
            puls_method = _puls_selected_method(calc_object[0].Plate.get_puls_method())
            if puls_method == 'buckling':
                puls_uf = PULSrun.get_puls_line_results(x_id)["Buckling strength"]["Actual usage Factor"][0]
            elif puls_method == 'ultimate':
                puls_uf = PULSrun.get_puls_line_results(x_id)["Ultimate capacity"]["Actual usage Factor"][0]
            else:
                puls_uf = None
        else:
            return False, 'SemiAnalytical', x, all_checks

        if type(puls_uf) == str or puls_uf is None:
            return False, 'SemiAnalytical', x, all_checks

        all_checks[8] = puls_uf / puls_acceptance
        if puls_uf / puls_acceptance >= 1:
            if print_result:
                print('SemiAnalytical', calc_object[0].get_one_line_string(), False)
            return False, 'SemiAnalytical', x, all_checks

    if chk[8]:
        return False, 'Buckling ML-CL deactivated', x, all_checks

    # Buckling ML-Numeric
    if chk[9]:
        numeric_ok = False
        numeric_uf = float('inf')

        try:
            valid_prediction = int(ml_results[2]) if ml_results is not None and len(ml_results) > 2 else 0

            if valid_prediction == 1:
                puls_method = _puls_selected_method(calc_object[0].Plate.get_puls_method())
                if puls_method == 'buckling':
                    numeric_uf = float(ml_results[0])
                elif puls_method == 'ultimate':
                    numeric_uf = float(ml_results[1])
                else:
                    numeric_uf = float('inf')

                numeric_ok = numeric_uf <= 1.0
        except Exception:
            numeric_ok = False
            numeric_uf = float('inf')

        all_checks[10] = numeric_uf

        if not numeric_ok:
            if print_result:
                stf_text = calc_object[0].Stiffener.get_one_line_string() if calc_object[
                                                                                 0].Stiffener is not None else 'No stiffener'
                print('Buckling ML-Numeric', stf_text, False)
            return False, 'Buckling ML-Numeric', x, all_checks

    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    if init_weight != float('inf'):
        try:
            stiffener_type = (
                obj.Stiffener.get_stiffener_type()
                if obj.Stiffener is not None
                else 'T'
            )
        except Exception:
            stiffener_type = 'T'

        if cost_factors is not None:
            filter_value = calc_flat_objective_value(
                x,
                stiffener_type=stiffener_type,
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )
            filter_name = 'Cost filter'
        elif weld_bias >= 1.0:
            filter_value = calc_weld_objective(
                x,
                stiffener_type=stiffener_type,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
            )
            filter_name = 'Weld filter'
        else:
            filter_value = calc_weight(x)
            filter_name = 'Weight filter'

        if filter_value > init_weight:
            if init_weight == 0:
                all_checks[0] = float('inf')
            else:
                all_checks[0] = filter_value / init_weight
            return False, filter_name, x, all_checks

    # Section modulus
    if chk[0] and calc_object[0].Stiffener is not None:
        section_modulus = min(calc_object[0].Stiffener.get_section_modulus())
        min_section_modulus = calc_object[0].Stiffener.get_dnv_min_section_modulus(lat_press * 1000)
        section_frac = section_modulus / min_section_modulus
        all_checks[1] = section_frac
        if not section_modulus > min_section_modulus:
            if print_result:
                print('Section modulus', calc_object[0].get_one_line_string(), False)
            return False, 'Section modulus', x, all_checks

    # Local stiffener / CSR geometry check.
    #
    # Controlled only by chk[6].
    # For RP-C201 prescriptive buckling, this gives the RP-C201/CSR
    # web/flange geometry restrictions.
    #
    # For SemiAnalytical / ML-Numeric, no extra check is added here. If the GUI
    # keeps chk[6] enabled, the existing CSR requirement is applied through this
    # same check.
    if chk[6] and calc_object[0].Stiffener is not None:
        buckling_local = calc_object[0].local_buckling(optimizing=True)
        max_web_height = buckling_local['Stiffener'][0]
        max_flange_width = buckling_local['Stiffener'][1]

        web_ok = True if max_web_height == 0 else calc_object[0].Stiffener.hw <= max_web_height
        flange_ok = True if max_flange_width == 0 else calc_object[0].Stiffener.b <= max_flange_width
        check_items = [web_ok, flange_ok]

        local_fractions = [
            0 if max_web_height == 0 else calc_object[0].Stiffener.hw / max_web_height,
            0 if max_flange_width == 0 else calc_object[0].Stiffener.b / max_flange_width,
        ]

        if calc_object[0].Girder is not None:
            max_girder_web_height = buckling_local['Girder'][0]
            max_girder_flange_width = buckling_local['Girder'][1]
            girder_web_ok = True if max_girder_web_height == 0 else calc_object[0].Girder.hw <= max_girder_web_height
            girder_flange_ok = True if max_girder_flange_width == 0 else \
                calc_object[0].Girder.b <= max_girder_flange_width
            check_items.extend([girder_web_ok, girder_flange_ok])
            local_fractions.extend([
                0 if max_girder_web_height == 0 else calc_object[0].Girder.hw / max_girder_web_height,
                0 if max_girder_flange_width == 0 else calc_object[0].Girder.b / max_girder_flange_width,
            ])

        check = all(check_items)
        all_checks[2] = max(local_fractions)

        if not check:
            if print_result:
                print('Local stiffener buckling', calc_object[0].get_one_line_string(), False)
            return False, 'Local stiffener buckling', x, all_checks

    # Buckling
    if chk[3]:
        buckling_results = calc_object[0].plate_buckling(optimizing=True)
        res = [buckling_results['Plate']['Plate buckling'], ]
        for val in buckling_results['Stiffener'].values():
            res.append(val)
        for val in buckling_results['Girder'].values():
            res.append(val)
        buckling_results = res
        all_checks[3] = max(buckling_results)
        if not all([uf <= 1 for uf in buckling_results]):
            if print_result:
                print('Buckling', calc_object[0].get_one_line_string(), False)
            return False, 'Buckling', x, all_checks

    # Minimum plate thickness
    if chk[1]:
        act_pl_thk = calc_object[0].Plate.get_pl_thk()
        min_pl_thk = calc_object[0].Plate.get_dnv_min_thickness(lat_press * 1000) / 1000
        plate_frac = min_pl_thk / act_pl_thk
        all_checks[4] = plate_frac
        if not act_pl_thk > min_pl_thk:
            if print_result:
                print('Minimum plate thickeness', calc_object[0].get_one_line_string(), False)
            return False, 'Minimum plate thickness', x, all_checks

    # Shear area
    if chk[2]:
        pass

    # Fatigue
    if chk[4] and fat_dict is not None and fat_press is not None:
        fatigue_uf = calc_object[1].get_total_damage(ext_press=fat_press[0],
                                                     int_press=fat_press[1]) * calc_object[1].get_dff()
        all_checks[6] = fatigue_uf
        if fatigue_uf > 1:
            if print_result:
                print('Fatigue', calc_object[0].Stiffener.get_one_line_string(), False)
            return False, 'Fatigue', x, all_checks

    # Slamming
    if chk[5] and slamming_press != 0 and calc_object[0].Stiffener is not None:
        slam_check = calc_object[0].Stiffener.check_all_slamming(slamming_press)
        all_checks[7] = slam_check[1]
        if slam_check[0] is False:
            if print_result:
                print('Slamming', calc_object[0].Stiffener.get_one_line_string(), False)
            return False, 'Slamming', x, all_checks

    if print_result:
        stf_text = calc_object[0].Stiffener.get_one_line_string() if calc_object[
                                                                         0].Stiffener is not None else 'No stiffener'
        print('OK Section', stf_text, True)

    return True, 'Check OK', x, all_checks


def constraint_geometric(fractions, *args):
    return sum(fractions) == 1


def pso_constraint_geometric(x, *args):
    ''' The sum of the fractions must be 1.'''
    return 1 - sum(x)


def _get_mm_attr(obj, attr_name, default=0.0):
    """Return object attribute in mm."""
    if obj is None:
        return default
    try:
        value = getattr(obj, attr_name)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _set_mm_attr(obj, attr_name, value_mm):
    """Set object attribute in mm if possible."""
    if obj is None:
        return
    try:
        setattr(obj, attr_name, float(value_mm))
    except Exception:
        pass


def _get_longitudinal_spacing_mm(long_obj, panel_spacing_m):
    """Return longitudinal stiffener spacing in mm."""
    if long_obj is None:
        return float(panel_spacing_m) * 1000.0
    try:
        return float(long_obj.spacing)
    except Exception:
        pass
    try:
        return float(long_obj.s)
    except Exception:
        pass
    return float(panel_spacing_m) * 1000.0


def create_new_cylinder_obj(init_obj, x_new):
    """
    Create a new cylinder object for a candidate geometry.

    Optimizer tuple values are in meters. Structure stiffener properties are
    exposed as mm properties: spacing, hw, tw, b, tf. Older objects may use .s.
    """
    stress_press = [init_obj.sasd, init_obj.smsd, init_obj.tTsd, init_obj.tQsd, init_obj.shsd]
    shell_obj = init_obj.ShellObj
    long_obj = init_obj.LongStfObj

    old_long_spacing_mm = _get_longitudinal_spacing_mm(long_obj, init_obj.panel_spacing)
    x_old = (
        shell_obj.thk,
        shell_obj.radius,
        old_long_spacing_mm / 1000.0,
        0.0 if long_obj is None else _get_mm_attr(long_obj, 'hw') / 1000.0,
        0.0 if long_obj is None else _get_mm_attr(long_obj, 'tw') / 1000.0,
        0.0 if long_obj is None else _get_mm_attr(long_obj, 'b') / 1000.0,
        0.0 if long_obj is None else _get_mm_attr(long_obj, 'tf') / 1000.0,
    )

    x_new_stress_scaling = (
        x_new[0][0] if not np.isnan(x_new[0][0]) else shell_obj.thk,
        x_new[0][1] if not np.isnan(x_new[0][1]) else shell_obj.radius,
        x_new[0][5] if long_obj is None else x_new[1][0],
        0.0 if long_obj is None else x_new[1][2],
        0.0 if long_obj is None else x_new[1][3],
        0.0 if long_obj is None else x_new[1][4],
        0.0 if long_obj is None else x_new[1][5],
    )

    new_stresses = stress_scaling_cylinder(x_old, x_new_stress_scaling, stress_press)
    new_obj = copy.deepcopy(init_obj)
    new_obj.sasd, new_obj.smsd, new_obj.tTsd, new_obj.tQsd, new_obj.shsd = new_stresses
    new_obj.ShellObj.radius = x_new[0][1]
    new_obj.ShellObj.thk = x_new[0][0]

    if long_obj is None:
        new_obj.panel_spacing = x_new[0][5]
    else:
        _set_mm_attr(new_obj.LongStfObj, 'spacing', x_new[1][0] * 1000.0)
        _set_mm_attr(new_obj.LongStfObj, 'hw', x_new[1][2] * 1000.0)
        _set_mm_attr(new_obj.LongStfObj, 'tw', x_new[1][3] * 1000.0)
        _set_mm_attr(new_obj.LongStfObj, 'b', x_new[1][4] * 1000.0)
        _set_mm_attr(new_obj.LongStfObj, 'tf', x_new[1][5] * 1000.0)
        if hasattr(new_obj.LongStfObj, 's'):
            _set_mm_attr(new_obj.LongStfObj, 's', x_new[1][0] * 1000.0)

    return new_obj


def _candidate_float(x, index, default=0.0):
    try:
        value = float(x[index])
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def _candidate_has_girder_dimensions(x):
    return len(x) >= 12


def create_new_calc_obj(init_obj, x, fat_dict=None, fdwn=1, fup=0.5):
    '''
    Returns a new calculation object to be used in optimization
    :param init_obj:
    :return:
    '''

    if type(init_obj) == calc.AllStructure:
        plate = init_obj.Plate
        stiffener = init_obj.Stiffener
        girder = init_obj.Girder

        if stiffener is not None:
            x_old = (plate.get_s(), plate.get_pl_thk(), stiffener.get_web_h(), stiffener.get_web_thk(),
                     stiffener.get_fl_w(), stiffener.get_fl_thk(), plate.span,
                     stiffener.girder_lg if girder is None else girder.girder_lg, stiffener.stiffener_type)
        else:
            x_old = plate.get_tuple()

        sigma_y1_new = stress_scaling(plate.sigma_y1, plate.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
        sigma_y2_new = stress_scaling(plate.sigma_y2, plate.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
        tau_xy_new = stress_scaling(plate.tau_xy, plate.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
        sigma_x1_new = stress_scaling_area(plate.sigma_x1,
                                           sum(get_field_tot_area(x_old)),
                                           sum(get_field_tot_area(x)), fdwn=fdwn, fup=fup)
        sigma_x2_new = stress_scaling_area(plate.sigma_x2,
                                           sum(get_field_tot_area(x_old)),
                                           sum(get_field_tot_area(x)), fdwn=fdwn, fup=fup)
        default_stf_type = stiffener.get_stiffener_type() if stiffener is not None else 'T'
        stf_type = _get_stiffener_type_from_x(x, default=default_stf_type)

        main_dict = {'mat_yield': [plate.get_fy(), 'Pa'], 'mat_factor': [plate.mat_factor, ''],
                     'span': [plate.span, 'm'],
                     'spacing': [x[0], 'm'], 'plate_thk': [x[1], 'm'], 'stf_web_height': [x[2], 'm'],
                     'stf_web_thk': [x[3], 'm'], 'stf_flange_width': [x[4], 'm'],
                     'stf_flange_thk': [x[5], 'm'], 'structure_type': [plate.get_structure_type(), ''],
                     'stf_type': [stf_type, ''], 'sigma_y1': [sigma_y1_new, 'MPa'],
                     'sigma_y2': [sigma_y2_new, 'MPa'], 'sigma_x1': [sigma_x1_new, 'MPa'],
                     'sigma_x2': [sigma_x2_new, 'MPa'],
                     'tau_xy': [tau_xy_new, 'MPa'], 'plate_kpp': [plate.get_kpp(), ''],
                     'stf_kps': [plate.get_kps(), ''], 'stf_km1': [plate.get_km1(), ''],
                     'stf_km2': [plate.get_km2(), ''], 'stf_km3': [plate.get_km3(), ''],
                     'structure_types': [plate.get_structure_types(), ''],
                     'zstar_optimization': [plate.get_z_opt(), ''],
                     'puls buckling method': [plate.get_puls_method(), ''],
                     'puls boundary': [plate.get_puls_boundary(), ''],
                     'puls stiffener end': [plate.get_puls_stf_end(), ''],
                     'puls sp or up': [plate.get_puls_sp_or_up(), ''],
                     'puls up boundary': [plate.get_puls_up_boundary(), ''],
                     'panel or shell': [plate.panel_or_shell, ''],
                     'girder_lg': [x[7], 'm']}
        all_dict = copy.deepcopy(init_obj.get_main_properties())
        all_dict['Plate'] = main_dict
        all_dict['Stiffener'] = None if stiffener is None else main_dict

        if girder is None:
            all_dict['Girder'] = None
        else:
            girder_dict = copy.deepcopy(main_dict)
            girder_dict['spacing'] = [x[7], 'm']
            girder_dict['stf_web_height'] = [
                _candidate_float(x, 8, girder.get_web_h()) if _candidate_has_girder_dimensions(x)
                else girder.get_web_h(), 'm']
            girder_dict['stf_web_thk'] = [
                _candidate_float(x, 9, girder.get_web_thk()) if _candidate_has_girder_dimensions(x)
                else girder.get_web_thk(), 'm']
            girder_dict['stf_flange_width'] = [
                _candidate_float(x, 10, girder.get_fl_w()) if _candidate_has_girder_dimensions(x)
                else girder.get_fl_w(), 'm']
            girder_dict['stf_flange_thk'] = [
                _candidate_float(x, 11, girder.get_fl_thk()) if _candidate_has_girder_dimensions(x)
                else girder.get_fl_thk(), 'm']
            girder_dict['stf_type'] = [girder.get_stiffener_type(), '']
            all_dict['Girder'] = girder_dict

        if fat_dict == None:
            return calc.AllStructure(Plate=None if all_dict['Plate'] is None
            else calc.CalcScantlings(all_dict['Plate']),
                                     Stiffener=None if all_dict['Stiffener'] is None
                                     else calc.CalcScantlings(all_dict['Stiffener']),
                                     Girder=None if all_dict['Girder'] is None
                                     else calc.CalcScantlings(all_dict['Girder']),
                                     main_dict=all_dict['main dict']), None
        else:
            return calc.AllStructure(Plate=None if all_dict['Plate'] is None
            else calc.CalcScantlings(all_dict['Plate']),
                                     Stiffener=None if all_dict['Stiffener'] is None
                                     else calc.CalcScantlings(all_dict['Stiffener']),
                                     Girder=None if all_dict['Girder'] is None
                                     else calc.CalcScantlings(all_dict['Girder']),
                                     main_dict=all_dict['main dict']), \
                calc.CalcFatigue(main_dict, fat_dict)
    else:
        x_old = [init_obj.get_s(), init_obj.get_pl_thk(), init_obj.get_web_h(), init_obj.get_web_thk(),
                 init_obj.get_fl_w(), init_obj.get_fl_thk(), init_obj.span, init_obj.girder_lg]

        sigma_y1_new = stress_scaling(init_obj.sigma_y1, init_obj.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
        sigma_y2_new = stress_scaling(init_obj.sigma_y2, init_obj.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
        tau_xy_new = stress_scaling(init_obj.tau_xy, init_obj.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
        sigma_x1_new = stress_scaling_area(init_obj.sigma_x1,
                                           sum(get_field_tot_area(x_old)),
                                           sum(get_field_tot_area(x)), fdwn=fdwn, fup=fup)
        sigma_x2_new = stress_scaling_area(init_obj.sigma_x2,
                                           sum(get_field_tot_area(x_old)),
                                           sum(get_field_tot_area(x)), fdwn=fdwn, fup=fup)
        try:
            stf_type = x[8]
        except IndexError:
            stf_type = init_obj.get_stiffener_type()

        main_dict = {'mat_yield': [init_obj.get_fy(), 'Pa'], 'mat_factor': [init_obj.mat_factor, ''],
                     'span': [init_obj.span, 'm'],
                     'spacing': [x[0], 'm'], 'plate_thk': [x[1], 'm'], 'stf_web_height': [x[2], 'm'],
                     'stf_web_thk': [x[3], 'm'], 'stf_flange_width': [x[4], 'm'],
                     'stf_flange_thk': [x[5], 'm'], 'structure_type': [init_obj.get_structure_type(), ''],
                     'stf_type': [stf_type, ''], 'sigma_y1': [sigma_y1_new, 'MPa'],
                     'sigma_y2': [sigma_y2_new, 'MPa'], 'sigma_x1': [sigma_x1_new, 'MPa'],
                     'sigma_x2': [sigma_x2_new, 'MPa'],
                     'tau_xy': [tau_xy_new, 'MPa'], 'plate_kpp': [init_obj.get_kpp(), ''],
                     'stf_kps': [init_obj.get_kps(), ''], 'stf_km1': [init_obj.get_km1(), ''],
                     'stf_km2': [init_obj.get_km2(), ''], 'stf_km3': [init_obj.get_km3(), ''],
                     'structure_types': [init_obj.get_structure_types(), ''],
                     'zstar_optimization': [init_obj.get_z_opt(), ''],
                     'puls buckling method': [init_obj.get_puls_method(), ''],
                     'puls boundary': [init_obj.get_puls_boundary(), ''],
                     'puls stiffener end': [init_obj.get_puls_stf_end(), ''],
                     'puls sp or up': [init_obj.get_puls_sp_or_up(), ''],
                     'puls up boundary': [init_obj.get_puls_up_boundary(), ''],
                     'panel or shell': [init_obj.panel_or_shell, '']}
        if fat_dict == None:
            return calc.CalcScantlings(main_dict), None
        else:
            return calc.CalcScantlings(main_dict), calc.CalcFatigue(main_dict, fat_dict)


def create_new_structure_obj(init_obj, x, fat_dict=None, fdwn=1, fup=0.5):
    '''
    Returns a new calculation object to be used in optimization
    :param init_obj:
    :return:
    '''
    x_old = [init_obj.get_s(), init_obj.get_pl_thk(), init_obj.get_web_h(), init_obj.get_web_thk(),
             init_obj.get_fl_w(), init_obj.get_fl_thk(), init_obj.span, init_obj.girder_lg]

    sigma_y1_new = stress_scaling(init_obj.sigma_y1, init_obj.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
    sigma_y2_new = stress_scaling(init_obj.sigma_y2, init_obj.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
    tau_xy_new = stress_scaling(init_obj.tau_xy, init_obj.get_pl_thk(), x[1], fdwn=fdwn, fup=fup)
    sigma_x1_new = stress_scaling_area(init_obj.sigma_x1, sum(get_field_tot_area(x_old)), sum(get_field_tot_area(x)),
                                       fdwn=fdwn, fup=fup)
    sigma_x2_new = stress_scaling_area(init_obj.sigma_x2, sum(get_field_tot_area(x_old)), sum(get_field_tot_area(x)),
                                       fdwn=fdwn, fup=fup)

    try:
        stf_type = x[8]
    except IndexError:
        stf_type = init_obj.get_stiffener_type()

    main_dict = {'mat_yield': [init_obj.get_fy(), 'Pa'], 'span': [init_obj.span, 'm'],
                 'mat_factor': [init_obj.Plate.mat_factor, ''],
                 'spacing': [x[0], 'm'], 'plate_thk': [x[1], 'm'], 'stf_web_height': [x[2], 'm'],
                 'stf_web_thk': [x[3], 'm'], 'stf_flange_width': [x[4], 'm'],
                 'stf_flange_thk': [x[5], 'm'], 'structure_type': [init_obj.get_structure_type(), ''],
                 'stf_type': [stf_type, ''], 'sigma_y1': [sigma_y1_new, 'MPa'],
                 'sigma_y2': [sigma_y2_new, 'MPa'], 'sigma_x1': [sigma_x1_new, 'MPa'],
                 'sigma_x2': [sigma_x2_new, 'MPa'],
                 'tau_xy': [tau_xy_new, 'MPa'], 'plate_kpp': [init_obj.get_kpp(), ''],
                 'stf_kps': [init_obj.get_kps(), ''], 'stf_km1': [init_obj.get_km1(), ''],
                 'stf_km2': [init_obj.get_km2(), ''], 'stf_km3': [init_obj.get_km3(), ''],
                 'structure_types': [init_obj.get_structure_types(), ''],
                 'zstar_optimization': [init_obj.get_z_opt(), ''],
                 'puls buckling method': [init_obj.get_puls_method(), ''],
                 'puls boundary': [init_obj.get_puls_boundary(), ''],
                 'puls stiffener end': [init_obj.get_puls_stf_end(), ''],
                 'puls sp or up': [init_obj.get_puls_sp_or_up(), ''],
                 'puls up boundary': [init_obj.get_puls_up_boundary(), ''],
                 }

    # if fat_dict == None:
    return calc.Structure(main_dict)



def _get_stiffener_type_from_x(x, default='T'):
    for idx in (8, 7):
        try:
            stiffener_type = str(x[idx]).strip()
            if stiffener_type != '' and stiffener_type.lower() != 'nan':
                try:
                    float(stiffener_type)
                except Exception:
                    return stiffener_type
        except Exception:
            pass
    return default


def _get_plate_to_stiffener_weld_lines(stiffener_type):
    try:
        stf_type = str(stiffener_type).strip().lower()
    except Exception:
        return 2.0

    if stf_type in ('l', 'l-bulb', 'bulb', 'hp'):
        return 1.0

    return 2.0


def estimate_fillet_weld_leg(thickness_1, thickness_2,
                             min_leg=0.003,
                             max_leg=0.012,
                             thickness_factor=0.7):
    """Estimate fillet weld leg length [m] from connected material thickness."""
    try:
        t_min = min(float(thickness_1), float(thickness_2))
    except Exception:
        return min_leg
    leg = thickness_factor * t_min
    return min(max(leg, min_leg), max_leg)


def normalize_weld_metric(weld_metric):
    """Return supported weld objective metric key."""
    try:
        metric = str(weld_metric).strip().lower().replace('_', ' ')
    except Exception:
        return 'weld_consumables'

    if metric in ('length', 'weld length', 'weld_length'):
        return 'weld_length'

    return 'weld_consumables'


def normalize_cost_factors(cost_factors):
    """Return optimizer cost factors or None when cost optimization is inactive."""
    if cost_factors is None:
        return None

    if isinstance(cost_factors, dict):
        steel_cost = cost_factors.get('steel')
        weld_cost = cost_factors.get('weld')
    else:
        try:
            steel_cost, weld_cost = cost_factors
        except Exception:
            return None

    try:
        steel_cost = max(float(steel_cost), 0.0)
        weld_cost = max(float(weld_cost), 0.0)
    except Exception:
        return None

    if steel_cost <= 0.0 and weld_cost <= 0.0:
        return None

    return {'steel': steel_cost, 'weld': weld_cost}


def _uses_cost_objective(cost_factors):
    return normalize_cost_factors(cost_factors) is not None


def calc_weld_consumable(x, stiffener_type='T', density=7850.0, weld_area_factor=0.5,
                         include_plate_to_stiffener=True, include_web_to_flange=False):
    """Estimate weld consumable mass [kg] for one stiffened plate field."""
    try:
        spacing = float(x[0]); plate_thk = float(x[1]); web_h = float(x[2]); web_thk = float(x[3])
        fl_w = float(x[4]); fl_thk = float(x[5]); span = float(x[6]); width = float(x[7])
    except Exception:
        return float('inf')
    if spacing <= 0.0 or span <= 0.0 or width <= 0.0:
        return float('inf')
    if web_h <= 0.0 or web_thk <= 0.0:
        return 0.0
    stf_type = _get_stiffener_type_from_x(x, default=stiffener_type)
    number_of_stiffeners = estimate_number_of_stiffeners(width, spacing)
    weld_mass = 0.0
    if include_plate_to_stiffener:
        weld_lines_to_plate = _get_plate_to_stiffener_weld_lines(stf_type)
        plate_web_leg = estimate_fillet_weld_leg(plate_thk, web_thk)
        plate_web_area = weld_area_factor * plate_web_leg ** 2
        weld_mass += number_of_stiffeners * span * weld_lines_to_plate * plate_web_area * density
    if include_web_to_flange and fl_w > 0.0 and fl_thk > 0.0:
        flange_leg = estimate_fillet_weld_leg(web_thk, fl_thk)
        flange_area = weld_area_factor * flange_leg ** 2
        weld_mass += number_of_stiffeners * span * 2.0 * flange_area * density
    return weld_mass


def calc_weld_length(x, stiffener_type='T', include_plate_to_stiffener=True, include_web_to_flange=False):
    """Estimate total weld length [m] for one stiffened plate field."""
    try:
        spacing = float(x[0]); web_h = float(x[2]); web_thk = float(x[3])
        fl_w = float(x[4]); fl_thk = float(x[5]); span = float(x[6]); width = float(x[7])
    except Exception:
        return float('inf')
    if spacing <= 0.0 or span <= 0.0 or width <= 0.0:
        return float('inf')
    if web_h <= 0.0 or web_thk <= 0.0:
        return 0.0

    stf_type = _get_stiffener_type_from_x(x, default=stiffener_type)
    number_of_stiffeners = estimate_number_of_stiffeners(width, spacing)
    weld_length = 0.0

    if include_plate_to_stiffener:
        weld_lines_to_plate = _get_plate_to_stiffener_weld_lines(stf_type)
        weld_length += number_of_stiffeners * span * weld_lines_to_plate

    if include_web_to_flange and fl_w > 0.0 and fl_thk > 0.0:
        weld_length += number_of_stiffeners * span * 2.0

    return weld_length


def calc_weld_objective(x, stiffener_type='T', include_web_to_flange=False, weld_metric='weld_consumables'):
    """Return selected flat-panel weld metric."""
    if normalize_weld_metric(weld_metric) == 'weld_length':
        return calc_weld_length(
            x,
            stiffener_type=stiffener_type,
            include_web_to_flange=include_web_to_flange,
        )

    return calc_weld_consumable(
        x,
        stiffener_type=stiffener_type,
        include_web_to_flange=include_web_to_flange,
    )


def calc_flat_objective_value(x, stiffener_type='T', weld_bias=0.0, include_web_to_flange=False,
                              weld_metric='weld_consumables', cost_factors=None):
    """Return the scalar flat-panel objective used for filtering and winner selection."""
    cost_factors = normalize_cost_factors(cost_factors)
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0

    weight = calc_weight(x)

    if cost_factors is None and weld_bias <= 0.0:
        return weight

    weld_value = calc_weld_objective(
        x,
        stiffener_type=stiffener_type,
        include_web_to_flange=include_web_to_flange,
        weld_metric=weld_metric,
    )

    if cost_factors is not None:
        return cost_factors['steel'] * weight + cost_factors['weld'] * weld_value

    if weld_bias >= 1.0:
        return weld_value

    return (1.0 - weld_bias) * weight + weld_bias * weld_value


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def _component_has_stiffener_geometry(component):
    if component is None:
        return False
    try:
        web_h = _safe_float(component[2]); web_thk = _safe_float(component[3])
    except Exception:
        return False
    return web_h > 0.0 and web_thk > 0.0


def calc_weld_consumable_cylinder(x, density=7850.0, weld_area_factor=0.5,
                                  include_web_to_flange=False):
    """Estimate weld consumable mass [kg] for a stiffened cylinder candidate."""
    try:
        shell = x[0]; long_stf = x[1]; ring_stf = x[2]; ring_frame = x[3]
    except Exception:
        return float('inf')
    shell_thk = _safe_float(shell[0]); radius = _safe_float(shell[1])
    ring_stf_spacing = _safe_float(shell[2]); ring_frame_spacing = _safe_float(shell[3])
    cylinder_length = _safe_float(shell[4])
    if shell_thk <= 0.0 or radius <= 0.0 or cylinder_length <= 0.0:
        return float('inf')
    circumference = 2.0 * math.pi * radius
    weld_mass = 0.0
    if _component_has_stiffener_geometry(long_stf):
        long_spacing = _safe_float(long_stf[0]); long_web_thk = _safe_float(long_stf[3])
        long_fl_w = _safe_float(long_stf[4]); long_fl_thk = _safe_float(long_stf[5])
        if long_spacing > 0.0:
            num_long_stf = circumference / long_spacing
            leg = estimate_fillet_weld_leg(shell_thk, long_web_thk)
            weld_area = weld_area_factor * leg ** 2
            weld_lines_to_shell = _get_plate_to_stiffener_weld_lines(_get_stiffener_type_from_x(long_stf))
            weld_mass += num_long_stf * cylinder_length * weld_lines_to_shell * weld_area * density
            if include_web_to_flange and long_fl_w > 0.0 and long_fl_thk > 0.0:
                flange_leg = estimate_fillet_weld_leg(long_web_thk, long_fl_thk)
                flange_area = weld_area_factor * flange_leg ** 2
                weld_mass += num_long_stf * cylinder_length * 2.0 * flange_area * density
    if _component_has_stiffener_geometry(ring_stf) and ring_stf_spacing > 0.0:
        ring_web_h = _safe_float(ring_stf[2]); ring_web_thk = _safe_float(ring_stf[3])
        ring_fl_w = _safe_float(ring_stf[4]); ring_fl_thk = _safe_float(ring_stf[5])
        num_ring_stf = cylinder_length / ring_stf_spacing
        leg = estimate_fillet_weld_leg(shell_thk, ring_web_thk)
        weld_area = weld_area_factor * leg ** 2
        weld_lines_to_shell = _get_plate_to_stiffener_weld_lines(_get_stiffener_type_from_x(ring_stf))
        weld_mass += num_ring_stf * circumference * weld_lines_to_shell * weld_area * density
        if include_web_to_flange and ring_fl_w > 0.0 and ring_fl_thk > 0.0:
            flange_radius = max(radius - ring_web_h, 0.0)
            flange_circumference = 2.0 * math.pi * flange_radius
            flange_leg = estimate_fillet_weld_leg(ring_web_thk, ring_fl_thk)
            flange_area = weld_area_factor * flange_leg ** 2
            weld_mass += num_ring_stf * flange_circumference * 2.0 * flange_area * density
    if _component_has_stiffener_geometry(ring_frame) and ring_frame_spacing > 0.0:
        frame_web_h = _safe_float(ring_frame[2]); frame_web_thk = _safe_float(ring_frame[3])
        frame_fl_w = _safe_float(ring_frame[4]); frame_fl_thk = _safe_float(ring_frame[5])
        num_ring_frames = cylinder_length / ring_frame_spacing
        leg = estimate_fillet_weld_leg(shell_thk, frame_web_thk)
        weld_area = weld_area_factor * leg ** 2
        weld_lines_to_shell = _get_plate_to_stiffener_weld_lines(_get_stiffener_type_from_x(ring_frame))
        weld_mass += num_ring_frames * circumference * weld_lines_to_shell * weld_area * density
        if include_web_to_flange and frame_fl_w > 0.0 and frame_fl_thk > 0.0:
            flange_radius = max(radius - frame_web_h, 0.0)
            flange_circumference = 2.0 * math.pi * flange_radius
            flange_leg = estimate_fillet_weld_leg(frame_web_thk, frame_fl_thk)
            flange_area = weld_area_factor * flange_leg ** 2
            weld_mass += num_ring_frames * flange_circumference * 2.0 * flange_area * density
    return weld_mass


def calc_weld_length_cylinder(x, include_web_to_flange=False):
    """Estimate total weld length [m] for a stiffened cylinder candidate."""
    try:
        shell = x[0]; long_stf = x[1]; ring_stf = x[2]; ring_frame = x[3]
    except Exception:
        return float('inf')

    shell_thk = _safe_float(shell[0]); radius = _safe_float(shell[1])
    ring_stf_spacing = _safe_float(shell[2]); ring_frame_spacing = _safe_float(shell[3])
    cylinder_length = _safe_float(shell[4])
    if shell_thk <= 0.0 or radius <= 0.0 or cylinder_length <= 0.0:
        return float('inf')

    circumference = 2.0 * math.pi * radius
    weld_length = 0.0

    if _component_has_stiffener_geometry(long_stf):
        long_spacing = _safe_float(long_stf[0])
        long_fl_w = _safe_float(long_stf[4]); long_fl_thk = _safe_float(long_stf[5])
        if long_spacing > 0.0:
            num_long_stf = circumference / long_spacing
            weld_lines_to_shell = _get_plate_to_stiffener_weld_lines(_get_stiffener_type_from_x(long_stf))
            weld_length += num_long_stf * cylinder_length * weld_lines_to_shell
            if include_web_to_flange and long_fl_w > 0.0 and long_fl_thk > 0.0:
                weld_length += num_long_stf * cylinder_length * 2.0

    if _component_has_stiffener_geometry(ring_stf) and ring_stf_spacing > 0.0:
        ring_fl_w = _safe_float(ring_stf[4]); ring_fl_thk = _safe_float(ring_stf[5])
        num_ring_stf = cylinder_length / ring_stf_spacing
        weld_lines_to_shell = _get_plate_to_stiffener_weld_lines(_get_stiffener_type_from_x(ring_stf))
        weld_length += num_ring_stf * circumference * weld_lines_to_shell
        if include_web_to_flange and ring_fl_w > 0.0 and ring_fl_thk > 0.0:
            weld_length += num_ring_stf * circumference * 2.0

    if _component_has_stiffener_geometry(ring_frame) and ring_frame_spacing > 0.0:
        frame_fl_w = _safe_float(ring_frame[4]); frame_fl_thk = _safe_float(ring_frame[5])
        num_ring_frames = cylinder_length / ring_frame_spacing
        weld_lines_to_shell = _get_plate_to_stiffener_weld_lines(_get_stiffener_type_from_x(ring_frame))
        weld_length += num_ring_frames * circumference * weld_lines_to_shell
        if include_web_to_flange and frame_fl_w > 0.0 and frame_fl_thk > 0.0:
            weld_length += num_ring_frames * circumference * 2.0

    return weld_length


def calc_weld_objective_cylinder(x, include_web_to_flange=False, weld_metric='weld_consumables'):
    """Return selected cylinder weld metric."""
    if normalize_weld_metric(weld_metric) == 'weld_length':
        return calc_weld_length_cylinder(
            x,
            include_web_to_flange=include_web_to_flange,
        )

    return calc_weld_consumable_cylinder(
        x,
        include_web_to_flange=include_web_to_flange,
    )


def calc_cylinder_objective_value(x, weld_bias=0.0, include_web_to_flange=False,
                                  weld_metric='weld_consumables', cost_factors=None):
    """Return the scalar cylinder objective used for filtering and winner selection."""
    cost_factors = normalize_cost_factors(cost_factors)
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0

    weight = calc_weight_cylinder(x)

    if cost_factors is None and weld_bias <= 0.0:
        return weight

    weld_value = calc_weld_objective_cylinder(
        x,
        include_web_to_flange=include_web_to_flange,
        weld_metric=weld_metric,
    )

    if cost_factors is not None:
        return cost_factors['steel'] * weight + cost_factors['weld'] * weld_value

    if weld_bias >= 1.0:
        return weld_value

    return (1.0 - weld_bias) * weight + weld_bias * weld_value



def estimate_number_of_stiffeners(width, spacing):
    """
    Estimate number of stiffeners across Lg/field width.

    Uses round(width / spacing) rather than floor division to avoid systematic
    under-counting when Lg is not an exact multiple of stiffener spacing.
    """
    try:
        width = float(width)
        spacing = float(spacing)
    except Exception:
        return 0

    if width <= 0.0 or spacing <= 0.0:
        return 0

    return max(int(round(width / spacing)), 1)

def get_field_tot_area(x):
    ''' Total area of a plate field. '''

    if len(x) == 6:
        width = 10
    else:
        width = x[7]
    plate_area = width * x[1]
    stiff_area = (x[2] * x[3] + x[4] * x[5]) * estimate_number_of_stiffeners(width, x[0])

    return plate_area, stiff_area


def calc_weight(x, prt=False):
    '''
    Calculating the current weight
    :param current_dict:
    :return:
    '''
    span = x[6]
    plate_area, stiff_area = get_field_tot_area(x)
    girder_area = 0.0
    girder_length = x[7] if len(x) > 7 else 0.0
    if _candidate_has_girder_dimensions(x):
        girder_area = x[8] * x[9] + x[10] * x[11]

    if prt:
        print('x is', x, 'plate area', plate_area, 'stiff area', stiff_area, 'weight',
              span * 7850 * (plate_area + stiff_area) + girder_length * 7850 * girder_area)
    return span * 7850 * (plate_area + stiff_area) + girder_length * 7850 * girder_area


def calc_weight_pso(x, *args):
    '''
    Calculating the current weight
    :param current_dict:
    :return:
    '''

    width = args[5]
    span = args[6]

    plate_area = width * x[1]
    stiff_area = (x[2] * x[3] + x[4] * x[5]) * estimate_number_of_stiffeners(width, x[0])
    return span * 7850 * (plate_area + stiff_area)


def calc_weight_pso_section(x, *args):
    '''
    Calculating the weight of a complete section.
    :param x:
    :param args:
    :return:
    '''
    stru_objects = args[1]
    tot_length = args[2]
    frame_height = args[3]
    frame_section_area = args[4]

    tot_weight = 0

    for dummy_i in range(len(stru_objects)):
        tot_weight += frame_section_area * frame_height * 7850

    count = 0
    for stru_object in stru_objects:
        span = tot_length * x[count]
        stru_object.Plate.set_span(span)
        stru_object.Stiffener.set_span(span)
        tot_weight += stru_object.Stiffener.get_weight_width_lg()

    return tot_weight


def calc_weight_cylinder(x):
    '''
    Calculation of total weigth.

    shell       (0.02, 2.5, 5, 5, 10, nan, nan, nan),
    long        (0.875, nan, 0.3, 0.01, 0.1, 0.01, nan, nan),
    ring        (nan, nan, 0.3, 0.01, 0.1, 0.01, nan, nan),
    ring        (nan, nan, 0.7, 0.02, 0.2, 0.02, nan, nan)]
    '''
    if sum(x[1][0:8]) != 0:
        num_long_stf = 2 * math.pi * x[0][1] / x[1][0]
        long_stf_area = x[1][2] * x[1][3] + x[1][4] * x[1][5]
        long_stf_volume = long_stf_area * x[0][4] * num_long_stf
    else:
        long_stf_volume = 0
    if sum(x[2][0:8]) != 0:
        num_ring_stf = x[0][4] / x[0][2]
        ring_stf_volume = math.pi * (math.pow(x[0][1], 2) - math.pow(x[0][1] - x[2][2], 2)) * x[2][3] + \
                          2 * math.pi * (x[0][1] - x[2][2]) * x[2][4] * x[2][5]
        ring_stf_tot_vol = ring_stf_volume * num_ring_stf
    else:
        ring_stf_tot_vol = 0
    if sum(x[3][0:8]) != 0:
        num_ring_girder = x[0][4] / x[0][3]
        ring_frame_volume = math.pi * (math.pow(x[0][1], 2) - math.pow(x[0][1] - x[3][2], 2)) * x[3][3] + \
                            2 * math.pi * (x[0][1] - x[3][2]) * x[3][4] * x[3][5]
        tot_ring_frame_vol = ring_frame_volume * num_ring_girder
    else:
        tot_ring_frame_vol = 0

    shell_volume = 2 * math.pi * x[0][1] * x[0][0] * x[0][4]

    return (long_stf_volume + ring_stf_tot_vol + tot_ring_frame_vol + shell_volume) * 7850


def stress_scaling_cylinder(x1, x2, stress1):
    '''
    Scale stresses of a stiffened cylinder.

    To scale:

    Design axial stress,          sa,sd =
    Design bending stress,   sm,sd =
    Design torsional stress,   tT,sd=
    Design shear stress,        tQ,sd=
    Additional hoop stress, sh,sd =

    '''

    t1, r1, s1, hw1, tw1, b1, tf1 = x1
    t2, r2, s2, hw2, tw2, b2, tf2 = x2

    sasd1, smsd1, tTsd1, tQsd1, shsd1 = stress1

    A1 = hw1 * tw1 + b1 * tf1
    A2 = hw2 * tw2 + b2 * tf2
    # Axial stress changes by equivalent thickness

    thk_eq1 = t1 + 0 if s1 == 0 else A1 / s1
    thk_eq2 = t2 + 0 if s2 == 0 else A2 / s2

    # Moment stress changes by difference in moment of inertia

    Itot1 = calc.CylinderAndCurvedPlate.get_Itot(hw=hw1, tw=tw1, b=b1, tf=tf1, r=r1, s=s1, t=t1)
    Itot2 = calc.CylinderAndCurvedPlate.get_Itot(hw=hw2, tw=tw2, b=b2, tf=tf2, r=r2, s=s2, t=t2)

    # Torsional, shear and hoop changes by cylinder thickness.

    return sasd1 * (thk_eq1 / thk_eq2), smsd1 * (Itot1 / Itot2), tTsd1 * (t1 / t2), tQsd1 * (t1 / t2), shsd1 * (t1 / t2)


def stress_scaling(sigma_old, t_old, t_new, fdwn=1, fup=0.5):
    if t_new <= t_old:  # decreasing the thickness
        sigma_new = sigma_old * (t_old / (t_old - fdwn * abs((t_old - t_new))))
        # assert sigma_new >= sigma_old, 'ERROR no stress increase: \n' \
        #                               't_old '+str(t_old)+' sigma_old '+str(sigma_old)+ \
        #                               '\nt_new '+str(t_new)+' sigma_new '+str(sigma_new)

    else:  # increasing the thickness

        sigma_new = sigma_old * (t_old / (t_old + fup * abs((t_old - t_new))))
        # assert sigma_new <= sigma_old, 'ERROR no stress reduction: \n' \
        #                               't_old '+str(t_old)+' sigma_old '+str(sigma_old)+ \
        #                               '\nt_new '+str(t_new)+' sigma_new '+str(sigma_new)
    return sigma_new


def stress_scaling_area(sigma_old, a_old, a_new, fdwn=1, fup=0.5):
    ''' Scale stresses using input area '''

    if a_new <= a_old:  # decreasing the thickness
        sigma_new = sigma_old * (a_old / (a_old - fdwn * abs((a_old - a_new))))
        # assert sigma_new >= sigma_old, 'ERROR no stress increase: \n' \
        #                               't_old '+str(a_old)+' sigma_old '+str(sigma_old)+ \
        #                               '\nt_new '+str(a_new)+' sigma_new '+str(sigma_new)
    else:  # increasing the thickness
        sigma_new = sigma_old * (a_old / (a_old + fup * abs((a_old - a_new))))
        # assert sigma_new <= sigma_old, 'ERROR no stress reduction: \n' \
        #                               't_old '+str(a_old)+' sigma_old '+str(sigma_old)+ \
        #                               '\nt_new '+str(a_new)+' sigma_new '+str(sigma_new)
    # print('a_old', a_old, 'sigma_old', sigma_old, '|', 'a_new', a_new, 'sigma_new',sigma_new)
    return sigma_new


def x_to_string(x):
    ret = ''
    for val in x:
        ret += str(val) + '_'
    return ret



def _semianalytical_speed_staging_candidates(iterable_all, init_stuc_obj, lat_press, init_filter_weight, side,
                                             chk, fat_dict, fat_press, slamming_press, fdwn, fup,
                                             weld_bias, builtup_stiffener, weld_metric, cost_factors):
    """
    Return (surviving_candidates, early_rejects) for SemiAnalytical optimization.

    This helper deliberately uses any_constraints_all as the cheap-check oracle
    with SemiAnalytical/ML/fatigue/slamming/heavy buckling checks disabled. The
    final decision for every survivor is still made by any_constraints_all.
    """
    pre_chk = list(chk)
    while len(pre_chk) < 10:
        pre_chk.append(False)
    for disabled_idx in (3, 4, 5, 7, 8, 9):
        pre_chk[disabled_idx] = False

    candidates = []
    rejects = []
    for x in iterable_all:
        precheck = any_constraints_all(
            x,
            init_stuc_obj,
            lat_press,
            init_filter_weight,
            side,
            tuple(pre_chk),
            fat_dict,
            fat_press,
            slamming_press,
            None,
            False,
            fdwn,
            fup,
            None,
            False,
            weld_bias,
            builtup_stiffener,
            weld_metric,
            cost_factors,
        )
        if precheck[0]:
            candidates.append(x)
        else:
            rejects.append(precheck)
    return candidates, rejects

def get_filtered_results(iterable_all, init_stuc_obj, lat_press, init_filter_weight, side='p',
                         chk=(True, True, True, True, True, True, True, False), fat_dict=None, fat_press=None,
                         slamming_press=None, processes=None, puls_sheet=None, puls_acceptance=0.87,
                         fdwn=1, fup=0.5, ml_algo=None, weld_bias=0.0, builtup_stiffener=False,
                         weld_metric='weld_consumables', cost_factors=None,
                         use_semianalytical_speed_staging=DEFAULT_USE_SEMIANALYTICAL_SPEED_STAGING):
    '''
    Using multiprocessing to return list of applicable results.

    Supports:
        chk[7] = built-in SemiAnalytical replacement
        chk[8] = ML-CL classification pipeline, deactivated
        chk[9] = ML-Numeric UF pipeline
    '''

    if len(chk) > 8 and chk[8]:
        raise NotImplementedError('ML-CL buckling is deactivated. Use ML-Numeric or SemiAnalytical.')

    iterable_all = list(iterable_all)
    semianalytical_precheck_rejects = []
    semianalytical_candidates = iterable_all

    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    if chk[7]:
        # Built-in SemiAnalytical replacement. Columns: buckling UF, ultimate UF,
        # valid prediction, acceptance limit. Optional staging uses
        # any_constraints_all as the cheap-check oracle, and final survivor
        # decisions still pass through any_constraints_all below.
        if use_semianalytical_speed_staging:
            semianalytical_candidates, semianalytical_precheck_rejects = _semianalytical_speed_staging_candidates(
                iterable_all,
                init_stuc_obj,
                lat_press,
                init_filter_weight,
                side,
                chk,
                fat_dict,
                fat_press,
                slamming_press,
                fdwn,
                fup,
                weld_bias,
                builtup_stiffener,
                weld_metric,
                cost_factors,
            )
        else:
            semianalytical_candidates = iterable_all
            semianalytical_precheck_rejects = []

        sort_again = np.full([len(semianalytical_candidates), 4], np.inf, dtype=float)
        sort_again[:, 2] = 0.0
        sort_again[:, 3] = float(puls_acceptance)

        to_run = []
        for x in semianalytical_candidates:
            calc_object_stf = None if init_stuc_obj.Stiffener is None else create_new_calc_obj(init_stuc_obj.Stiffener,
                                                                                               x, fat_dict,
                                                                                               fdwn=fdwn, fup=fup)
            calc_object_pl = create_new_calc_obj(init_stuc_obj.Plate, x, fat_dict, fdwn=fdwn, fup=fup)
            calc_object = [calc.AllStructure(Plate=calc_object_pl[0],
                                             Stiffener=None if init_stuc_obj.Stiffener is None else calc_object_stf[0],
                                             Girder=None,
                                             main_dict=init_stuc_obj.get_main_properties()['main dict']),
                           calc_object_pl[1]]
            to_run.append((calc_object, x, lat_press))

        if len(to_run) > 0 and hasattr(semi_analytical, 'predict_anystructure_uf_batch'):
            sort_again[:, 0:4] = semi_analytical.predict_anystructure_uf_batch(
                to_run,
                default_acceptance=puls_acceptance,
                cache={},
            )
        else:
            local_cache = {}
            for idx, (calc_object, x, this_lat_press) in enumerate(to_run):
                sort_again[idx, 0:4] = _predict_semi_analytical_uf(
                    calc_object,
                    this_lat_press,
                    puls_acceptance,
                    cache=local_cache,
                )

        PULSrun = None

    elif chk[8]:
        # ML-CL is deactivated.
        sp_int, sp_gl_gt, up_int, up_gl_gt = list(), list(), list(), list()
        sp_int_idx, sp_gl_gt_idx, up_int_idx, up_gl_gt_idx = list(), list(), list(), list()

        for idx, x in enumerate(iterable_all):
            calc_object_stf = None if init_stuc_obj.Stiffener is None else create_new_calc_obj(init_stuc_obj.Stiffener,
                                                                                               x, fat_dict,
                                                                                               fdwn=fdwn, fup=fup)
            calc_object_pl = create_new_calc_obj(init_stuc_obj.Plate, x, fat_dict, fdwn=fdwn, fup=fup)
            calc_object = [calc.AllStructure(Plate=calc_object_pl[0],
                                             Stiffener=None if init_stuc_obj.Stiffener is None else calc_object_stf[0],
                                             Girder=None,
                                             main_dict=init_stuc_obj.get_main_properties()['main dict']),
                           calc_object_pl[1]]

            if calc_object[0].Plate.get_puls_sp_or_up() == 'UP':
                if _is_integrated_puls_boundary(calc_object[0].Plate.get_puls_boundary()):
                    up_int.append(calc_object[0].Plate.get_buckling_ml_input(lat_press, alone=False))
                    up_int_idx.append(idx)
                else:
                    up_gl_gt.append(calc_object[0].Plate.get_buckling_ml_input(lat_press, alone=False))
                    up_gl_gt_idx.append(idx)
            else:
                ml_input = _get_ml_input_for_optimization(calc_object, lat_press)
                if _is_integrated_puls_boundary(calc_object[0].Stiffener.get_puls_boundary() if calc_object[0].Stiffener is not None else calc_object[0].Plate.get_puls_boundary()):
                    sp_int.append(ml_input)
                    sp_int_idx.append(idx)
                else:
                    sp_gl_gt.append(ml_input)
                    sp_gl_gt_idx.append(idx)

        # Existing classification result container.
        sort_again = np.zeros([len(iterable_all), 2])

        if len(sp_int) != 0:
            sp_int_res = [ml_algo['cl SP buc int predictor'].predict(ml_algo['cl SP buc int scaler']
                                                                     .transform(sp_int)),
                          ml_algo['cl SP ult int predictor'].predict(ml_algo['cl SP ult int scaler']
                                                                     .transform(sp_int))]
            for idx, res_buc, res_ult in zip(sp_int_idx, sp_int_res[0], sp_int_res[1]):
                sort_again[idx] = [res_buc, res_ult]

        if len(sp_gl_gt) != 0:
            sp_gl_gt_res = [ml_algo['cl SP buc GLGT predictor'].predict(ml_algo['cl SP buc GLGT scaler']
                                                                        .transform(sp_gl_gt)),
                            ml_algo['cl SP ult GLGT predictor'].predict(ml_algo['cl SP ult GLGT scaler']
                                                                        .transform(sp_gl_gt))]
            for idx, res_buc, res_ult in zip(sp_gl_gt_idx, sp_gl_gt_res[0], sp_gl_gt_res[1]):
                sort_again[idx] = [res_buc, res_ult]

        if len(up_int) != 0:
            up_int_res = [ml_algo['cl UP buc int predictor'].predict(ml_algo['cl UP buc int scaler']
                                                                     .transform(up_int)),
                          ml_algo['cl UP ult int predictor'].predict(ml_algo['cl UP ult int scaler']
                                                                     .transform(up_int))]
            for idx, res_buc, res_ult in zip(up_int_idx, up_int_res[0], up_int_res[1]):
                sort_again[idx] = [res_buc, res_ult]

        if len(up_gl_gt) != 0:
            up_gl_gt_res = [ml_algo['cl UP buc GLGT predictor'].predict(ml_algo['cl UP buc GLGT scaler']
                                                                        .transform(up_gl_gt)),
                            ml_algo['cl UP ult GLGT predictor'].predict(ml_algo['cl UP ult GLGT scaler']
                                                                        .transform(up_gl_gt))]
            for idx, res_buc, res_ult in zip(up_gl_gt_idx, up_gl_gt_res[0], up_gl_gt_res[1]):
                sort_again[idx] = [res_buc, res_ult]

        PULSrun = None

    elif chk[9]:
        # ML-Numeric UF to be used.
        numeric_groups = {
            'num SP int': {'inputs': [], 'indices': []},
            'num SP GLGT': {'inputs': [], 'indices': []},
            'num UP int': {'inputs': [], 'indices': []},
            'num UP GLGT': {'inputs': [], 'indices': []},
        }

        for idx, x in enumerate(iterable_all):
            calc_object_stf = None if init_stuc_obj.Stiffener is None else create_new_calc_obj(init_stuc_obj.Stiffener,
                                                                                               x, fat_dict,
                                                                                               fdwn=fdwn, fup=fup)
            calc_object_pl = create_new_calc_obj(init_stuc_obj.Plate, x, fat_dict, fdwn=fdwn, fup=fup)
            calc_object = [calc.AllStructure(Plate=calc_object_pl[0],
                                             Stiffener=None if init_stuc_obj.Stiffener is None else calc_object_stf[0],
                                             Girder=None,
                                             main_dict=init_stuc_obj.get_main_properties()['main dict']),
                           calc_object_pl[1]]

            prefix = _get_numeric_pipeline_prefix(calc_object)
            ml_input = _get_ml_input_for_optimization(calc_object, lat_press)
            numeric_groups[prefix]['inputs'].append(ml_input)
            numeric_groups[prefix]['indices'].append(idx)

        # columns: buckling UF, ultimate UF, valid prediction
        sort_again = np.full([len(iterable_all), 3], np.inf, dtype=float)
        sort_again[:, 2] = 0.0

        try:
            mat_fac = float(init_stuc_obj.Plate.mat_factor)
        except Exception:
            mat_fac = 1.15

        for prefix, group in numeric_groups.items():
            if len(group['inputs']) == 0:
                continue

            group_res = _predict_numeric_uf_group(
                ml_algo=ml_algo,
                input_rows=group['inputs'],
                prefix=prefix,
                mat_fac=mat_fac,
            )

            for idx, res in zip(group['indices'], group_res):
                sort_again[idx] = res

        PULSrun = None

    else:
        PULSrun = None
        for _ in iterable_all:
            pass
        sort_again = None

    iter_var = list()
    final_iterable = semianalytical_candidates if chk[7] else iterable_all
    for idx, item in enumerate(final_iterable):
        if chk[7] or chk[8] or chk[9]:
            this_ml_result = sort_again[idx]
        else:
            this_ml_result = None

        iter_var.append((item, init_stuc_obj, lat_press, init_filter_weight, side, chk, fat_dict, fat_press,
                         slamming_press, PULSrun, False, fdwn, fup, this_ml_result, False, weld_bias,
                         builtup_stiffener, weld_metric, cost_factors))

    iter_var = tuple(iter_var)

    if processes is None:
        processes = max(cpu_count() - 1, 1)

    if processes == 1:
        res_pre = [any_constraints_all(*args) for args in iter_var]
    else:
        with Pool(processes) as my_process:
            res_pre = my_process.starmap(any_constraints_all, iter_var)

    check_ok, check_not_ok = list(), list(semianalytical_precheck_rejects)
    for item in res_pre:
        if item[0] is False:
            check_not_ok.append(item)
        else:
            check_ok.append(item)

    return check_ok, check_not_ok



def _is_defined_number(value):
    """
    Return True for usable numeric optimizer values.

    None and NaN are treated as not defined. Some stiffener tuples use NaN
    for unused fields.
    """
    if value is None:
        return False
    try:
        return not np.isnan(float(value))
    except Exception:
        return False


def _candidate_value_within_bounds(value, lower, upper, tol=1e-12):
    """
    Check one numeric candidate value against optional lower/upper bounds.
    """
    if not _is_defined_number(value):
        return True

    value = float(value)

    if _is_defined_number(lower) and value < float(lower) - tol:
        return False

    if _is_defined_number(upper) and value > float(upper) + tol:
        return False

    return True


def _predefined_stiffener_within_bounds(predef_tuple, min_var, max_var):
    """
    Check whether a predefined stiffener respects the optimizer bounds.

    In the predefined-stiffener branch, spacing and plate thickness are still
    iterated from the GUI bounds. The predefined section itself must still
    respect the GUI limits for:
        web height
        web thickness
        flange width
        flange thickness
    """
    for idx in (2, 3, 4, 5):
        if idx >= len(predef_tuple) or idx >= len(min_var) or idx >= len(max_var):
            continue

        if not _candidate_value_within_bounds(predef_tuple[idx], min_var[idx], max_var[idx]):
            return False

    return True


def any_get_all_combs(min_var, max_var, deltas, init_weight=float('inf'), predef_stiffeners=None, stf_type=None):
    '''
    Calulating initial values.
    :param min:
    :param max:
    :return:
    '''
    '''
    shell_upper_bounds = np.array( [0.03, 2.5, 5, 0.8, 6, 6])
    shell_deltas = np.array(       [0.01, 2.5, 1, 0.1, 1, 1])
    shell_lower_bounds = np.array( [0.02, 2.5, 5, 0.6, 4, 4])

    long_upper_bounds = np.array(   [0.875, None, 0.5, 0.018, 0.2, 0.03])
    long_deltas = np.array(         [0.025, None, 0.1, 0.004, 0.05, 0.005])
    long_lower_bounds = np.array(   [0.875, None, 0.3,  0.010, 0.1, 0.010])

    ring_stf_upper_bounds = np.array(   [None, None, 0.5, 0.018, 0.2, 0.03])
    ring_stf_deltas = np.array(         [None, None, 0.1, 0.004, 0.05, 0.005])
    ring_stf_lower_bounds = np.array(   [None, None, 0.3,  0.010, 0.1, 0.010])

    ring_frame_upper_bounds = np.array( [None, None, 0.9, 0.04, 0.3, 0.04])
    ring_frame_deltas = np.array(       [None, None, 0.2, 0.01, 0.1, 0.01])
    ring_frame_lower_bounds = np.array( [None, None, 0.5,  0.02, 0.2, 0.020])
    '''
    if min_var[0] is not None:
        spacing_array = (np.arange(min_var[0], max_var[0] + deltas[0], deltas[0])) if min_var[0] != max_var[0] \
            else np.array([min_var[0]])
        spacing_array = spacing_array[spacing_array <= max_var[0] + abs(deltas[0]) * 1e-9]
    else:
        spacing_array = np.array([np.nan])

    if min_var[1] is not None:
        pl_thk_array = (np.arange(min_var[1], max_var[1] + deltas[1], deltas[1])) if min_var[1] != max_var[1] \
            else np.array([min_var[1]])
        pl_thk_array = pl_thk_array[pl_thk_array <= max_var[1] + abs(deltas[1]) * 1e-9]
    else:
        pl_thk_array = np.array([np.nan])

    if predef_stiffeners is not None:
        predef_iterable = list()

        filtered_predef_stiffeners = [
            pre_str for pre_str in predef_stiffeners
            if _predefined_stiffener_within_bounds(pre_str, min_var, max_var)
        ]

        for pre_str in filtered_predef_stiffeners:
            for spacing in spacing_array:

                for pl_thk in pl_thk_array:
                    new_field = list(pre_str)
                    new_field[0] = spacing
                    new_field[1] = pl_thk
                    predef_iterable.append(tuple(new_field))

        return predef_iterable

    web_h_array = (np.arange(min_var[2], max_var[2] + deltas[2], deltas[2])) if min_var[2] != max_var[2] \
        else np.array([min_var[2]])
    web_h_array = web_h_array[web_h_array <= max_var[2] + abs(deltas[2]) * 1e-9]

    web_thk_array = (np.arange(min_var[3], max_var[3] + deltas[3], deltas[3])) if min_var[3] != max_var[3] \
        else np.array([min_var[3]])
    web_thk_array = web_thk_array[web_thk_array <= max_var[3] + abs(deltas[3]) * 1e-9]

    flange_w_array = (np.arange(min_var[4], max_var[4] + deltas[4], deltas[4])) if min_var[4] != max_var[4] \
        else np.array([min_var[4]])
    flange_w_array = flange_w_array[flange_w_array <= max_var[4] + abs(deltas[4]) * 1e-9]

    if min_var[5] is not None:
        flange_thk_array = (np.arange(min_var[5], max_var[5] + deltas[5], deltas[5])) if min_var[5] != max_var[5] \
            else np.array([min_var[5]])
        flange_thk_array = flange_thk_array[flange_thk_array <= max_var[5] + abs(deltas[5]) * 1e-9]
    else:
        flange_thk_array = np.array([np.nan])

    if min_var[6] is not None:
        span_array = (np.arange(min_var[6], max_var[6], deltas[4])) if min_var[6] != max_var[6] \
            else np.array([min_var[6]])
    else:
        span_array = np.array([np.nan])

    if min_var[7] is not None:
        girder_array = (np.arange(min_var[7], max_var[7], deltas[7])) if min_var[7] != max_var[7] \
            else np.array([min_var[7]])
    else:
        girder_array = np.array([np.nan])

    if len(min_var) >= 12:
        girder_web_h_delta = deltas[6] if len(deltas) > 6 else deltas[2]
        girder_web_thk_delta = deltas[7] if len(deltas) > 7 else deltas[3]
        girder_flange_w_delta = deltas[8] if len(deltas) > 8 else deltas[4]
        girder_flange_thk_delta = deltas[9] if len(deltas) > 9 else deltas[5]

        girder_web_h_array = (
            np.arange(min_var[8], max_var[8] + girder_web_h_delta, girder_web_h_delta)
        ) if min_var[8] != max_var[8] else np.array([min_var[8]])
        girder_web_h_array = girder_web_h_array[
            girder_web_h_array <= max_var[8] + abs(girder_web_h_delta) * 1e-9]

        girder_web_thk_array = (
            np.arange(min_var[9], max_var[9] + girder_web_thk_delta, girder_web_thk_delta)
        ) if min_var[9] != max_var[9] else np.array([min_var[9]])
        girder_web_thk_array = girder_web_thk_array[
            girder_web_thk_array <= max_var[9] + abs(girder_web_thk_delta) * 1e-9]

        girder_flange_w_array = (
            np.arange(min_var[10], max_var[10] + girder_flange_w_delta, girder_flange_w_delta)
        ) if min_var[10] != max_var[10] else np.array([min_var[10]])
        girder_flange_w_array = girder_flange_w_array[
            girder_flange_w_array <= max_var[10] + abs(girder_flange_w_delta) * 1e-9]

        girder_flange_thk_array = (
            np.arange(min_var[11], max_var[11] + girder_flange_thk_delta, girder_flange_thk_delta)
        ) if min_var[11] != max_var[11] else np.array([min_var[11]])
        girder_flange_thk_array = girder_flange_thk_array[
            girder_flange_thk_array <= max_var[11] + abs(girder_flange_thk_delta) * 1e-9]

        comb = it.product(spacing_array, pl_thk_array, web_h_array, web_thk_array, flange_w_array, flange_thk_array,
                          span_array, girder_array, girder_web_h_array, girder_web_thk_array,
                          girder_flange_w_array, girder_flange_thk_array)
        return list(comb)

    comb = it.product(spacing_array, pl_thk_array, web_h_array, web_thk_array, flange_w_array, flange_thk_array,
                      span_array, girder_array)

    return list(comb)


def get_initial_weight(obj, lat_press, min_var, max_var, deltas, trials, fat_dict, fat_press, predefined_stiffener_iter,
                       slamming_press, fdwn=1, fup=0.5, ml_algo=None, weld_bias=0.0, builtup_stiffener=False,
                       weld_metric='weld_consumables', cost_factors=None):
    '''
    Return a guess of the initial objective used to filter constraints.
    Only aim is to reduce running time of the algorithm.
    '''

    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    min_value = float('inf')
    if predefined_stiffener_iter is None:
        combs = any_get_all_combs(min_var, max_var, deltas)
    else:
        combs = any_get_all_combs(min_var, max_var, deltas, predef_stiffeners=[item.get_tuple() for item in
                                                                               predefined_stiffener_iter])

    trial_selection = random_product(combs, repeat=trials)
    obj.lat_press = lat_press
    for x in trial_selection:
        if any_constraints_all(x=x, obj=obj, lat_press=lat_press, init_weight=min_value,
                               fat_dict=fat_dict, fat_press=fat_press, slamming_press=slamming_press,
                               fdwn=fdwn, fup=fup, weld_bias=weld_bias,
                               builtup_stiffener=builtup_stiffener, weld_metric=weld_metric,
                               cost_factors=cost_factors)[0]:
            try:
                stiffener_type = (
                    obj.Stiffener.get_stiffener_type()
                    if obj.Stiffener is not None
                    else 'T'
                )
            except Exception:
                stiffener_type = 'T'
            current_value = calc_flat_objective_value(
                x,
                stiffener_type=stiffener_type,
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )

            if current_value < min_value:
                min_value = current_value
    return min_value


def get_random_result(obj, lat_press, min_var, max_var, deltas, trials=10000, side='p',
                      const_chk=(True, True, True, True, True),
                      fat_dict=None, fat_press=None, weld_bias=0.0, builtup_stiffener=False,
                      weld_metric='weld_consumables', cost_factors=None):
    '''
    Return random results
    '''
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    min_value = float('inf')
    ass_var = None
    combs = any_get_all_combs(min_var, max_var, deltas)
    trial_selection = random_product(combs, repeat=trials)
    try:
        stiffener_type = obj.Stiffener.get_stiffener_type() if obj.Stiffener is not None else 'T'
    except Exception:
        stiffener_type = 'T'
    for x in trial_selection:
        init_value = min_value if (cost_factors is not None or weld_bias <= 0.0 or weld_bias >= 1.0) else float('inf')
        if any_constraints_all(x=x, obj=obj, lat_press=lat_press, init_weight=init_value, side=side, chk=const_chk,
                               fat_dict=fat_dict, fat_press=fat_press, weld_bias=weld_bias,
                               builtup_stiffener=builtup_stiffener, weld_metric=weld_metric,
                               cost_factors=cost_factors)[0] is not False:
            current_value = calc_flat_objective_value(
                x,
                stiffener_type=stiffener_type,
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )

            if current_value < min_value:
                min_value = current_value
                ass_var = x
    if ass_var == None:
        return ass_var
    return create_new_structure_obj(obj, [round(item, 5) for item in ass_var]), \
        create_new_calc_obj(obj, [round(item, 5) for item in ass_var])[0]


def get_random_result_no_bounds(obj, lat_press, min_var, max_var, trials=10000, side='p',
                                const_chk=(True, True, True, True, True)
                                , fat_dict=None, fat_press=None, weld_bias=0.0, builtup_stiffener=False,
                                weld_metric='weld_consumables', cost_factors=None):
    '''
    Return random results, ignoring the deltas
    '''
    try:
        weld_bias = min(max(float(weld_bias), 0.0), 1.0)
    except Exception:
        weld_bias = 0.0
    weld_metric = normalize_weld_metric(weld_metric)
    cost_factors = normalize_cost_factors(cost_factors)

    min_value = float('inf')
    ass_var = None
    try:
        stiffener_type = obj.Stiffener.get_stiffener_type() if obj.Stiffener is not None else 'T'
    except Exception:
        stiffener_type = 'T'
    for trial in range(trials):
        spacing = random.randrange(int(min_var[0] * 1000), int(max_var[0] * 1000), 1) / 1000
        pl_thk = random.randrange(int(min_var[1] * 1000), int(max_var[1] * 1000), 1) / 1000
        web_h = random.randrange(int(min_var[2] * 1000), int(max_var[2] * 1000), 1) / 1000
        web_thk = random.randrange(int(min_var[3] * 1000), int(max_var[3] * 1000), 1) / 1000
        fl_w = random.randrange(int(min_var[4] * 1000), int(max_var[4] * 1000), 1) / 1000
        fl_thk = random.randrange(int(min_var[5] * 1000), int(max_var[5] * 1000), 1) / 1000
        x = (spacing, pl_thk, web_h, web_thk, fl_w, fl_thk, min_var[6], min_var[7])
        init_value = min_value if (cost_factors is not None or weld_bias <= 0.0 or weld_bias >= 1.0) else float('inf')
        if any_constraints_all(x=x, obj=obj, lat_press=lat_press, init_weight=init_value, side=side, chk=const_chk,
                               fat_dict=fat_dict, fat_press=fat_press, weld_bias=weld_bias,
                               builtup_stiffener=builtup_stiffener, weld_metric=weld_metric,
                               cost_factors=cost_factors)[0]:
            current_value = calc_flat_objective_value(
                x,
                stiffener_type=stiffener_type,
                weld_bias=weld_bias,
                include_web_to_flange=builtup_stiffener,
                weld_metric=weld_metric,
                cost_factors=cost_factors,
            )

            if current_value < min_value:
                min_value = current_value
                ass_var = x
    if ass_var == None:
        return ass_var
    return create_new_structure_obj(obj, [round(item, 5) for item in ass_var]), \
        create_new_calc_obj(obj, [round(item, 5) for item in ass_var])[0]


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


def product_any(*args, repeat=1, weight=float('inf')):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        if calc_weight(prod) < weight:
            yield tuple(prod)


def plot_optimization_results(results, multiple=False):
    check_ok_array, check_array, section_array = list(), list(), list()
    save_to_csv = asksaveasfilename()

    if save_to_csv != '':
        csv_file = open(save_to_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Is OK', 'Check info', 'pl b', 'pl thk', 'web h', 'web thk', 'fl b', 'fl thk', 'span',
                             'girder width', 'stiffener type', 'uf weight', 'uf sec mod', 'uf loc stf buc',
                             'uf buckling', 'uf min pl', 'uf shear', 'uf fatigue', 'uf slamming'])

    for check_ok, check, section, ufres in results[3]:
        check_ok_array.append(check_ok)
        check_array.append(check)
        section_array.append(section)
        if save_to_csv != '':
            to_write = list()
            to_write.append(check_ok)
            to_write.append(check)
            [to_write.append(item) for item in section]
            if (len(section) == 8):
                to_write.append('T')
            [to_write.append(item) for item in ufres]
            csv_writer.writerow(to_write)

    if save_to_csv != '':
        csv_file.close()

    check_ok_array, check_array, section_array = np.array(check_ok_array), \
        np.array(check_array), \
        np.array(section_array)

    x_label = np.unique(check_array)
    y = [np.count_nonzero(check_array == item) for item in np.unique(check_array)]

    fig, axs = plt.subplots(2, 1)
    clust_data = np.append(np.array(x_label).reshape(len(x_label), 1), np.array(y).reshape(len(y), 1), axis=1)
    collabel = ('Check fail type or OK', 'Number of occurences')
    axs[0].axis('tight')
    axs[0].axis('off')
    the_table = axs[0].table(cellText=clust_data, colLabels=collabel, loc='center')
    axs[1].pie(y, labels=x_label, autopct='%1.1f%%', explode=[0.1 for dummy in range(len(x_label))])

    plt.show()


if __name__ == '__main__':
    import example_data as ex
    from calc_structure import CylinderAndCurvedPlate, Structure, Shell

    shell_main_dict = ex.shell_main_dict
    shell_main_dict['geometry'] = [7, '']
    # Structure(ex.obj_dict_cyl_ring)
    # Structure(ex.obj_dict_cyl_heavy_ring)
    # my_cyl = CylinderAndCurvedPlate(main_dict = ex.shell_main_dict, shell= Shell(ex.shell_dict),
    #                                 long_stf= Structure(ex.obj_dict_cyl_long2),
    #                                 ring_stf = Structure(ex.obj_dict_cyl_ring2),
    #                                 ring_frame= Structure(ex.obj_dict_cyl_heavy_ring2))
    my_cyl = CylinderAndCurvedPlate(main_dict=ex.shell_main_dict, shell=Shell(ex.shell_dict),
                                    long_stf=Structure(ex.obj_dict_cyl_long2),
                                    ring_stf=None,  # Structure(ex.obj_dict_cyl_ring2),
                                    ring_frame=None)  # Structure(ex.obj_dict_cyl_heavy_ring2))

    shell_upper_bounds = np.array([0.03, 5, 5, 5, 10, None, None, None])
    shell_deltas = np.array([0.005, 0.5, 1, 0.1, 1, None, None, None])
    shell_lower_bounds = np.array([0.02, 5, 5, 5, 10, None, None, None])

    long_upper_bounds = np.array([0.8, None, 0.5, 0.02, 0.2, 0.03, None, None])
    long_deltas = np.array([0.1, None, 0.1, 0.01, 0.1, 0.01, None, None])
    long_lower_bounds = np.array([0.7, None, 0.3, 0.01, 0.1, 0.01, None, None])

    ring_stf_upper_bounds = np.array([None, None, 0.5, 0.018, 0.2, 0.03, None, None])
    ring_stf_deltas = np.array([None, None, 0.1, 0.004, 0.1, 0.01, None, None])
    ring_stf_lower_bounds = np.array([None, None, 0.3, 0.010, 0.1, 0.010, None, None])

    ring_frame_upper_bounds = np.array([None, None, 0.9, 0.04, 0.3, 0.04, None, None])
    ring_frame_deltas = np.array([None, None, 0.2, 0.01, 0.1, 0.01, None, None])
    ring_frame_lower_bounds = np.array([None, None, 0.7, 0.02, 0.2, 0.02, None, None])

    max_var = [shell_upper_bounds, long_upper_bounds, ring_stf_upper_bounds, ring_frame_upper_bounds]
    deltas = [shell_deltas, long_deltas, ring_stf_deltas, ring_frame_deltas]
    min_var = [shell_lower_bounds, long_lower_bounds, ring_stf_lower_bounds, ring_frame_lower_bounds]

    results = run_optmizataion(initial_structure_obj=my_cyl, min_var=min_var, max_var=max_var, deltas=deltas,
                               cylinder=True, use_weight_filter=True)
    shell = ['Shell thk. [mm]', 'Shell radius [mm]', 'l rings [mm]', 'L shell [mm]', 'L tot. [mm]', 'N/A - future',
             'N/A - future', 'N/A - future']
    stf_long = ['Spacing [mm]', 'Plate thk. [mm]', 'Web height [mm]', 'Web thk. [mm]', 'Flange width [mm]',
                'Flange thk. [mm]', 'N/A - future', 'N/A - future']
    stf_ring = ['N/A', 'Plate thk. [mm]', 'Web height [mm]', 'Web thk. [mm]', 'Flange width [mm]', 'Flange thk. [mm]',
                'N/A - future', 'N/A - future']

    # obj_dict = ex.obj_dict_sec_error
    # fat_obj = ex.get_fatigue_object_problematic()
    # fp = ex.get_fatigue_pressures_problematic()
    # fat_press = ((fp['p_ext']['loaded'],fp['p_ext']['ballast'],fp['p_ext']['part']),
    #              (fp['p_int']['loaded'],fp['p_int']['ballast'],fp['p_int']['part']))
    # x0 = [obj_dict['spacing'][0], obj_dict['plate_thk'][0], obj_dict['stf_web_height'][0], obj_dict['stf_web_thk'][0],
    #       obj_dict['stf_flange_width'][0], obj_dict['stf_flange_thk'][0], obj_dict['span'][0], 10]
    #
    # obj = calc.Structure(obj_dict)
    # lat_press = 427.235
    # calc_object = calc.CalcScantlings(obj_dict)
    # lower_bounds = np.array([0.875, 0.012, 0.3, 0.012, 0.1, 0.012, 3.5, 10])
    # upper_bounds = np.array([0.875, 0.025, 0.5, 0.018, 0.2, 0.03, 3.5, 10])
    # deltas = np.array([0.025, 0.001, 0.01, 0.001, 0.01, 0.001])
    #
    #
    # t1 = time.time()
    # #
    # results = run_optmizataion(obj, lower_bounds,upper_bounds, lat_press, deltas, algorithm='anysmart',
    #                            fatigue_obj=fat_obj, fat_press_ext_int=fat_press, use_weight_filter=True)
    #
    # print(results[1])
    # print(results[1].get_dnv_min_section_modulus(lat_press))
    # print(min([round(results[1].get_section_modulus()[0], 5), round(results[1].get_section_modulus()[1], 5)]))

    # t1 = time.time()
    # check_ok_array, check_array, section_array = list(), list(), list()
    #
    # for check_ok, check, section in results[4]:
    #     check_ok_array.append(check_ok)
    #     check_array.append(check)
    #     section_array.append(section)
    # check_ok_array, check_array, section_array = np.array(check_ok_array),\
    #                                              np.array(check_array),\
    #                                              np.array(section_array)
    #
    # x_label = np.unique(check_array)
    # y = [np.count_nonzero(check_array == item) for item in np.unique(check_array)]
    #
    # fig, axs = plt.subplots(2, 1)
    # clust_data = np.append(np.array(x_label).reshape(len(x_label),1), np.array(y).reshape(len(y),1), axis=1)
    # collabel = ('Check fail type or OK', 'Number of occurences')
    # axs[0].axis('tight')
    # axs[0].axis('off')
    # the_table = axs[0].table(cellText=clust_data, colLabels=collabel, loc='center')
    # axs[1].pie(y, labels = x_label, autopct='%1.1f%%', explode=[0.1 for dummy in range(len(x_label))])
    # plt.show()
    #
    # cmap = plt.cm.get_cmap(plt.cm.viridis, len(x_label))
    #

    # Create data
    # N = 60
    # x = section_array[:,0] * section_array[:,1]
    # y = section_array[:,2] * section_array[:,3]
    # z = section_array[:,4] * section_array[:,5]
    #
    # #data = (g1, g2, g3)
    #
    # groups = x_label
    # colors = "bgrcmykw"
    # color_dict = dict()
    # for idx, group in enumerate(groups):
    #     color_dict[group] = colors[idx]
    #
    # # Create plot
    # fig = plt.figure()
    # #ax = fig.add_subplot(1, 1, 1)
    # ax = fig.gca(projection='3d')
    #
    # for xdata, ydata, zdata, group in zip(x, y, z, groups):
    #     if group == 'Check OK':
    #         ax.scatter(x, y, z, alpha= 0.6 if group != 'Weight filter' else 0.2,
    #                    c=color_dict[group], edgecolors='none', s=5, label=group)
    #
    # plt.title('Matplot 3d scatter plot')
    # plt.legend(loc=2)
    # plt.show()

    # for swarm_size in [100, 1000, 10000, 100000, 1000000]:
    #     t1 = time.time()
    #
    #     pso_options = (swarm_size, 0.5, 0.5, 0.5, 100, 1e-8, 1e-8)
    #     results = run_optmizataion(obj, upper_bounds, lower_bounds, lat_press, deltas, algorithm='anysmart',
    #                            fatigue_obj=fat_obj, fat_press_ext_int=fat_press, pso_options=pso_options)[0]
    #     print('Swarm size', swarm_size, 'running time', time.time()-t1, results.get_one_line_string())
    # fat_press_ext_int = list()
    # for pressure in ex.get_geo_opt_fat_press():
    #     fat_press_ext_int.append(((pressure['p_ext']['loaded'], pressure['p_ext']['ballast'],
    #                                pressure['p_ext']['part']),
    #                               (pressure['p_int']['loaded'], pressure['p_int']['ballast'],
    #                                pressure['p_int']['part'])))
    #
    # opt_girder_prop = (0.018, 0.25,0.015, 0,0, 1.1,0.9)
    #
    # results = run_optmizataion(ex.get_geo_opt_object(), lower_bounds, upper_bounds, ex.get_geo_opt_presure(), deltas,
    #                            is_geometric=True, fatigue_obj=ex.get_geo_opt_fatigue(),
    #                            fat_press_ext_int=fat_press_ext_int,
    #                            slamming_press=ex.get_geo_opt_slamming_none(), load_pre=False,
    #                            opt_girder_prop= opt_girder_prop,
    #                            min_max_span=(1,12), tot_len=12)

    # import pickle
    # with open('geo_opt_2.pickle', 'rb') as file:
    #     geo_results = pickle.load(file)
    #
    # print(geo_results.keys())
    # print(geo_results[1][0])
    # for val in range(6):
    #     plot_optimization_results(geo_results[3][1][val])






