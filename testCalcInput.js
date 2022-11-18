StiffenedPanel = {
    "mat_yield": 355e6,
    "mat_factor": 1.15,
    "span": 3.7,
    "spacing": 0.75,
    "plate_thk": 0.018,
    "stf_web_height": 0.4,
    "stf_web_thk": 0.012,
    "stf_flange_width": 0.25,
    "stf_flange_thk": 0.014,
    "structure_type": "BOTTOM",
    "plate_kpp": 1,
    "stf_kps": 1,
    "stf_km1": 12,
    "stf_km2": 24,
    "stf_km3": 12,
    "sigma_y1": 100,
    "sigma_y2": 100,
    "sigma_x2": 102.7,
    "sigma_x1": 102.7,
    "tau_xy": 5,
    "stf_type": "T",
    "structure_types": "structure_types",
    "zstar_optimization": true,
    "puls_buckling_method": 1,
    "puls_boundary": "Int",
    "puls_stiffener_end": "C",
    "puls_sp_or_up": "SP",
    "puls_up_boundary": "SSSS",
    "panel_or_shell": "panel",
    "pressure_side": "both sides",
    "girder_lg": 5
}

StiffenedPanel = {
    "Plate": {
        "mat_yield": 355e6,
        "mat_factor": 1.15,
        "span": 3.7,
        "spacing": 0.75,
        "plate_thk": 0.018,
        "stf_web_height": 0.4,
        "stf_web_thk": 0.012,
        "stf_flange_width": 0.25,
        "stf_flange_thk": 0.014,
        "structure_type": "BOTTOM",
        "plate_kpp": 1,
        "stf_kps": 1,
        "stf_km1": 12,
        "stf_km2": 24,
        "stf_km3": 12,
        "sigma_y1": 100,
        "sigma_y2": 100,
        "sigma_x2": 102.7,
        "sigma_x1": 102.7,
        "tau_xy": 5,
        "stf_type": "T",
        "structure_types": "structure_types",
        "zstar_optimization": true,
        "puls_buckling_method": 1,
        "puls_boundary": "Int",
        "puls_stiffener_end": "C",
        "puls_sp_or_up": "SP",
        "puls_up_boundary": "SSSS",
        "panel_or_shell": "panel",
        "pressure_side": "both sides",
        "girder_lg": 5
    },
    "Stiffener": {
        "mat_yield": 355e6,
        "mat_factor": 1.15,
        "span": 3.7,
        "spacing": 0.75,
        "plate_thk": 0.018,
        "stf_web_height": 0.4,
        "stf_web_thk": 0.012,
        "stf_flange_width": 0.25,
        "stf_flange_thk": 0.014,
        "structure_type": "BOTTOM",
        "plate_kpp": 1,
        "stf_kps": 1,
        "stf_km1": 12,
        "stf_km2": 24,
        "stf_km3": 12,
        "sigma_y1": 100,
        "sigma_y2": 100,
        "sigma_x2": 102.7,
        "sigma_x1": 102.7,
        "tau_xy": 5,
        "stf_type": "T",
        "structure_types": "structure_types",
        "zstar_optimization": true,
        "puls_buckling_method": 1,
        "puls_boundary": "Int",
        "puls_stiffener_end": "C",
        "puls_sp_or_up": "SP",
        "puls_up_boundary": "SSSS",
        "panel_or_shell": "panel",
        "pressure_side": "both sides",
        "girder_lg": 5
    },
    "Girder": {
        "mat_yield": 355e6,
        "mat_factor": 1.15,
        "span": 3.7,
        "spacing": 0.75,
        "plate_thk": 0.018,
        "stf_web_height": 0.5,
        "stf_web_thk": 0.012,
        "stf_flange_width": 0.15,
        "stf_flange_thk": 0.02,
        "structure_type": "BOTTOM",
        "plate_kpp": 1,
        "stf_kps": 1,
        "stf_km1": 12,
        "stf_km2": 24,
        "stf_km3": 12,
        "sigma_y1": 80,
        "sigma_y2": 80,
        "sigma_x2": 80,
        "sigma_x1": 80,
        "tau_xy": 5,
        "stf_type": "T",
        "structure_types": "structure_types",
        "zstar_optimization": true,
        "puls_buckling_method": 2,
        "puls_boundary": "Int",
        "puls_stiffener_end": "C",
        "puls_sp_or_up": "SP",
        "puls_up_boundary": "SSSS",
        "panel_or_shell": "panel",
        "pressure_side": "both sides",
        "girder_lg": 5
    },
    "Prescriptive": {
        "material_yield": 355e6,
        "load_factor_on_stresses": 1,
        "load_factor_on_pressure": 1,
        "buckling_method": "ultimate",
        "stiffener_end_support": "Continuous",
        "girder_end_support": "Continuous",
        "tension_field": "not allowed",
        "plate_effective_against_sigy": true,
        "km2": 24,
        "km3": 12,
        "pressure_side": "both sides",
        "fabrication_method_stiffener": "welded",
        "fabrication_method_girder": "welded",
        "calculation_domain": "Flat plate, stiffened"
    },
    "Pressure": 0.412197
}