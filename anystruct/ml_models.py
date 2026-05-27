"""Machine-learning model file resolution and loading helpers."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MATERIAL_FACTORS = (1.1, 1.15)
ML_CLASS_MESSAGES = {
    0: "N/A",
    1: "A negative utilisation factor is found.",
    2: "At least one of the in-plane loads must be non-zero.",
    3: "Division by zero",
    4: "Overflow",
    5: "The aspect ratio exceeds the PULS code limit",
    6: "The global slenderness exceeds 4. Please reduce stiffener span or increase stiffener height.",
    7: "The applied pressure is too high for this plate field.",
    8: "web-flange-ratio",
    9: "UF below or equal 0.87",
    10: "UF between 0.87 and 1.0",
    11: "UF above 1.0",
}


def default_ml_class_messages():
    """Return the ML classifier result messages used by GUI/report surfaces."""
    return dict(ML_CLASS_MESSAGES)


@dataclass(frozen=True)
class ModelFileSpec:
    """Named model with current and compatible legacy file locations."""

    name: str
    file_bases: tuple[str, ...]


BUCKLING_MODEL_SPECS = (
    # -------------------------------------------------------------------------
    # Classification pipeline - existing CLPIPE models
    # -------------------------------------------------------------------------
    ModelFileSpec("cl SP buc int predictor", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_predictor_In-plane_support_cl_1_SP",)),
    ModelFileSpec("cl SP buc int scaler", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_scaler_In-plane_support_cl_1_SP",)),
    ModelFileSpec("cl SP ult int predictor", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_predictor_In-plane_support_cl_1_SP",)),
    ModelFileSpec("cl SP ult int scaler", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_scaler_In-plane_support_cl_1_SP",)),

    ModelFileSpec("cl SP buc GLGT predictor", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_predictor_In-plane_support_cl_2,_3_SP",)),
    ModelFileSpec("cl SP buc GLGT scaler", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_scaler_In-plane_support_cl_2,_3_SP",)),
    ModelFileSpec("cl SP ult GLGT predictor", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_predictor_In-plane_support_cl_2,_3_SP",)),
    ModelFileSpec("cl SP ult GLGT scaler", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_scaler_In-plane_support_cl_2,_3_SP",)),

    ModelFileSpec("cl UP buc int predictor", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_predictor_In-plane_support_cl_1_UP",)),
    ModelFileSpec("cl UP buc int scaler", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_scaler_In-plane_support_cl_1_UP",)),
    ModelFileSpec("cl UP ult int predictor", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_predictor_In-plane_support_cl_1_UP",)),
    ModelFileSpec("cl UP ult int scaler", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_scaler_In-plane_support_cl_1_UP",)),

    ModelFileSpec("cl UP buc GLGT predictor", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_predictor_In-plane_support_cl_2,_3_UP",)),
    ModelFileSpec("cl UP buc GLGT scaler", ("ml_files\\CLPIPE_CL_output_cl_str_buc_XXX_scaler_In-plane_support_cl_2,_3_UP",)),
    ModelFileSpec("cl UP ult GLGT predictor", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_predictor_In-plane_support_cl_2,_3_UP",)),
    ModelFileSpec("cl UP ult GLGT scaler", ("ml_files\\CLPIPE_CL_output_cl_str_ult_XXX_scaler_In-plane_support_cl_2,_3_UP",)),

    ModelFileSpec(
        "CSR predictor UP",
        (
            "ml_files\\CLPIPE_CL_CSR-Tank_req_cl_predictor_UP",
            "ml_files\\CLPIPE_CL_CSR-Tank_req_cl_predictor",
        ),
    ),
    ModelFileSpec(
        "CSR scaler UP",
        (
            "ml_files\\CLPIPE_CL_CSR-Tank_req_cl_scaler_UP",
            "ml_files\\CLPIPE_CL_CSR-Tank_req_cl_scaler",
        ),
    ),
    ModelFileSpec(
        "CSR predictor SP",
        (
            "ml_files\\CLPIPE_CL_CSR_plate_cl,_CSR_web_cl,_CSR_web_flange_cl,_CSR_flange_cl_predictor_SP",
            "ml_files\\CLPIPE_CL_CSR_plate_cl,_CSR_web_cl,_CSR_web_flange_cl,_CSR_flange_cl_predictor",
        ),
    ),
    ModelFileSpec(
        "CSR scaler SP",
        (
            "ml_files\\CLPIPE_CL_CSR_plate_cl,_CSR_web_cl,_CSR_web_flange_cl,_CSR_flange_cl_scaler_SP",
            "ml_files\\CLPIPE_CL_CSR_plate_cl,_CSR_web_cl,_CSR_web_flange_cl,_CSR_flange_cl_scaler",
        ),
    ),

    # -------------------------------------------------------------------------
    # Numeric UF pipeline - validity classifier + two-output UF regressor
    # -------------------------------------------------------------------------
    # SP, integrated support
    ModelFileSpec(
        "num SP int validity predictor",
        ("ml_files\\NUMPIPE_VALID_predictor_SP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num SP int validity xscaler",
        ("ml_files\\NUMPIPE_VALID_xscaler_SP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num SP int UF reg predictor",
        ("ml_files\\NUMPIPE_REG_predictor_SP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num SP int UF reg xscaler",
        ("ml_files\\NUMPIPE_REG_xscaler_SP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num SP int UF reg yscaler",
        ("ml_files\\NUMPIPE_REG_yscaler_SP_UF_numeric_In-plane_support_cl_1",),
    ),

    # SP, GL/GT support
    ModelFileSpec(
        "num SP GLGT validity predictor",
        ("ml_files\\NUMPIPE_VALID_predictor_SP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num SP GLGT validity xscaler",
        ("ml_files\\NUMPIPE_VALID_xscaler_SP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num SP GLGT UF reg predictor",
        ("ml_files\\NUMPIPE_REG_predictor_SP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num SP GLGT UF reg xscaler",
        ("ml_files\\NUMPIPE_REG_xscaler_SP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num SP GLGT UF reg yscaler",
        ("ml_files\\NUMPIPE_REG_yscaler_SP_UF_numeric_In-plane_support_cl_2,_3",),
    ),

    # UP, integrated support
    ModelFileSpec(
        "num UP int validity predictor",
        ("ml_files\\NUMPIPE_VALID_predictor_UP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num UP int validity xscaler",
        ("ml_files\\NUMPIPE_VALID_xscaler_UP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num UP int UF reg predictor",
        ("ml_files\\NUMPIPE_REG_predictor_UP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num UP int UF reg xscaler",
        ("ml_files\\NUMPIPE_REG_xscaler_UP_UF_numeric_In-plane_support_cl_1",),
    ),
    ModelFileSpec(
        "num UP int UF reg yscaler",
        ("ml_files\\NUMPIPE_REG_yscaler_UP_UF_numeric_In-plane_support_cl_1",),
    ),

    # UP, GL/GT support
    ModelFileSpec(
        "num UP GLGT validity predictor",
        ("ml_files\\NUMPIPE_VALID_predictor_UP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num UP GLGT validity xscaler",
        ("ml_files\\NUMPIPE_VALID_xscaler_UP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num UP GLGT UF reg predictor",
        ("ml_files\\NUMPIPE_REG_predictor_UP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num UP GLGT UF reg xscaler",
        ("ml_files\\NUMPIPE_REG_xscaler_UP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
    ModelFileSpec(
        "num UP GLGT UF reg yscaler",
        ("ml_files\\NUMPIPE_REG_yscaler_UP_UF_numeric_In-plane_support_cl_2,_3",),
    ),
)


def material_factor_token(material_factor: float) -> str:
    """Return the token embedded in material-factor-specific model filenames."""
    return (str(round(material_factor, 2)).replace(".", "") + "0")[:3]


def resolve_model_pickle(
    spec: ModelFileSpec,
    search_roots: Iterable[str | Path],
    material_factor: float,
) -> Path:
    """Find the first available current or legacy pickle for a model spec."""
    attempted = []
    roots = tuple(Path(root) for root in search_roots)
    for file_base in spec.file_bases:
        relative_path = _pickle_path(file_base, material_factor)
        for candidate in _candidate_paths(relative_path, roots):
            attempted.append(candidate)
            if candidate.is_file():
                return candidate

    searched = ", ".join(str(path) for path in attempted)
    raise FileNotFoundError(f"Could not find ML model {spec.name!r}. Checked: {searched}")


def load_buckling_models(
    search_roots: Iterable[str | Path],
    material_factors: Iterable[float] = MATERIAL_FACTORS,
) -> dict[float, dict[str, Any]]:
    """Load the buckling and CSR classifier models used by the main workspace."""
    roots = tuple(search_roots)
    models = {}
    for material_factor in material_factors:
        models[material_factor] = {}
        for spec in BUCKLING_MODEL_SPECS:
            model_path = resolve_model_pickle(spec, roots, material_factor)
            with model_path.open("rb") as model_file:
                models[material_factor][spec.name] = pickle.load(model_file)
    return models


def _pickle_path(file_base: str, material_factor: float) -> Path:
    path = file_base.replace("XXX", material_factor_token(material_factor)).replace("\\", "/")
    return Path(path + ".pickle")


def _candidate_paths(relative_path: Path, roots: tuple[Path, ...]) -> tuple[Path, ...]:
    candidates = [relative_path]
    for root in roots:
        candidates.append(root / relative_path)
        if relative_path.parts and relative_path.parts[0] == "ml_files":
            candidates.append(root / Path(*relative_path.parts[1:]))
    return tuple(dict.fromkeys(candidates))
