from pathlib import Path
from tempfile import TemporaryDirectory

from anystruct import ml_models


def _spec(name):
    return next(spec for spec in ml_models.BUCKLING_MODEL_SPECS if spec.name == name)


def test_resolve_model_pickle_supports_current_split_csr_names():
    with TemporaryDirectory(dir=Path.cwd()) as model_dir:
        package_root = Path(model_dir)
        current_model = package_root / "ml_files" / "CLPIPE_CL_CSR-Tank_req_cl_predictor_UP.pickle"
        current_model.parent.mkdir()
        current_model.write_bytes(b"current")

        resolved = ml_models.resolve_model_pickle(_spec("CSR predictor UP"), (package_root,), 1.15)

        assert resolved == current_model


def test_resolve_model_pickle_keeps_legacy_unsuffixed_csr_names():
    with TemporaryDirectory(dir=Path.cwd()) as model_dir:
        package_root = Path(model_dir)
        legacy_model = package_root / "ml_files" / "CLPIPE_CL_CSR-Tank_req_cl_predictor.pickle"
        legacy_model.parent.mkdir()
        legacy_model.write_bytes(b"legacy")

        resolved = ml_models.resolve_model_pickle(_spec("CSR predictor UP"), (package_root,), 1.15)

        assert resolved == legacy_model


def test_resolve_model_pickle_accepts_ml_files_directory_roots_and_material_factor_tokens():
    spec = ml_models.ModelFileSpec("factor model", ("ml_files\\factor_XXX",))
    with TemporaryDirectory(dir=Path.cwd()) as model_dir:
        ml_files = Path(model_dir)
        factor_model = ml_files / "factor_110.pickle"
        factor_model.write_bytes(b"factor")

        resolved = ml_models.resolve_model_pickle(spec, (ml_files,), 1.1)

        assert resolved == factor_model
