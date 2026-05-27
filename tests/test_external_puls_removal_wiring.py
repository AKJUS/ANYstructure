from pathlib import Path


def read_main_source():
    return (Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py").read_text(
        encoding="utf-8"
    )


def read_optimize_window_source():
    return (Path(__file__).resolve().parents[1] / "anystruct" / "optimize_window.py").read_text(
        encoding="utf-8"
    )


def read_report_source():
    return (Path(__file__).resolve().parents[1] / "anystruct" / "report_generator.py").read_text(
        encoding="utf-8"
    )


def test_main_gui_has_no_external_puls_run_or_result_controls():
    source = read_main_source()

    assert "label='Run all PULS lines'" not in source
    assert "label='Delete all PULS results'" not in source
    assert "text='Run PULS -" not in source
    assert "text='PULS results for line'" not in source
    assert "def puls_run_all_lines" not in source
    assert "def puls_run_one_line" not in source
    assert "def puls_delete_all" not in source
    assert "def on_puls_results_for_line" not in source
    assert "Set location of PULS excel sheet" not in source
    assert "['DNV-RP-C201 - prescriptive','DNV PULS','ML-CL (SemiAnalytical based)']" not in source
    assert "self._PULS_results" not in source
    assert "== 'DNV PULS'" not in source


def test_optimizer_has_no_external_puls_excel_selection():
    source = read_optimize_window_source()

    assert "Check for buckling (PULS)" not in source
    assert "Check for buckling (SemiAnalytical S3/U3)" in source
    assert "Set location of PULS excel sheet" not in source
    assert "puls_sheet = puls_sheet_location" not in source


def test_reports_have_no_external_puls_result_branch():
    source = read_report_source()

    assert "== 'DNV PULS'" not in source
    assert "_PULS_results" not in source
    assert "PULS colors" not in source


def test_cylinder_optimizer_does_not_capture_external_puls_results():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "optimize_cylinder.py").read_text(
        encoding="utf-8"
    )

    assert "_PULS_results" not in source
    assert "_PULS_object" not in source


def test_excel_import_and_ml_cl_buckling_option_remain_available():
    source = read_main_source()

    assert "sub_sesam.add_command(label='Import excel file', command=self.open_excel_file)" in source
    assert "sub_menu.add_command(label='Open excel input', command=self.open_excel_file)" in source
    buckling_options = source[
        source.index("self._new_buckling_method = tk.StringVar()"):
        source.index("self._lab_buckling_method = ttk.Label")
    ]
    assert "'ML-CL (SemiAnalytical based)'" not in buckling_options
    assert "'ML-Numeric (PULS based)'" in buckling_options
    assert "'SemiAnalytical S3/U3'" in buckling_options
    assert "'DNV PULS'" not in buckling_options
