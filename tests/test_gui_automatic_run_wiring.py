from pathlib import Path


def test_gui_automatic_run_resolves_ml_files_without_github_path():
    gui_source = Path(__file__).resolve().parent / "gui_automatic_run.py"
    source = gui_source.read_text(encoding="utf-8")

    assert "ANYSTRUCTURE_ML_FILES" in source
    assert r"C:\python_projects\ANYstructure\anystruct\ml_files" in source
    assert "REPO_ROOT / \"anystruct\" / \"ml_files\"" in source
    assert "resolve_ml_pickle(file_base)" in source
    assert r"C:\Github\ANYstructure" not in source


def test_gui_automatic_run_patches_dialogs_for_unattended_execution():
    gui_source = Path(__file__).resolve().parent / "gui_automatic_run.py"
    source = gui_source.read_text(encoding="utf-8")

    assert "configure_noninteractive_dialogs()" in source
    assert "messagebox.askquestion = lambda *args, **kwargs: \"no\"" in source
    assert "messagebox.showwarning = lambda *args, **kwargs: \"ok\"" in source
    assert "root.destroy()" in source


def test_gui_automatic_run_suppresses_plot_windows():
    gui_source = Path(__file__).resolve().parent / "gui_automatic_run.py"
    source = gui_source.read_text(encoding="utf-8")

    assert "configure_noninteractive_plots()" in source
    assert "plt.show = lambda *args, **kwargs: None" in source
    assert "plt.close(\"all\")" in source
