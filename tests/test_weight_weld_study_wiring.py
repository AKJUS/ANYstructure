from pathlib import Path


def read_optimize_window_source():
    return (Path(__file__).resolve().parents[1] / "anystruct" / "optimize_window.py").read_text(
        encoding="utf-8"
    )


def test_single_optimizer_has_weight_weld_study_button():
    source = read_optimize_window_source()

    assert "text='weight/weld study'" in source
    assert "command=self.run_weight_weld_study" in source
    assert "def run_weight_weld_study(self):" in source
    assert "self._new_weld_study_delta" in source
    assert "self._new_weld_study_delta.set(0.1)" in source
    assert "self._ent_weld_study_delta" in source
    assert "text='show previous study'" in source
    assert "command=self.show_previous_weight_weld_study" in source
    assert "def show_previous_weight_weld_study(self):" in source
    assert "self._last_weight_weld_study_rows" in source


def test_weight_weld_study_shows_table_and_plot():
    source = read_optimize_window_source()

    assert "ttk.Treeview" in source
    assert "FigureCanvasTkAgg" in source
    assert "Weight/weld study" in source


def test_weight_weld_study_uses_delta_input():
    source = read_optimize_window_source()

    assert "delta = float(self._new_weld_study_delta.get())" in source
    assert "Weight/weld study delta must be larger than 0" in source


def test_running_time_warns_for_mixed_weight_weld_combination():
    source = read_optimize_window_source()

    assert "mixed weight/weld combination disables the initial filter" in source
    assert "Pure weld objective: initial filter uses ' + self._get_weld_metric_text()" in source


def test_single_optimizer_has_weld_metric_selector():
    source = read_optimize_window_source()

    assert "self._new_weld_metric" in source
    assert "'Weld consumables'" in source
    assert "'Weld length'" in source
    assert "def _get_weld_metric_for_optimization(self):" in source
    assert "weld_metric=self._get_weld_metric_for_optimization()" in source
    assert "op.calc_weld_objective(" in source


def test_single_optimizer_has_cost_study_button_and_dialog():
    source = read_optimize_window_source()

    assert "text='cost study'" in source
    assert "command=self.open_cost_study_window" in source
    assert "def open_cost_study_window(self):" in source
    assert "Steel cost per kg" in source
    assert "'Weld cost per ' + self._get_weld_metric_unit()" in source
    assert "def run_cost_study(self, cost_factors):" in source
    assert "cost_factors=cost_factors" in source
    assert "Cost optimization result" in source
