#!/usr/bin/env python3
"""Prepare dialog for neuroCombat harmonization from matrix stacks."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from window.shared.controls import make_toggle_button as _make_toggle_button
    from window.shared.covariate_table import (
        CovariateFilterPage as _CovariateFilterPage,
        filter_row_indices as _filter_row_indices,
    )
    from window.shared.covars import (
        column_is_numeric as _column_is_numeric,
        covars_to_rows as _covars_to_rows,
        display_text as _display_text,
    )
    from window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag as _is_editable_flag,
        is_enabled_flag as _is_enabled_flag,
        is_selectable_flag as _is_selectable_flag,
        is_user_checkable_flag as _is_user_checkable_flag,
    )
    from window.shared.workflow import WorkflowShell as _WorkflowShell
except Exception:
    from mrsi_viewer.window.shared.controls import make_toggle_button as _make_toggle_button
    from mrsi_viewer.window.shared.covariate_table import (
        CovariateFilterPage as _CovariateFilterPage,
        filter_row_indices as _filter_row_indices,
    )
    from mrsi_viewer.window.shared.covars import (
        column_is_numeric as _column_is_numeric,
        covars_to_rows as _covars_to_rows,
        display_text as _display_text,
    )
    from mrsi_viewer.window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag as _is_editable_flag,
        is_enabled_flag as _is_enabled_flag,
        is_selectable_flag as _is_selectable_flag,
        is_user_checkable_flag as _is_user_checkable_flag,
    )
    from mrsi_viewer.window.shared.workflow import WorkflowShell as _WorkflowShell

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


install_qt_compat_aliases(Qt, QT_LIB)


def _load_harmonize_module():
    try:
        from scripts import harmonize_connectivity as mod

        return mod
    except Exception:
        pass

    try:
        from mrsi_viewer.scripts import harmonize_connectivity as mod

        return mod
    except Exception:
        pass

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "harmonize_connectivity.py"
    spec = importlib.util.spec_from_file_location("harmonize_connectivity", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load harmonize_connectivity.py from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HarmonizePrepareDialog(QDialog):
    """Popup dialog to filter rows and run neuroCombat harmonization."""

    STEP_TITLES = ("Data", "Model", "Run")
    NUMERIC_RANGE_RE = re.compile(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
    )
    TYPE_OPTIONS = ("categorical", "continuous")

    def __init__(
        self,
        covars_info,
        source_path,
        matrix_key,
        output_dir_default="",
        theme_name="Dark",
        export_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._matrix_key = str(matrix_key)
        self._output_dir_default = str(output_dir_default or "").strip()
        self._columns, self._rows = _covars_to_rows(covars_info)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._data_table_refreshing = False
        self._current_step = 0
        self._running = False
        self._export_callback = export_callback
        self._last_run = None
        self._last_output_path = None
        self._output_name_auto = True
        self._confound_controls = {}

        self._harm_mod = _load_harmonize_module()

        self.setWindowTitle("Harmonize Prepare")
        self.resize(1220, 850)
        self._build_ui()
        self.set_theme(theme_name)
        self._refresh_data_table()
        self._init_model_table()
        self._go_to_step(0)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        self.workflow_shell = _WorkflowShell(self.STEP_TITLES, on_step_selected=self._go_to_step)
        self._step_buttons = self.workflow_shell.step_buttons
        self.step_stack = self.workflow_shell.step_stack
        self.workflow_shell.add_step(self._build_step_data())
        self.workflow_shell.add_step(self._build_step_model())
        self.workflow_shell.add_step(self._build_step_run())
        root_layout.addWidget(self.workflow_shell, 1)

        self.status_label = QLabel("")
        root_layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_prev_step)
        actions.addWidget(self.back_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._go_next_step)
        actions.addWidget(self.next_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        root_layout.addLayout(actions)

    def _build_step_data(self):
        self.data_page = _CovariateFilterPage(self._columns, exclude_mode="toggle_button")
        self.dataset_summary_label = self.data_page.dataset_summary_label
        self.filter_covar_combo = self.data_page.filter_covar_combo
        self.filter_value_edit = self.data_page.filter_value_edit
        self.filter_button = self.data_page.filter_button
        self.filter_reset_button = self.data_page.filter_reset_button
        self.data_table = self.data_page.table
        self.showing_rows_label = self.data_page.showing_rows_label
        self.data_page.set_dataset_summary(
            self._source_path.name,
            self._matrix_key,
            len(self._rows),
            len(self._columns),
        )
        self.data_page.set_filter_callbacks(self._apply_filter, self._reset_filter)
        return self.data_page

    def _build_step_model(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        model_group = QGroupBox("Batch and confounds")
        model_layout = QVBoxLayout(model_group)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch column"))
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(self._columns)
        self.batch_combo.currentTextChanged.connect(self._on_batch_changed)
        batch_row.addWidget(self.batch_combo, 1)
        model_layout.addLayout(batch_row)

        self.confound_table = QTableWidget()
        self.confound_table.setColumnCount(4)
        self.confound_table.setHorizontalHeaderLabels(["Include", "Covariate", "Continuous", "Categorical"])
        self.confound_table.setRowCount(len(self._columns))
        self.confound_table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            header = self.confound_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            self.confound_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.confound_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            header = self.confound_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.confound_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.confound_table.setSelectionMode(QAbstractItemView.SingleSelection)
        model_layout.addWidget(self.confound_table, 1)

        self.model_hint_label = QLabel(
            "Select nuisance/confound covariates and define each type for neuroCombat."
        )
        self.model_hint_label.setWordWrap(True)
        model_layout.addWidget(self.model_hint_label)

        layout.addWidget(model_group, 1)
        return page

    def _build_step_run(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        output_group = QGroupBox("Output")
        out_grid = QGridLayout(output_group)
        out_grid.addWidget(QLabel("Output folder"), 0, 0)
        self.output_dir_edit = QLineEdit(str(self._source_path.parent))
        out_grid.addWidget(self.output_dir_edit, 0, 1)
        self.output_dir_button = QPushButton("Browse")
        self.output_dir_button.clicked.connect(self._browse_output_dir)
        out_grid.addWidget(self.output_dir_button, 0, 2)

        out_grid.addWidget(QLabel("Output file"), 1, 0)
        self.output_name_edit = QLineEdit("")
        self.output_name_edit.setPlaceholderText("harmonized_connectivity.npz")
        self.output_name_edit.textEdited.connect(self._on_output_name_edited)
        out_grid.addWidget(self.output_name_edit, 1, 1, 1, 2)

        out_grid.addWidget(QLabel("Full path"), 2, 0)
        self.output_path_label = QLabel("")
        self.output_path_label.setWordWrap(True)
        out_grid.addWidget(self.output_path_label, 2, 1, 1, 2)
        layout.addWidget(output_group)

        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)
        self.fisher_checkbox = QCheckBox("Apply Fisher transform on upper-triangle edges")
        self.fisher_checkbox.setChecked(True)
        self.fisher_checkbox.toggled.connect(
            lambda _checked=False, self=self: self._apply_default_output_name(force=False)
        )
        run_layout.addWidget(self.fisher_checkbox)

        run_buttons = QHBoxLayout()
        self.run_button = QPushButton("Run neuroCombat")
        self.run_button.clicked.connect(self._run_harmonization)
        run_buttons.addWidget(self.run_button)
        self.export_button = QPushButton("Export To Workspace")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_to_workspace)
        run_buttons.addWidget(self.export_button)
        run_buttons.addStretch(1)
        run_layout.addLayout(run_buttons)

        self.run_summary_label = QLabel("No harmonization computed yet.")
        self.run_summary_label.setWordWrap(True)
        run_layout.addWidget(self.run_summary_label)

        self.log_output = QPlainTextEdit()
        self.log_output.setObjectName("harmonizeTerminal")
        self.log_output.setReadOnly(True)
        self.log_output.document().setMaximumBlockCount(4000)
        if QT_LIB == 6:
            self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        else:
            self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_output.setPlaceholderText("Harmonization progress will appear here.")
        self.log_output.setMinimumHeight(160)
        run_layout.addWidget(self.log_output)

        self.result_figure = Figure(figsize=(10, 6))
        self.result_canvas = FigureCanvas(self.result_figure)
        self.result_canvas.setVisible(False)
        run_layout.addWidget(self.result_canvas, 1)
        layout.addWidget(run_group, 1)

        self._apply_terminal_style()
        self._refresh_output_path_preview()
        return page

    def _go_to_step(self, step_index):
        step = self.workflow_shell.set_current_step(step_index)
        self._current_step = step
        self.back_button.setEnabled(step > 0 and not self._running)
        self.next_button.setVisible(step < (len(self.STEP_TITLES) - 1))
        self.next_button.setEnabled((step < (len(self.STEP_TITLES) - 1)) and (not self._running))

    def _go_next_step(self):
        self._go_to_step(self._current_step + 1)

    def _go_prev_step(self):
        self._go_to_step(self._current_step - 1)

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _append_terminal_line(self, text):
        if not hasattr(self, "log_output") or self.log_output is None:
            return
        if text is None:
            return
        lines = str(text).splitlines() or [str(text)]
        for line in lines:
            if line.strip() == "":
                continue
            self.log_output.appendPlainText(line)
        scroll = self.log_output.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _apply_terminal_style(self):
        if not hasattr(self, "log_output") or self.log_output is None:
            return
        self.log_output.setStyleSheet(
            "QPlainTextEdit#harmonizeTerminal {"
            " background-color: #000000;"
            " color: #b8f7c6;"
            " border: 1px solid #404040;"
            " border-radius: 4px;"
            " selection-background-color: #2d6cdf;"
            " font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
            " font-size: 10.5pt;"
            "}"
        )

    def _on_exclude_row_toggled(self, source_idx, checked):
        if self._data_table_refreshing:
            return
        source_idx = int(source_idx)
        if checked:
            self._excluded_indices.add(source_idx)
        else:
            self._excluded_indices.discard(source_idx)
        self._update_showing_rows_label()

    def _apply_filter(self):
        if not self._columns:
            self._set_status("No covariates available to filter.")
            return

        matched, target_values, error = _filter_row_indices(
            self._rows,
            self.filter_covar_combo.currentText(),
            self.filter_value_edit.text(),
            self.NUMERIC_RANGE_RE,
        )
        if error:
            self._set_status(error)
            return
        self._filtered_indices = matched
        self._refresh_data_table()
        self._set_status(
            f"Filter applied: {self.filter_covar_combo.currentText().strip()} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_data_table()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")

    def _refresh_data_table(self):
        self._data_table_refreshing = True
        self.data_page.populate_rows(
            self._filtered_indices,
            self._rows,
            self._excluded_indices,
            on_exclude_toggled=self._on_exclude_row_toggled,
        )
        self._data_table_refreshing = False
        self._update_showing_rows_label()

    def _update_showing_rows_label(self):
        self.data_page.update_showing_rows(
            len(self._filtered_indices),
            len(self._rows),
            len(self.selected_row_indices()),
        )

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]

    def _default_batch_col(self) -> str:
        preferred = {"scanner", "site", "batch"}
        for name in self._columns:
            if name.strip().lower() in preferred:
                return name
        return self._columns[0] if self._columns else ""

    def _covariate_type(self, covar_name: str) -> str:
        values = [row.get(covar_name) for row in self._rows]
        return "continuous" if _column_is_numeric(values) else "categorical"

    def _is_id_like(self, covar_name: str) -> bool:
        return covar_name.strip().lower() in {
            "participant_id",
            "subject_id",
            "session_id",
            "id",
            "sub",
            "ses",
        }

    def _init_model_table(self):
        default_batch = self._default_batch_col()
        if default_batch and self.batch_combo.findText(default_batch) >= 0:
            self.batch_combo.setCurrentText(default_batch)

        self._confound_controls = {}
        for row_idx, covar_name in enumerate(self._columns):
            include_button = _make_toggle_button(checked=False)
            include_button.clicked.connect(
                lambda _checked=False, idx=row_idx: self._on_include_confound_toggled(idx)
            )
            self.confound_table.setCellWidget(row_idx, 0, include_button)

            name_item = QTableWidgetItem(covar_name)
            name_item.setFlags(name_item.flags() & ~_is_editable_flag())
            self.confound_table.setItem(row_idx, 1, name_item)

            continuous_button = _make_toggle_button(checked=False)
            categorical_button = _make_toggle_button(checked=False)
            continuous_button.clicked.connect(
                lambda _checked=False, idx=row_idx: self._set_confound_type(idx, "continuous")
            )
            categorical_button.clicked.connect(
                lambda _checked=False, idx=row_idx: self._set_confound_type(idx, "categorical")
            )
            self.confound_table.setCellWidget(row_idx, 2, continuous_button)
            self.confound_table.setCellWidget(row_idx, 3, categorical_button)

            self._confound_controls[row_idx] = {
                "include": include_button,
                "continuous": continuous_button,
                "categorical": categorical_button,
            }
            self._sync_confound_row_buttons(row_idx, emit_update=False)
        self._apply_default_output_name(force=True)

    def _on_include_confound_toggled(self, row_idx):
        self._sync_confound_row_buttons(row_idx, emit_update=True)

    def _set_confound_type(self, row_idx, covar_type: str):
        controls = self._confound_controls.get(int(row_idx), {})
        include_button = controls.get("include")
        continuous_button = controls.get("continuous")
        categorical_button = controls.get("categorical")
        if include_button is None or continuous_button is None or categorical_button is None:
            return
        include_button.setChecked(True)
        covar_type = str(covar_type or "").strip().lower()
        is_continuous = covar_type == "continuous"
        continuous_button.setChecked(is_continuous)
        categorical_button.setChecked(not is_continuous)
        self._sync_confound_row_buttons(row_idx, emit_update=True)

    def _sync_confound_row_buttons(self, row_idx, emit_update=True):
        controls = self._confound_controls.get(int(row_idx), {})
        include_button = controls.get("include")
        continuous_button = controls.get("continuous")
        categorical_button = controls.get("categorical")
        if include_button is None or continuous_button is None or categorical_button is None:
            return

        include_checked = bool(include_button.isChecked())
        continuous_button.setEnabled(include_checked)
        categorical_button.setEnabled(include_checked)
        if not include_checked:
            continuous_button.setChecked(False)
            categorical_button.setChecked(False)
        elif not (continuous_button.isChecked() or categorical_button.isChecked()):
            default_type = self._covariate_type(self._columns[int(row_idx)])
            continuous_button.setChecked(default_type == "continuous")
            categorical_button.setChecked(default_type != "continuous")
        if emit_update:
            self._apply_default_output_name(force=False)

    def _selected_confounds(self):
        categorical = []
        continuous = []
        for row_idx, covar_name in enumerate(self._columns):
            controls = self._confound_controls.get(row_idx, {})
            include_button = controls.get("include")
            continuous_button = controls.get("continuous")
            categorical_button = controls.get("categorical")
            if include_button is None or not include_button.isChecked():
                continue
            if continuous_button is not None and continuous_button.isChecked():
                continuous.append(covar_name)
            else:
                categorical.append(covar_name)
        return categorical, continuous

    def _on_batch_changed(self, _value):
        self._apply_default_output_name(force=False)

    def _on_output_name_edited(self, _text):
        self._output_name_auto = False
        self._refresh_output_path_preview()

    def _apply_default_output_name(self, force=False):
        if (not force) and (not self._output_name_auto):
            self._refresh_output_path_preview()
            return
        try:
            default_path = self._harm_mod.build_default_output_path(
                source_path=self._source_path,
                matrix_key=self._matrix_key,
                batch_col=self.batch_combo.currentText().strip() if hasattr(self, "batch_combo") else "batch",
                categorical_cols=self._selected_confounds()[0] if hasattr(self, "confound_table") else [],
                continuous_cols=self._selected_confounds()[1] if hasattr(self, "confound_table") else [],
                apply_fisher=bool(self.fisher_checkbox.isChecked()) if hasattr(self, "fisher_checkbox") else False,
                output_dir=self.output_dir_edit.text().strip() if hasattr(self, "output_dir_edit") else self._source_path.parent,
            )
            self.output_name_edit.setText(default_path.name)
            self._output_name_auto = True
        except Exception:
            if force:
                self.output_name_edit.setText(f"{self._source_path.stem}_harmonized_connectivity_unknown.npz")
        self._refresh_output_path_preview()

    def _browse_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or str(self._source_path.parent)
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if selected:
            self.output_dir_edit.setText(selected)
            self._apply_default_output_name(force=False)

    def _output_path(self) -> Path:
        out_dir = self.output_dir_edit.text().strip() or str(self._source_path.parent)
        out_name = self.output_name_edit.text().strip() or f"{self._source_path.stem}_harmonized_connectivity_unknown.npz"
        output_path = Path(out_dir).expanduser() / out_name
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        return output_path

    def _refresh_output_path_preview(self):
        if not hasattr(self, "output_path_label"):
            return
        try:
            output = self._output_path()
            self.output_path_label.setText(str(output))
        except Exception:
            self.output_path_label.setText("Invalid output path")

    def _set_busy(self, busy: bool):
        self._running = bool(busy)
        self.run_button.setEnabled(not busy)
        self.export_button.setEnabled((not busy) and (self._last_output_path is not None))
        self.back_button.setEnabled((not busy) and self._current_step > 0)
        self.next_button.setEnabled((not busy) and self._current_step < (len(self.STEP_TITLES) - 1))
        self.close_button.setEnabled(not busy)

    def _run_harmonization(self):
        if self._running:
            return

        selected_rows = self.selected_row_indices()
        if len(selected_rows) < 2:
            self._set_status("Select at least two rows before harmonization.")
            return

        batch_col = self.batch_combo.currentText().strip()
        if not batch_col:
            self._set_status("Select a batch column.")
            return

        categorical, continuous = self._selected_confounds()
        categorical = [col for col in categorical if col != batch_col]
        continuous = [col for col in continuous if col != batch_col]
        apply_fisher = bool(self.fisher_checkbox.isChecked()) if hasattr(self, "fisher_checkbox") else False

        output_path = self._output_path()

        self._set_busy(True)
        self._set_status("Running neuroCombat harmonization...")
        self.run_summary_label.setText("Running...")
        if hasattr(self, "log_output") and self.log_output is not None:
            self.log_output.clear()
        if hasattr(self, "result_canvas") and self.result_canvas is not None:
            self.result_figure.clear()
            self.result_canvas.setVisible(False)
        self._append_terminal_line(
            (
                f"[HARMONIZE] Starting run for {self._source_path.name} "
                f"(key={self._matrix_key}, batch={batch_col}, fisher={apply_fisher})."
            )
        )

        try:
            run = self._harm_mod.run_harmonization(
                source_path=self._source_path,
                matrix_key=self._matrix_key,
                batch_col=batch_col,
                categorical_cols=categorical,
                continuous_cols=continuous,
                selected_indices=selected_rows,
                apply_fisher=apply_fisher,
                output_path=output_path,
                show_plots=False,
                log_fn=self._append_terminal_line,
            )
        except Exception as exc:
            self._set_busy(False)
            self._append_terminal_line(f"[HARMONIZE] Run failed: {exc}")
            self._set_status(f"Harmonization failed: {exc}")
            self.run_summary_label.setText("Run failed.")
            return

        self._last_run = run
        self._last_output_path = Path(run["output_path"])

        summary = run["result"]["summary"]
        self.run_summary_label.setText(
            f"Saved: {self._last_output_path}\n"
            f"N subjects: {summary.get('n_subjects')} | parcels: {summary.get('n_parcels')}\n"
            f"Edges: {summary.get('n_edges')} | upper triangle only: {summary.get('upper_triangle_only')} | "
            f"Fisher: {summary.get('apply_fisher')}\n"
            f"Batch: {summary.get('batch_col')} | counts: {summary.get('batch_counts')}\n"
            f"Confounds categorical: {', '.join(summary.get('categorical_cols') or []) or 'none'}\n"
            f"Confounds continuous: {', '.join(summary.get('continuous_cols') or []) or 'none'}\n"
            f"Mean original={summary.get('mean_original'):.4g}, harmonized={summary.get('mean_harmonized'):.4g}"
        )
        self._append_terminal_line(f"[HARMONIZE] Completed: {self._last_output_path}")

        try:
            self._harm_mod.create_harmonization_plots(
                original_stack=run["prepared"]["matrix_stack"],
                harmonized_stack=run["result"]["harmonized_stack"],
                batch_values=run["result"]["batch_values"],
                batch_col=run["result"]["batch_col"],
                figure=self.result_figure,
            )
            self.result_canvas.setVisible(True)
            self.result_canvas.draw_idle()
        except Exception as exc:
            self.result_figure.clear()
            self.result_canvas.setVisible(False)
            self._append_terminal_line(f"[HARMONIZE] Plotting failed: {exc}")
            self._set_status(f"Harmonization complete, but plotting failed: {exc}")

        self._set_busy(False)
        self._refresh_output_path_preview()
        self._set_status(f"Harmonization complete: {self._last_output_path.name}")

        if self._export_callback is not None:
            if self._ask_yes_no(
                "Export to workspace",
                "Harmonization finished. Export the result to the workspace now?",
            ):
                self._export_to_workspace()

    def _ask_yes_no(self, title: str, text: str) -> bool:
        if QT_LIB == 6:
            buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            choice = QMessageBox.question(
                self,
                title,
                text,
                buttons,
                QMessageBox.StandardButton.Yes,
            )
            return choice == QMessageBox.StandardButton.Yes
        choice = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return choice == QMessageBox.Yes

    def _export_to_workspace(self):
        if self._last_output_path is None:
            self._set_status("Run harmonization first.")
            return
        if self._export_callback is None:
            self._set_status("No workspace export callback configured.")
            return
        payload = {
            "output_path": str(self._last_output_path),
            "source_path": str(self._source_path),
            "matrix_key": self._matrix_key,
            "summary": self._last_run["result"].get("summary", {}) if self._last_run else {},
        }
        try:
            ok = bool(self._export_callback(payload))
        except Exception as exc:
            self._set_status(f"Workspace export failed: {exc}")
            return
        if ok:
            self._set_status("Exported harmonized result to workspace.")
        else:
            self._set_status("Workspace export was not applied.")

    def set_theme(self, theme_name):
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        toggle_style = (
            "QPushButton#tableToggleButton { "
            "min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; padding: 0px; "
            "background: transparent; color: transparent; border: 1px solid #94a3b8; border-radius: 5px; } "
            "QPushButton#tableToggleButton:hover { border: 1px solid #64748b; background: rgba(148, 163, 184, 0.12); } "
            "QPushButton#tableToggleButton:checked { background: #16a34a; color: transparent; border: 1px solid #86efac; } "
            "QPushButton#tableToggleButton:disabled { background: transparent; color: transparent; border: 1px solid #cbd5e1; } "
            "QPushButton#tableToggleButton:checked:disabled { background: #86efac; color: transparent; border: 1px solid #86efac; } "
            "QPushButton#tableExcludeButton { "
            "min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; padding: 0px; "
            "background: transparent; color: transparent; border: 1px solid #94a3b8; border-radius: 5px; } "
            "QPushButton#tableExcludeButton:hover { border: 1px solid #64748b; background: rgba(148, 163, 184, 0.12); } "
            "QPushButton#tableExcludeButton:checked { background: #dc2626; color: transparent; border: 1px solid #fca5a5; } "
            "QPushButton#tableExcludeButton:disabled { background: transparent; color: transparent; border: 1px solid #cbd5e1; } "
            "QPushButton#tableExcludeButton:checked:disabled { background: #fca5a5; color: transparent; border: 1px solid #fca5a5; } "
        )
        if theme == "Dark":
            style = (
                "QWidget { background: #1f2430; color: #e5e7eb; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #344054; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #2d3646; color: #e5e7eb; border: 1px solid #556070; } "
                "QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
            )
        elif theme == "Teya":
            style = (
                "QWidget { background: #ffd0e5; color: #0b7f7a; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #ffd9ea; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #ffc4df; color: #0b7f7a; border: 1px solid #1db8b2; } "
                "QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
            )
        elif theme == "Donald":
            style = (
                "QWidget { background: #d97706; color: #ffffff; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #c76b06; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #c96a04; color: #ffffff; border: 1px solid #f3a451; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        else:
            style = (
                "QWidget { background: #f4f6f9; color: #1f2937; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #ffffff; color: #1f2937; border: 1px solid #c9d0da; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #edf2f7; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #eef2f7; color: #1f2937; border: 1px solid #c9d0da; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        self.setStyleSheet(style + toggle_style)
        self._apply_terminal_style()


__all__ = ["HarmonizePrepareDialog"]
