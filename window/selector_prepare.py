#!/usr/bin/env python3
"""Selector preparation dialog for filtered matrix aggregation."""

from pathlib import Path
import re

import numpy as np

try:
    from window.shared.covariate_table import (
        CovariateFilterPage as _CovariateFilterPage,
        filter_row_indices as _filter_row_indices,
        parse_filter_values as _parse_filter_values,
    )
    from window.shared.covars import (
        column_is_numeric as _column_is_numeric,
        covars_to_rows as _covars_to_rows,
        display_text as _display_text,
    )
    from window.shared.matrix import stack_axis as _stack_axis
    from window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag as _is_editable_flag,
        is_enabled_flag as _is_enabled_flag,
        is_selectable_flag as _is_selectable_flag,
        is_user_checkable_flag as _is_user_checkable_flag,
    )
    from window.shared.workflow import WorkflowShell as _WorkflowShell
    from window.shared.theme import workflow_dialog_stylesheet as _workflow_dialog_stylesheet
except Exception:
    from mrsi_viewer.window.shared.covariate_table import (
        CovariateFilterPage as _CovariateFilterPage,
        filter_row_indices as _filter_row_indices,
        parse_filter_values as _parse_filter_values,
    )
    from mrsi_viewer.window.shared.covars import (
        column_is_numeric as _column_is_numeric,
        covars_to_rows as _covars_to_rows,
        display_text as _display_text,
    )
    from mrsi_viewer.window.shared.matrix import stack_axis as _stack_axis
    from mrsi_viewer.window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag as _is_editable_flag,
        is_enabled_flag as _is_enabled_flag,
        is_selectable_flag as _is_selectable_flag,
        is_user_checkable_flag as _is_user_checkable_flag,
    )
    from mrsi_viewer.window.shared.workflow import WorkflowShell as _WorkflowShell
    from mrsi_viewer.window.shared.theme import workflow_dialog_stylesheet as _workflow_dialog_stylesheet

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5


install_qt_compat_aliases(Qt, QT_LIB)


class SelectorPrepareDialog(QDialog):
    """Popup dialog to filter subjects and aggregate matrix stacks."""

    STEP_TITLES = ("Data", "Aggregate", "Export")
    NUMERIC_RANGE_RE = re.compile(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
    )

    def __init__(
        self,
        covars_info,
        source_path,
        matrix_key,
        theme_name="Dark",
        export_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._matrix_key = str(matrix_key)
        self._columns, self._rows = _covars_to_rows(covars_info)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._data_table_refreshing = False
        self._current_step = 0
        self._aggregated_matrix = None
        self._aggregated_method = None
        self._export_callback = export_callback

        self.setWindowTitle("Selector Prepare")
        self.resize(1080, 760)
        self._build_ui()
        self.set_theme(theme_name)
        self._refresh_table()
        self._go_to_step(0)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        self.workflow_shell = _WorkflowShell(self.STEP_TITLES, on_step_selected=self._go_to_step)
        self._step_buttons = self.workflow_shell.step_buttons
        self.step_stack = self.workflow_shell.step_stack
        self.workflow_shell.add_step(self._build_step_data())
        self.workflow_shell.add_step(self._build_step_aggregate())
        self.workflow_shell.add_step(self._build_step_export())
        root_layout.addWidget(self.workflow_shell, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(False)
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
        self.data_page = _CovariateFilterPage(self._columns, exclude_mode="checkbox_item")
        self.dataset_summary_label = self.data_page.dataset_summary_label
        self.filter_covar_combo = self.data_page.filter_covar_combo
        self.filter_value_edit = self.data_page.filter_value_edit
        self.filter_button = self.data_page.filter_button
        self.filter_reset_button = self.data_page.filter_reset_button
        self.table = self.data_page.table
        self.showing_rows_label = self.data_page.showing_rows_label
        self.data_page.set_dataset_summary(
            self._source_path.name,
            self._matrix_key,
            len(self._rows),
            len(self._columns),
        )
        self.data_page.set_filter_callbacks(self._apply_filter, self._reset_filter)
        self.table.itemChanged.connect(self._on_table_item_changed)
        return self.data_page

    def _build_step_aggregate(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Aggregator")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("Method"), 0, 0)
        self.aggregate_method_combo = QComboBox()
        self.aggregate_method_combo.addItem("Mean (average)", "mean")
        self.aggregate_method_combo.addItem("Fisher Z (arctanh mean tanh)", "zfisher")
        grid.addWidget(self.aggregate_method_combo, 0, 1)
        self.aggregate_button = QPushButton("Aggregate")
        self.aggregate_button.clicked.connect(self._aggregate_selected)
        grid.addWidget(self.aggregate_button, 0, 2)
        layout.addWidget(group)

        self.aggregate_status_label = QLabel("No aggregation computed yet.")
        self.aggregate_status_label.setWordWrap(True)
        layout.addWidget(self.aggregate_status_label)
        layout.addStretch(1)
        return page

    def _build_step_export(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        summary_group = QGroupBox("Export Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.export_summary_label = QLabel("Run aggregation first.")
        self.export_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.export_summary_label)
        layout.addWidget(summary_group)

        export_row = QHBoxLayout()
        self.export_button = QPushButton("Export To Workspace")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_to_workspace)
        export_row.addWidget(self.export_button)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        layout.addStretch(1)
        return page

    def _go_to_step(self, step_index):
        step = self.workflow_shell.set_current_step(step_index)
        self._current_step = step
        self.back_button.setEnabled(step > 0)
        self.next_button.setVisible(step < (len(self.STEP_TITLES) - 1))

    def _go_next_step(self):
        self._go_to_step(self._current_step + 1)

    def _go_prev_step(self):
        self._go_to_step(self._current_step - 1)

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _on_table_item_changed(self, item):
        if item is None or self._data_table_refreshing:
            return
        if item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self._filtered_indices):
            return
        source_idx = self._filtered_indices[row]
        if item.checkState() == Qt.Checked:
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
        self._refresh_table()
        self._set_status(
            f"Filter applied: {self.filter_covar_combo.currentText().strip()} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_table()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")

    def _refresh_table(self):
        self._data_table_refreshing = True
        self.data_page.populate_rows(
            self._filtered_indices,
            self._rows,
            self._excluded_indices,
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

    def _load_subject_stack(self):
        with np.load(self._source_path, allow_pickle=True) as npz:
            if self._matrix_key not in npz:
                raise ValueError(f"Matrix key '{self._matrix_key}' not found in source file.")
            raw_matrix = np.asarray(npz[self._matrix_key], dtype=float)
        if raw_matrix.ndim != 3:
            raise ValueError("Aggregation requires a 3D matrix stack.")
        axis = _stack_axis(raw_matrix.shape)
        if axis is None:
            raise ValueError("Selected matrix is not a stack of square matrices.")
        if raw_matrix.shape[axis] != len(self._rows):
            raise ValueError("Covars length does not match matrix stack size.")
        selected = np.asarray(self.selected_row_indices(), dtype=int)
        if selected.size == 0:
            raise ValueError("No rows selected for aggregation.")

        if axis == 0:
            stack = raw_matrix[selected, :, :]
        elif axis == 1:
            stack = raw_matrix[:, selected, :].transpose(1, 0, 2)
        else:
            stack = raw_matrix[:, :, selected].transpose(2, 0, 1)
        return np.asarray(stack, dtype=float), selected

    def _aggregate_selected(self):
        try:
            stack, selected = self._load_subject_stack()
        except Exception as exc:
            self._set_status(f"Aggregation failed: {exc}")
            return

        method = str(self.aggregate_method_combo.currentData() or "mean")
        if method == "zfisher":
            clipped = np.clip(stack, -0.999999, 0.999999)
            with np.errstate(invalid="ignore"):
                z_stack = np.arctanh(clipped)
            z_mean = np.nanmean(z_stack, axis=0)
            aggregated = np.tanh(z_mean)
            method_label = "Fisher Z"
        else:
            aggregated = np.nanmean(stack, axis=0)
            method_label = "Mean"

        aggregated = np.asarray(aggregated, dtype=float)
        if aggregated.ndim != 2 or aggregated.shape[0] != aggregated.shape[1]:
            self._set_status("Aggregation failed: output is not a square matrix.")
            return
        aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)

        self._aggregated_matrix = aggregated
        self._aggregated_method = method
        self.aggregate_status_label.setText(
            f"Computed {method_label} aggregation with {selected.size} selected rows."
        )
        self.export_summary_label.setText(
            f"Source: {self._source_path.name}\n"
            f"Key: {self._matrix_key}\n"
            f"Method: {method_label}\n"
            f"Selected rows: {selected.size}/{len(self._rows)}\n"
            f"Output shape: {aggregated.shape[0]} x {aggregated.shape[1]}"
        )
        self.export_button.setEnabled(True)
        self._set_status("Aggregation complete. Go to Export to import into workspace.")
        self._go_to_step(2)

    def _export_to_workspace(self):
        if self._aggregated_matrix is None:
            self._set_status("Run aggregation first.")
            return
        payload = {
            "matrix": np.asarray(self._aggregated_matrix, dtype=float),
            "method": self._aggregated_method or "mean",
            "source_path": str(self._source_path),
            "matrix_key": self._matrix_key,
            "selected_rows": self.selected_row_indices(),
            "n_total_rows": len(self._rows),
            "filter_covar": self.filter_covar_combo.currentText().strip(),
            "filter_values": _parse_filter_values(self.filter_value_edit.text()),
        }
        if self._export_callback is None:
            self._set_status("No export callback defined.")
            return
        try:
            ok = bool(self._export_callback(payload))
        except Exception as exc:
            self._set_status(f"Export failed: {exc}")
            return
        if ok:
            self._set_status("Exported aggregated matrix to workspace.")
            self.accept()
        else:
            self._set_status("Export was not applied.")

    def set_theme(self, theme_name):
        _theme, style = _workflow_dialog_stylesheet(theme_name)
        self.setStyleSheet(style)


__all__ = ["SelectorPrepareDialog"]
