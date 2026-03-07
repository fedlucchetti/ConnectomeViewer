#!/usr/bin/env python3
"""Prepare and stack selected connectivity NPZ files into a population stack."""

from pathlib import Path
import contextlib
import importlib.util
import os
import re
import sys
import traceback

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MRSI_TOOLBOX_ROOT = _REPO_ROOT / "mrsitoolbox"
for _path_entry in (str(_REPO_ROOT), str(_MRSI_TOOLBOX_ROOT)):
    if _path_entry not in sys.path:
        sys.path.insert(0, _path_entry)

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
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
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
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


if QT_LIB == 6:
    Qt.Checked = Qt.CheckState.Checked
    Qt.Unchecked = Qt.CheckState.Unchecked


NUMERIC_RANGE_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


def _ensure_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for covariate TSV loading.")


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_selectable_flag():
    return getattr(Qt, "ItemIsSelectable", getattr(Qt.ItemFlag, "ItemIsSelectable"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


def _is_editable_flag():
    return getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))


def _load_construct_matrix_pop_main():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "construct_matrix_pop.py"
    spec = importlib.util.spec_from_file_location("construct_matrix_pop_gui", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load construct_matrix_pop module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise ImportError("construct_matrix_pop.py does not define main")
    return module.main


class _GuiLogStream:
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        if text is None:
            return 0
        chunk = str(text)
        if chunk == "":
            return 0
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line)
        return len(chunk)

    def flush(self):
        if self._buffer:
            self._callback(self._buffer)
            self._buffer = ""


def _decode_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:
            return str(value)
    return value


def _display_text(value):
    value = _decode_scalar(value)
    if value is None:
        return ""
    return str(value)


def _column_is_numeric(values):
    has_value = False
    for value in values:
        text = _display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value


def _covars_to_rows(covars_df):
    if covars_df is None:
        return [], []
    columns = [str(col) for col in covars_df.columns]
    records = covars_df.to_dict(orient="records")
    rows = []
    for record in records:
        rows.append({col: _decode_scalar(record.get(col)) for col in columns})
    return columns, rows


def _slugify(text):
    token = str(text or "").strip()
    if not token:
        return "stack"
    chars = []
    for ch in token:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "stack"


def _infer_group_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "derivatives" in parts:
        idx = parts.index("derivatives")
        if idx > 0:
            return parts[idx - 1]
    return "group"


def _infer_modality_from_path(path: Path) -> str:
    lowered = str(path).lower()
    if "connectivity_mrsi" in lowered or "/mrsi/" in lowered:
        return "mrsi"
    if "connectivity_dwi" in lowered or "/dwi/" in lowered:
        return "dwi"
    if "connectivity_func" in lowered or "/func/" in lowered:
        return "func"
    return "connectivity"


def _extract_metadata(filename):
    base_filename = os.path.basename(str(filename))
    results = {}
    patterns = {
        "sub": r"sub-([^_]+)",
        "ses": r"ses-([^_]+)",
        "run": r"run-([^_]+)",
        "acq": r"acq-([^_]+)",
        "space": r"space-([^_]+)",
        "parcscheme": r"atlas-(?:chimera)?([^_]+)",
        "scale": r"scale(\d+)",
        "grow": r"grow(\d+)mm",
        "npert": r"npert[-_](\d+)",
        "filt": r"filt([^_]+)",
        "met": r"met-([^_]+)",
        "res": r"res-([^_]+)",
    }
    for key, pat in patterns.items():
        match = re.search(pat, base_filename)
        results[key] = match.group(1) if match else None
    for key in ("scale", "grow", "npert"):
        if results[key] is not None:
            results[key] = int(results[key])
    return results


def _group_root_from_path(path: Path) -> Path:
    current = Path(path).resolve()
    for parent in [current.parent] + list(current.parents):
        if parent.name == "derivatives":
            return parent.parent
    return current.parent


def _atlas_tag_from_path(path: Path) -> str:
    meta = _extract_metadata(str(path))
    base_name = Path(path).name
    atlas_match = re.search(r"atlas-([^_]+)", base_name)
    atlas = atlas_match.group(1).strip() if atlas_match else ""
    scale = meta.get("scale")
    if atlas and scale is not None and f"scale{scale}" not in atlas:
        return f"{atlas}_scale{scale}"
    if atlas:
        return atlas
    return "unknown"


class StackPrepareDialog(QDialog):
    def __init__(self, selected_paths, theme_name="Dark", export_callback=None, parent=None):
        super().__init__(parent)
        _ensure_pandas()
        self._selected_paths = [Path(path) for path in (selected_paths or [])]
        self._theme_name = "Dark"
        self._export_callback = export_callback
        self._covars_df = None
        self._covars_path = None
        self._covars_error = ""
        self._columns = []
        self._rows = []
        self._filtered_indices = []
        self._excluded_indices = set()
        self._table_refreshing = False
        self._output_path_auto = True
        self._processing = False
        self._last_output_path = None
        self._last_subjects_tsv_path = None
        self._last_gradients_path = None
        self._last_voxel_count_path = None
        self._detected_atlas = _atlas_tag_from_path(self._selected_paths[0]) if self._selected_paths else "unknown"
        self._detected_modality = (
            _infer_modality_from_path(self._selected_paths[0]) if self._selected_paths else "connectivity"
        )
        self.setWindowTitle("Stack Connectivity")
        self.resize(1080, 760)
        self.setAcceptDrops(True)
        self._build_ui()
        self._set_default_output_path(force=True)
        self.set_theme(theme_name)
        self._update_process_enabled()

    def _build_ui(self):
        root = QVBoxLayout(self)

        summary_group = QGroupBox("Selection")
        summary_layout = QVBoxLayout(summary_group)
        first_path = self._selected_paths[0] if self._selected_paths else None
        self.selection_label = QLabel(
            f"Selected NPZ files: {len(self._selected_paths)}"
            + (f"\nFirst file: {first_path}" if first_path else "")
            + (f"\nAtlas: {self._detected_atlas}" if self._selected_paths else "")
            + (f"\nModality: {self._detected_modality}" if self._selected_paths else "")
        )
        self.selection_label.setWordWrap(True)
        summary_layout.addWidget(self.selection_label)
        root.addWidget(summary_group)

        options_group = QGroupBox("Processing")
        options_layout = QGridLayout(options_group)
        options_layout.addWidget(QLabel("Ignore parcels"), 0, 0)
        self.ignore_parc_edit = QLineEdit("")
        self.ignore_parc_edit.setPlaceholderText("Comma-separated substrings, e.g. wm-, ventricle")
        options_layout.addWidget(self.ignore_parc_edit, 0, 1, 1, 3)

        options_layout.addWidget(QLabel("Qmask (optional)"), 1, 0)
        self.qmask_path_edit = QLineEdit("")
        self.qmask_path_edit.setPlaceholderText("Optional qmask .nii or .nii.gz path")
        options_layout.addWidget(self.qmask_path_edit, 1, 1, 1, 2)
        self.qmask_browse_button = QPushButton("Browse")
        self.qmask_browse_button.clicked.connect(self._browse_qmask_file)
        options_layout.addWidget(self.qmask_browse_button, 1, 3)

        options_layout.addWidget(QLabel("MRSI coverage"), 2, 0)
        self.mrsi_cov_spin = QDoubleSpinBox()
        self.mrsi_cov_spin.setRange(0.0, 1.0)
        self.mrsi_cov_spin.setSingleStep(0.01)
        self.mrsi_cov_spin.setDecimals(2)
        self.mrsi_cov_spin.setValue(0.67)
        options_layout.addWidget(self.mrsi_cov_spin, 2, 1)

        self.compute_gradients_check = QCheckBox("Compute gradients")
        self.compute_gradients_check.setChecked(True)
        options_layout.addWidget(self.compute_gradients_check, 2, 2, 1, 2)
        root.addWidget(options_group)

        self._sync_processing_options()

        covars_group = QGroupBox("Covariates TSV (Optional)")
        covars_layout = QVBoxLayout(covars_group)
        covars_top = QHBoxLayout()
        self.tsv_drop_label = QLabel("Drop a .tsv file here or use Browse.")
        self.tsv_drop_label.setWordWrap(True)
        self.tsv_drop_label.setStyleSheet("padding: 8px; border: 1px dashed #6b7280;")
        covars_top.addWidget(self.tsv_drop_label, 1)
        self.browse_covars_button = QPushButton("Browse TSV")
        self.browse_covars_button.clicked.connect(self._browse_covars_file)
        covars_top.addWidget(self.browse_covars_button)
        self.clear_covars_button = QPushButton("Clear TSV")
        self.clear_covars_button.clicked.connect(self._clear_covars_file)
        covars_top.addWidget(self.clear_covars_button)
        covars_layout.addLayout(covars_top)

        self.covars_path_label = QLabel("No covars file loaded.")
        self.covars_path_label.setWordWrap(True)
        covars_layout.addWidget(self.covars_path_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Covariate"))
        self.filter_covar_combo = QComboBox()
        self.filter_covar_combo.currentIndexChanged.connect(self._on_filter_inputs_changed)
        filter_row.addWidget(self.filter_covar_combo)
        filter_row.addWidget(QLabel("Value"))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText("e.g. control, 0,1 or 32-46")
        self.filter_value_edit.textChanged.connect(self._on_filter_inputs_changed)
        self.filter_value_edit.returnPressed.connect(self._apply_filter)
        filter_row.addWidget(self.filter_value_edit, 1)
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self._apply_filter)
        filter_row.addWidget(self.filter_button)
        self.filter_reset_button = QPushButton("Reset")
        self.filter_reset_button.clicked.connect(self._reset_filter)
        filter_row.addWidget(self.filter_reset_button)
        covars_layout.addLayout(filter_row)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_table_item_changed)
        if QT_LIB == 6:
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        covars_layout.addWidget(self.table, 1)

        self.showing_rows_label = QLabel("No covars filter applied.")
        covars_layout.addWidget(self.showing_rows_label)
        root.addWidget(covars_group, 1)

        output_group = QGroupBox("Output")
        output_layout = QGridLayout(output_group)
        output_layout.addWidget(QLabel("Output NPZ"), 0, 0)
        self.output_path_edit = QLineEdit("")
        self.output_path_edit.textEdited.connect(self._on_output_path_edited)
        output_layout.addWidget(self.output_path_edit, 0, 1)
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self._browse_output_path)
        output_layout.addWidget(self.output_browse_button, 0, 2)
        root.addWidget(output_group)

        log_group = QGroupBox("Integrated Terminal")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        if QT_LIB == 6:
            self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        else:
            self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_output.setPlaceholderText("Stack progress will appear here.")
        log_layout.addWidget(self.log_output, 1)
        root.addWidget(log_group, 1)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.import_button = QPushButton("Import to workspace")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_last_output)
        actions.addWidget(self.import_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self._process)
        actions.addWidget(self.process_button)
        root.addLayout(actions)

    def _sync_processing_options(self):
        is_mrsi = self._detected_modality == "mrsi"
        self.qmask_path_edit.setEnabled(is_mrsi)
        self.qmask_browse_button.setEnabled(is_mrsi)
        self.mrsi_cov_spin.setEnabled(is_mrsi)
        if not is_mrsi:
            self.qmask_path_edit.clear()

    def _browse_qmask_file(self):
        start_dir = str(self._selected_paths[0].parent) if self._selected_paths else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select qmask image",
            start_dir,
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if selected:
            self.qmask_path_edit.setText(selected)

    def _current_selection_suffix(self):
        covar = self.filter_covar_combo.currentText().strip()
        values = self._parse_filter_values(self.filter_value_edit.text())
        if not covar or not values:
            return "all", "all"
        joined = "-".join(values)
        return _slugify(covar), _slugify(joined)

    def _on_filter_inputs_changed(self, *_args):
        if self._output_path_auto:
            self._set_default_output_path(force=True)

    def _set_default_output_path(self, force=False):
        if not self._selected_paths:
            return
        if not force and not self._output_path_auto:
            return
        first = self._selected_paths[0]
        group_root = _group_root_from_path(first)
        group_name = _slugify(_infer_group_from_path(first))
        atlas_tag = _slugify(_atlas_tag_from_path(first))
        modality = _slugify(_infer_modality_from_path(first))
        filter_var, filter_values = self._current_selection_suffix()
        name = (
            f"{group_name}_atlas-{atlas_tag}_sel-{filter_var}_{filter_values}"
            f"_desc-group_connectivity_{modality}.npz"
        )
        output_dir = group_root / "derivatives" / "group" / "connectivity" / modality
        self.output_path_edit.setText(str(output_dir / name))
        self._output_path_auto = True

    def _on_output_path_edited(self, _text):
        self._output_path_auto = False
        self._update_process_enabled()

    def _browse_output_path(self):
        start_dir = str(self._selected_paths[0].parent) if self._selected_paths else str(Path.home())
        suggested = self.output_path_edit.text().strip() or str(Path(start_dir) / "stacked_connectivity.npz")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save stacked connectivity file",
            suggested,
            "NumPy archive (*.npz);;All files (*)",
        )
        if selected:
            if not selected.lower().endswith(".npz"):
                selected = f"{selected}.npz"
            self.output_path_edit.setText(selected)
            self._output_path_auto = False
        self._update_process_enabled()

    def _browse_covars_file(self):
        start_dir = str(self._selected_paths[0].parent) if self._selected_paths else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select covariates TSV",
            start_dir,
            "TSV files (*.tsv);;All files (*)",
        )
        if selected:
            self._load_covars_file(Path(selected))

    def _clear_covars_file(self):
        self._covars_df = None
        self._covars_path = None
        self._covars_error = ""
        self._columns = []
        self._rows = []
        self._filtered_indices = []
        self._excluded_indices = set()
        self.covars_path_label.setText("No covars file loaded.")
        self.tsv_drop_label.setText("Drop a .tsv file here or use Browse.")
        self.filter_covar_combo.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.showing_rows_label.setText("No covars filter applied.")
        self._set_status("Covars file cleared. All selected NPZ files will be used.")
        self._update_process_enabled()

    def _load_covars_file(self, path: Path):
        _ensure_pandas()
        if path.suffix.lower() != ".tsv":
            QMessageBox.warning(self, "Invalid Covars File", "Select a .tsv covariates file.")
            return
        df = pd.read_csv(path, sep="\t")
        self._covars_df = df
        self._covars_path = Path(path)
        self._columns, self._rows = _covars_to_rows(df)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._covars_error = ""
        sub_col = None
        ses_col = None
        if df is not None:
            col_map = {str(col).lower(): col for col in df.columns}
            sub_col = (
                col_map.get("participant_id")
                or col_map.get("subject_id")
                or col_map.get("subject")
                or col_map.get("sub")
                or col_map.get("id")
            )
            ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
        if not sub_col or not ses_col:
            self._covars_error = (
                "Covars file is missing participant_id/subject_id and/or session_id/session columns."
            )
        self.filter_covar_combo.clear()
        self.filter_covar_combo.addItems(self._columns)
        self._refresh_table()
        self.covars_path_label.setText(f"Loaded covars: {path}")
        if self._covars_error:
            self._set_status(self._covars_error)
        else:
            self._set_status(
                f"Loaded covars TSV with {len(self._rows)} rows. Filter rows before processing if needed."
            )
        self._update_process_enabled()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".tsv"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".tsv"):
                self._load_covars_file(Path(url.toLocalFile()))
                event.acceptProposedAction()
                return
        event.ignore()

    @staticmethod
    def _parse_filter_values(text):
        values = [token.strip() for token in str(text).split(",")]
        return [token for token in values if token != ""]

    @classmethod
    def _parse_numeric_filter_targets(cls, tokens):
        exact_targets = []
        range_targets = []
        for token in tokens:
            text = str(token).strip()
            if text == "":
                continue
            range_match = NUMERIC_RANGE_RE.match(text)
            if range_match:
                start = float(range_match.group(1))
                stop = float(range_match.group(2))
                low, high = (start, stop) if start <= stop else (stop, start)
                range_targets.append((low, high))
                continue
            exact_targets.append(float(text))
        return exact_targets, range_targets

    @staticmethod
    def _matches_numeric(value, exact_targets, range_targets):
        try:
            val = float(_display_text(value).strip())
        except Exception:
            return False
        if any(np.isclose(val, target) for target in exact_targets):
            return True
        for low, high in range_targets:
            if (val > low or np.isclose(val, low)) and (val < high or np.isclose(val, high)):
                return True
        return False

    @staticmethod
    def _matches_any(source_value, targets):
        text = _display_text(source_value).strip()
        if text == "":
            return False
        return text in targets

    def _apply_filter(self):
        if not self._columns:
            self._set_status("No covariates available to filter.")
            return
        covar_name = self.filter_covar_combo.currentText().strip()
        target_text = self.filter_value_edit.text().strip()
        if not covar_name:
            self._set_status("Select a covariate to filter.")
            return
        if target_text == "":
            self._set_status("Enter a covariate value to filter.")
            return
        target_values = self._parse_filter_values(target_text)
        if not target_values:
            self._set_status("Enter at least one filter value.")
            return

        values = [row.get(covar_name) for row in self._rows]
        numeric = _column_is_numeric(values)
        matched = []
        if numeric:
            try:
                exact_targets, range_targets = self._parse_numeric_filter_targets(target_values)
            except Exception:
                self._set_status("Numeric filters support values like 0,1 or ranges like 32-46.")
                return
            for row_idx, value in enumerate(values):
                if self._matches_numeric(value, exact_targets, range_targets):
                    matched.append(row_idx)
        else:
            for row_idx, value in enumerate(values):
                if self._matches_any(value, target_values):
                    matched.append(row_idx)
        self._filtered_indices = matched
        self._refresh_table()
        self._set_status(
            f"Filter applied: {covar_name} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )
        self._update_process_enabled()

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_table()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")
        self._update_process_enabled()

    def _refresh_table(self):
        self._table_refreshing = True
        self.table.blockSignals(True)
        self.table.setColumnCount(len(self._columns) + 1)
        self.table.setHorizontalHeaderLabels(["Exclude"] + list(self._columns))
        self.table.setRowCount(len(self._filtered_indices))
        enabled_flag = _is_enabled_flag()
        selectable_flag = _is_selectable_flag()
        checkable_flag = _is_user_checkable_flag()
        editable_flag = _is_editable_flag()
        for table_row, source_idx in enumerate(self._filtered_indices):
            row_data = self._rows[source_idx]
            include_item = QTableWidgetItem("")
            include_item.setFlags(enabled_flag | selectable_flag | checkable_flag)
            include_item.setCheckState(
                Qt.Checked if source_idx in self._excluded_indices else Qt.Unchecked
            )
            self.table.setItem(table_row, 0, include_item)
            for col_idx, covar_name in enumerate(self._columns, start=1):
                item = QTableWidgetItem(_display_text(row_data.get(covar_name)))
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(table_row, col_idx, item)
        if self.table.columnCount() > 0:
            header = self.table.horizontalHeader()
            if QT_LIB == 6:
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
                for col_idx in range(1, self.table.columnCount()):
                    header.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                for col_idx in range(1, self.table.columnCount()):
                    header.setSectionResizeMode(col_idx, QHeaderView.Stretch)
        self.table.blockSignals(False)
        self._table_refreshing = False
        self._update_showing_rows_label()

    def _on_table_item_changed(self, item):
        if item is None or self._table_refreshing:
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
        self._update_process_enabled()

    def _update_showing_rows_label(self):
        self.showing_rows_label.setText(
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows | "
            f"Included: {len(self.selected_row_indices())}"
        )

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]

    def _selected_covars_df(self):
        if self._covars_df is None:
            return None
        selected = self.selected_row_indices()
        if not selected:
            return self._covars_df.iloc[0:0].copy()
        return self._covars_df.iloc[selected].reset_index(drop=True)

    def _parse_ignore_parcels(self):
        values = [token.strip() for token in self.ignore_parc_edit.text().split(",")]
        return [token for token in values if token]

    def _update_process_enabled(self):
        required_ok = bool(self.output_path_edit.text().strip()) and bool(self._selected_paths)
        covars_ok = not self._covars_error
        if self._covars_df is not None:
            covars_ok = covars_ok and len(self.selected_row_indices()) > 0
        can_process = required_ok and covars_ok and not self._process_running()
        self.process_button.setEnabled(can_process)
        can_import = (
            not self._process_running()
            and self._export_callback is not None
            and bool(self._last_output_path)
            and Path(str(self._last_output_path)).is_file()
        )
        self.import_button.setEnabled(can_import)

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _process_running(self):
        return bool(self._processing)

    def _append_log(self, text):
        cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.rstrip("\n")
        if not cleaned:
            return
        for line in cleaned.split("\n"):
            self.log_output.appendPlainText(line)

    def _clear_last_result(self):
        self._last_output_path = None
        self._last_subjects_tsv_path = None
        self._last_gradients_path = None
        self._last_voxel_count_path = None

    def _build_process_kwargs(self):
        covars_df = self._selected_covars_df()
        if covars_df is not None and covars_df.empty:
            raise ValueError("No covariate rows remain after filtering.")

        return {
            "con_path_list": [str(path) for path in self._selected_paths],
            "covar_df": covars_df,
            "atlas": self._detected_atlas,
            "modality": self._detected_modality,
            "ignore_parc_list": self._parse_ignore_parcels(),
            "qmask_path": self.qmask_path_edit.text().strip() or None,
            "mrsi_cov": float(self.mrsi_cov_spin.value()),
            "comp_gradients": bool(self.compute_gradients_check.isChecked()),
            "matrix_outpath": self.output_path_edit.text().strip(),
        }

    def _append_processor_log(self, text):
        self._append_log(text)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _import_last_output(self):
        if not self._last_output_path:
            self._set_status("No completed stack output is available to import.")
            return
        if self._export_callback is None:
            self._set_status("No workspace import callback is configured.")
            return
        try:
            ok = bool(self._export_callback({"output_path": self._last_output_path}))
        except Exception as exc:
            self._set_status(f"Workspace import failed: {exc}")
            return
        if ok:
            self._set_status(f"Imported {Path(self._last_output_path).name} into the workspace.")
        else:
            self._set_status("The stacked file was written, but the workspace import was rejected.")
        self._update_process_enabled()

    def _process(self):
        if self._process_running():
            self._set_status("A stack process is already running.")
            return
        if not self.process_button.isEnabled():
            self._set_status("Complete the required selections before processing.")
            return

        try:
            process_kwargs = self._build_process_kwargs()
        except Exception as exc:
            self._set_status(f"Failed to prepare processing inputs: {exc}")
            return

        self._clear_last_result()
        self.log_output.clear()
        self._append_log("Starting construct_matrix_pop.main(...).")
        self._processing = True
        self._set_status("Processing started. Progress is shown in the integrated terminal.")
        self._update_process_enabled()

        try:
            construct_main = _load_construct_matrix_pop_main()
            log_stream = _GuiLogStream(self._append_processor_log)
            with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                summary = construct_main(**process_kwargs)
            log_stream.flush()
        except Exception as exc:
            self._append_log(traceback.format_exc())
            self._set_status(f"Processing failed: {exc}")
        else:
            message = str(summary).strip() if summary is not None else ""
            matrix_outpath = str(process_kwargs.get("matrix_outpath") or "").strip()
            if matrix_outpath and Path(matrix_outpath).is_file():
                self._last_output_path = matrix_outpath
                subjects_path = matrix_outpath.replace(".npz", "_subjects.tsv")
                if Path(subjects_path).is_file():
                    self._last_subjects_tsv_path = subjects_path
                if bool(process_kwargs.get("comp_gradients")):
                    gradients_path = matrix_outpath.replace("connectivity", "gradients")
                    if Path(gradients_path).is_file():
                        self._last_gradients_path = gradients_path
                if str(process_kwargs.get("modality") or "").strip().lower() == "mrsi":
                    voxel_count_path = (
                        str(Path(matrix_outpath).parent / Path(matrix_outpath).name)
                        .replace(f"connectivity_{process_kwargs.get('modality')}", "voxelcount_per_parcel")
                        .replace(".npz", ".csv")
                    )
                    if Path(voxel_count_path).is_file():
                        self._last_voxel_count_path = voxel_count_path
                if message:
                    self._append_log(message)
                    self._set_status(f"{message} Use 'Import to workspace' to load the output.")
                else:
                    self._set_status(
                        f"Saved {Path(matrix_outpath).name}. Use 'Import to workspace' to load the output."
                    )
            else:
                self._set_status(message or "Processing finished, but no output file was created.")
        finally:
            self._processing = False
            self._update_process_enabled()

    def closeEvent(self, event):
        super().closeEvent(event)

    def set_theme(self, theme_name):
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        self._theme_name = theme
        if theme == "Dark":
            style = (
                "QWidget { background: #1f2430; color: #e5e7eb; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget, QPlainTextEdit { "
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #344054; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #2d3646; color: #e5e7eb; border: 1px solid #556070; } "
                "QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
            )
        elif theme == "Teya":
            style = (
                "QWidget { background: #ffd0e5; color: #0b7f7a; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget, QPlainTextEdit { "
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #ffd9ea; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #ffc4df; color: #0b7f7a; border: 1px solid #1db8b2; } "
                "QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
            )
        elif theme == "Donald":
            style = (
                "QWidget { background: #a64b00; color: #ffffff; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget, QPlainTextEdit { "
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #db7a13; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #c96a04; color: #ffffff; border: 1px solid #f3a451; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        else:
            style = (
                "QWidget { background: #f5f7fb; color: #1f2937; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget, QPlainTextEdit { "
                "background: #ffffff; color: #1f2937; border: 1px solid #c9d0da; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #eef2f7; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #eef2f7; color: #1f2937; border: 1px solid #c9d0da; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        self.setStyleSheet(style)
        self.log_output.setStyleSheet("font-family: monospace;")
