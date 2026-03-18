#!/usr/bin/env python3
"""Popup dialog for gradients projection and network rendering."""

from __future__ import annotations

from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )


class GradientsPrepareDialog(QDialog):
    """Small modeless window for the gradients workflow."""

    def __init__(
        self,
        *,
        theme_name="Dark",
        component_count=4,
        colormap_names=None,
        current_colormap="",
        parcellation_path=None,
        open_parcellation_callback=None,
        compute_callback=None,
        save_callback=None,
        render_3d_callback=None,
        classify_callback=None,
        render_network_callback=None,
        matrix_changed_callback=None,
        component_changed_callback=None,
        colormap_changed_callback=None,
        hemisphere_changed_callback=None,
        surface_mesh_changed_callback=None,
        scatter_rotation_changed_callback=None,
        triangular_rgb_changed_callback=None,
        classification_fit_mode_changed_callback=None,
        triangular_color_order_changed_callback=None,
        classification_surface_mesh_changed_callback=None,
        classification_hemisphere_changed_callback=None,
        classification_colormap_changed_callback=None,
        classification_component_changed_callback=None,
        classification_x_axis_changed_callback=None,
        classification_y_axis_changed_callback=None,
        open_classification_adjacency_callback=None,
        remove_classification_adjacency_callback=None,
        network_component_changed_callback=None,
        rotation_changed_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._open_parcellation_callback = open_parcellation_callback
        self._compute_callback = compute_callback
        self._save_callback = save_callback
        self._render_3d_callback = render_3d_callback
        self._classify_callback = classify_callback
        self._render_network_callback = render_network_callback
        self._matrix_changed_callback = matrix_changed_callback
        self._component_changed_callback = component_changed_callback
        self._colormap_changed_callback = colormap_changed_callback
        self._hemisphere_changed_callback = hemisphere_changed_callback
        self._surface_mesh_changed_callback = surface_mesh_changed_callback
        self._scatter_rotation_changed_callback = scatter_rotation_changed_callback
        self._triangular_rgb_changed_callback = triangular_rgb_changed_callback
        self._classification_fit_mode_changed_callback = classification_fit_mode_changed_callback
        self._triangular_color_order_changed_callback = triangular_color_order_changed_callback
        self._classification_surface_mesh_changed_callback = classification_surface_mesh_changed_callback
        self._classification_hemisphere_changed_callback = classification_hemisphere_changed_callback
        self._classification_colormap_changed_callback = classification_colormap_changed_callback
        self._classification_component_changed_callback = classification_component_changed_callback
        self._classification_x_axis_changed_callback = classification_x_axis_changed_callback
        self._classification_y_axis_changed_callback = classification_y_axis_changed_callback
        self._open_classification_adjacency_callback = open_classification_adjacency_callback
        self._remove_classification_adjacency_callback = remove_classification_adjacency_callback
        self._network_component_changed_callback = network_component_changed_callback
        self._rotation_changed_callback = rotation_changed_callback
        self._busy = False
        self._can_compute = False
        self._has_results = False
        self._can_classify = False
        self._has_classification_adjacency = False

        self.setWindowTitle("Gradients")
        self.resize(580, 560)
        self._build_ui()
        self.set_matrix_source(None)
        self.set_component_count(component_count)
        self.set_colormap_names(colormap_names or [], current_colormap=current_colormap)
        self.set_parcellation_path(parcellation_path)
        self.set_hemisphere_mode("both")
        self.set_surface_mesh("fsaverage4")
        self.set_scatter_rotation("Default")
        self.set_triangular_rgb(False)
        self.set_classification_fit_mode("triangle")
        self.set_triangular_color_order("RBG")
        self.set_classification_surface_mesh("fsaverage4")
        self.set_classification_hemisphere_mode("both")
        self.set_classification_component_options(component_count, selected_component="1")
        self.set_classification_axes("gradient2", "gradient1")
        self.set_classification_adjacency_path(None)
        self.set_rotation_presets([])
        self.set_progress(0, 1, 0, "Idle")
        self.set_theme(theme_name)
        self._refresh_action_state()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        self.matrix_source_label = QLabel("Matrix: none")
        self.matrix_source_label.setWordWrap(True)
        root_layout.addWidget(self.matrix_source_label, 0)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)

        options_tab = QWidget()
        options_layout = QGridLayout(options_tab)
        row = 0
        options_layout.addWidget(QLabel("Workspace matrix"), row, 0)
        self.matrix_combo = QComboBox()
        self.matrix_combo.currentIndexChanged.connect(self._on_matrix_changed)
        options_layout.addWidget(self.matrix_combo, row, 1)
        row += 1

        options_layout.addWidget(QLabel("N components"), row, 0)
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 10)
        self.components_spin.setValue(4)
        self.components_spin.valueChanged.connect(self._on_component_changed)
        options_layout.addWidget(self.components_spin, row, 1)
        row += 1

        options_layout.addWidget(QLabel("Parcellation"), row, 0)
        self.parcellation_button = QPushButton("Set Parcellation")
        self.parcellation_button.clicked.connect(self._trigger_open_parcellation)
        options_layout.addWidget(self.parcellation_button, row, 1)
        row += 1

        self.parcellation_label = QLabel("Parcellation: none")
        self.parcellation_label.setWordWrap(True)
        options_layout.addWidget(self.parcellation_label, row, 0, 1, 2)
        row += 1

        compute_actions = QHBoxLayout()
        self.compute_button = QPushButton("Process")
        self.compute_button.clicked.connect(self._trigger_compute)
        compute_actions.addWidget(self.compute_button)
        self.save_button = QPushButton("Write to File")
        self.save_button.clicked.connect(self._trigger_save)
        compute_actions.addWidget(self.save_button)
        options_layout.addLayout(compute_actions, row, 0, 1, 2)
        options_layout.setRowStretch(row + 1, 1)
        self.tabs.addTab(options_tab, "Compute")

        display_tab = QWidget()
        display_layout = QGridLayout(display_tab)
        row = 0
        display_layout.addWidget(QLabel("3D colorbar"), row, 0)
        self.colorbar_combo = QComboBox()
        self.colorbar_combo.currentTextChanged.connect(self._on_colormap_changed)
        display_layout.addWidget(self.colorbar_combo, row, 1)
        row += 1

        display_layout.addWidget(QLabel("Hemisphere"), row, 0)
        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(["Both", "LH", "RH"])
        self.hemisphere_combo.currentTextChanged.connect(self._on_hemisphere_changed)
        display_layout.addWidget(self.hemisphere_combo, row, 1)
        row += 1

        display_layout.addWidget(QLabel("fsaverage scale"), row, 0)
        self.surface_mesh_combo = QComboBox()
        self.surface_mesh_combo.addItems(
            ["fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"]
        )
        self.surface_mesh_combo.currentTextChanged.connect(self._on_surface_mesh_changed)
        display_layout.addWidget(self.surface_mesh_combo, row, 1)
        row += 1

        self.render_3d_button = QPushButton("Render")
        self.render_3d_button.clicked.connect(self._trigger_render_3d)
        display_layout.addWidget(self.render_3d_button, row, 0, 1, 2)
        row += 1
        display_layout.setRowStretch(row, 1)

        self.tabs.addTab(display_tab, "fsaverage")

        classification_tab = QWidget()
        classification_layout = QGridLayout(classification_tab)
        row = 0
        classification_layout.addWidget(QLabel("Color scheme"), row, 0)
        self.classification_colorbar_combo = QComboBox()
        self.classification_colorbar_combo.currentTextChanged.connect(
            self._on_classification_colormap_changed
        )
        classification_layout.addWidget(self.classification_colorbar_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Gradient mapping"), row, 0)
        self.classification_component_combo = QComboBox()
        self.classification_component_combo.currentIndexChanged.connect(
            self._on_classification_component_changed
        )
        classification_layout.addWidget(self.classification_component_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("X axis"), row, 0)
        self.classification_x_axis_combo = QComboBox()
        self.classification_x_axis_combo.addItem("Gradient 1", "gradient1")
        self.classification_x_axis_combo.addItem("Gradient 2", "gradient2")
        self.classification_x_axis_combo.addItem("Spatial", "spatial")
        self.classification_x_axis_combo.currentIndexChanged.connect(
            self._on_classification_x_axis_changed
        )
        classification_layout.addWidget(self.classification_x_axis_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Y axis"), row, 0)
        self.classification_y_axis_combo = QComboBox()
        self.classification_y_axis_combo.addItem("Gradient 1", "gradient1")
        self.classification_y_axis_combo.addItem("Gradient 2", "gradient2")
        self.classification_y_axis_combo.addItem("Spatial", "spatial")
        self.classification_y_axis_combo.currentIndexChanged.connect(
            self._on_classification_y_axis_changed
        )
        classification_layout.addWidget(self.classification_y_axis_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("fsaverage scale"), row, 0)
        self.classification_surface_mesh_combo = QComboBox()
        self.classification_surface_mesh_combo.addItems(
            ["fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"]
        )
        self.classification_surface_mesh_combo.currentTextChanged.connect(
            self._on_classification_surface_mesh_changed
        )
        classification_layout.addWidget(self.classification_surface_mesh_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Hemisphere"), row, 0)
        self.classification_hemisphere_combo = QComboBox()
        self.classification_hemisphere_combo.addItems(["Both", "LH", "RH"])
        self.classification_hemisphere_combo.currentTextChanged.connect(
            self._on_classification_hemisphere_changed
        )
        classification_layout.addWidget(self.classification_hemisphere_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Scatter rotation"), row, 0)
        self.scatter_rotation_combo = QComboBox()
        self.scatter_rotation_combo.addItems(["Default", "+90", "-90", "180"])
        self.scatter_rotation_combo.currentTextChanged.connect(self._on_scatter_rotation_changed)
        classification_layout.addWidget(self.scatter_rotation_combo, row, 1)
        row += 1

        self.triangular_rgb_check = QCheckBox("Triangular RGB coloring")
        self.triangular_rgb_check.setObjectName("triangularRgbCheck")
        self.triangular_rgb_check.toggled.connect(self._on_triangular_rgb_changed)
        classification_layout.addWidget(self.triangular_rgb_check, row, 0, 1, 2)
        row += 1

        classification_layout.addWidget(QLabel("Fit mode"), row, 0)
        self.classification_fit_mode_combo = QComboBox()
        self.classification_fit_mode_combo.addItem("Triangular", "triangle")
        self.classification_fit_mode_combo.addItem("Square", "square")
        self.classification_fit_mode_combo.currentIndexChanged.connect(
            self._on_classification_fit_mode_changed
        )
        classification_layout.addWidget(self.classification_fit_mode_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("RGB order (top/left/right)"), row, 0)
        self.triangular_color_order_combo = QComboBox()
        self.triangular_color_order_combo.addItems(["RBG", "RGB", "BRG", "BGR", "GRB", "GBR"])
        self.triangular_color_order_combo.currentTextChanged.connect(
            self._on_triangular_color_order_changed
        )
        classification_layout.addWidget(self.triangular_color_order_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Adjacency"), row, 0)
        adjacency_actions = QHBoxLayout()
        self.classification_adjacency_button = QPushButton("Upload NPZ")
        self.classification_adjacency_button.clicked.connect(
            self._trigger_open_classification_adjacency
        )
        adjacency_actions.addWidget(self.classification_adjacency_button, 1)
        self.classification_adjacency_remove_button = QPushButton("Remove")
        self.classification_adjacency_remove_button.clicked.connect(
            self._trigger_remove_classification_adjacency
        )
        adjacency_actions.addWidget(self.classification_adjacency_remove_button, 0)
        classification_layout.addLayout(adjacency_actions, row, 1)
        row += 1

        self.classification_adjacency_label = QLabel("Adjacency: none")
        self.classification_adjacency_label.setWordWrap(True)
        classification_layout.addWidget(self.classification_adjacency_label, row, 0, 1, 2)
        row += 1

        self.classify_button = QPushButton("View")
        self.classify_button.clicked.connect(self._trigger_classify)
        classification_layout.addWidget(self.classify_button, row, 0, 1, 2)
        classification_layout.setRowStretch(row + 1, 1)
        self.tabs.addTab(classification_tab, "Classification")

        network_tab = QWidget()
        rotation_layout = QGridLayout(network_tab)
        row = 0
        rotation_layout.addWidget(QLabel("Display component"), row, 0)
        self.network_component_combo = QComboBox()
        self.network_component_combo.currentIndexChanged.connect(self._on_network_component_changed)
        rotation_layout.addWidget(self.network_component_combo, row, 1, 1, 3)
        row += 1

        rotation_layout.addWidget(QLabel("Per-component rotation"), row, 0, 1, 4)
        self._rotation_labels = []
        self._rotation_combos = []
        rotation_options = [
            "Default",
            "X +90",
            "X -90",
            "Y +90",
            "Y -90",
            "Y 180",
            "Z +90",
            "Z -90",
        ]
        for idx in range(10):
            grid_row = row + 1 + (idx // 2)
            grid_col = (idx % 2) * 2
            label = QLabel(f"C{idx + 1}")
            combo = QComboBox()
            combo.addItems(rotation_options)
            combo.currentTextChanged.connect(
                lambda value, comp_idx=idx: self._on_rotation_changed(comp_idx, value)
            )
            rotation_layout.addWidget(label, grid_row, grid_col)
            rotation_layout.addWidget(combo, grid_row, grid_col + 1)
            self._rotation_labels.append(label)
            self._rotation_combos.append(combo)

        network_button_row = row + 6
        self.render_network_button = QPushButton("Render")
        self.render_network_button.clicked.connect(self._trigger_render_network)
        rotation_layout.addWidget(self.render_network_button, network_button_row, 0, 1, 4)
        rotation_layout.setRowStretch(network_button_row + 1, 1)
        self.tabs.addTab(network_tab, "Network")

        progress_row = QHBoxLayout()
        progress_row.addWidget(QLabel("Progress"))
        self.progress_bar = QProgressBar()
        progress_row.addWidget(self.progress_bar, 1)
        root_layout.addLayout(progress_row)

    def _on_component_changed(self, value):
        if self._component_changed_callback is not None:
            try:
                self._component_changed_callback(int(value))
            except Exception:
                pass

    def _on_matrix_changed(self, _index):
        if self._matrix_changed_callback is not None:
            try:
                self._matrix_changed_callback(self.selected_matrix_entry_id())
            except Exception:
                pass

    def _on_colormap_changed(self, value):
        if self._colormap_changed_callback is not None:
            try:
                self._colormap_changed_callback(str(value))
            except Exception:
                pass

    def _on_hemisphere_changed(self, value):
        if self._hemisphere_changed_callback is not None:
            try:
                self._hemisphere_changed_callback(str(value))
            except Exception:
                pass

    def _on_surface_mesh_changed(self, value):
        if self._surface_mesh_changed_callback is not None:
            try:
                self._surface_mesh_changed_callback(str(value))
            except Exception:
                pass

    def _on_scatter_rotation_changed(self, value):
        if self._scatter_rotation_changed_callback is not None:
            try:
                self._scatter_rotation_changed_callback(str(value))
            except Exception:
                pass

    def _on_triangular_rgb_changed(self, checked):
        self._refresh_action_state()
        if self._triangular_rgb_changed_callback is not None:
            try:
                self._triangular_rgb_changed_callback(bool(checked))
            except Exception:
                pass

    def _on_classification_fit_mode_changed(self, _index):
        if self._classification_fit_mode_changed_callback is not None:
            try:
                self._classification_fit_mode_changed_callback(self.selected_classification_fit_mode())
            except Exception:
                pass

    def _on_triangular_color_order_changed(self, value):
        if self._triangular_color_order_changed_callback is not None:
            try:
                self._triangular_color_order_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_surface_mesh_changed(self, value):
        if self._classification_surface_mesh_changed_callback is not None:
            try:
                self._classification_surface_mesh_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_hemisphere_changed(self, value):
        if self._classification_hemisphere_changed_callback is not None:
            try:
                self._classification_hemisphere_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_colormap_changed(self, value):
        if self._classification_colormap_changed_callback is not None:
            try:
                self._classification_colormap_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_component_changed(self, _index):
        if self._classification_component_changed_callback is not None:
            try:
                self._classification_component_changed_callback(
                    self.selected_classification_component()
                )
            except Exception:
                pass

    def _on_classification_x_axis_changed(self, _index):
        if self._classification_x_axis_changed_callback is not None:
            try:
                self._classification_x_axis_changed_callback(self.selected_classification_x_axis())
            except Exception:
                pass

    def _on_classification_y_axis_changed(self, _index):
        if self._classification_y_axis_changed_callback is not None:
            try:
                self._classification_y_axis_changed_callback(self.selected_classification_y_axis())
            except Exception:
                pass

    def _on_rotation_changed(self, index, value):
        if self._rotation_changed_callback is not None:
            try:
                self._rotation_changed_callback(int(index), str(value))
            except Exception:
                pass

    def _on_network_component_changed(self, _index):
        if self._network_component_changed_callback is not None:
            try:
                self._network_component_changed_callback(self.selected_network_component())
            except Exception:
                pass

    def _trigger_open_parcellation(self):
        if self._open_parcellation_callback is not None:
            self._open_parcellation_callback()

    def _trigger_compute(self):
        if self._compute_callback is not None:
            self._compute_callback()

    def _trigger_save(self):
        if self._save_callback is not None:
            self._save_callback()

    def _trigger_render_3d(self):
        if self._render_3d_callback is not None:
            self._render_3d_callback()

    def _trigger_classify(self):
        if self._classify_callback is not None:
            self._classify_callback()

    def _trigger_open_classification_adjacency(self):
        if self._open_classification_adjacency_callback is not None:
            self._open_classification_adjacency_callback()

    def _trigger_remove_classification_adjacency(self):
        if self._remove_classification_adjacency_callback is not None:
            self._remove_classification_adjacency_callback()

    def _trigger_render_network(self):
        if self._render_network_callback is not None:
            self._render_network_callback()

    def component_count(self) -> int:
        return int(self.components_spin.value())

    def selected_matrix_entry_id(self):
        return self.matrix_combo.currentData()

    def selected_colormap(self) -> str:
        return self.colorbar_combo.currentText().strip()

    def selected_hemisphere(self) -> str:
        return self.hemisphere_combo.currentText().strip()

    def selected_surface_mesh(self) -> str:
        return self.surface_mesh_combo.currentText().strip()

    def selected_scatter_rotation(self) -> str:
        return self.scatter_rotation_combo.currentText().strip()

    def use_triangular_rgb(self) -> bool:
        return bool(self.triangular_rgb_check.isChecked())

    def selected_classification_fit_mode(self) -> str:
        value = self.classification_fit_mode_combo.currentData()
        if value is None:
            return "triangle"
        text = str(value).strip().lower()
        return text if text in {"triangle", "square"} else "triangle"

    def selected_classification_colormap(self) -> str:
        return self.classification_colorbar_combo.currentText().strip()

    def selected_triangular_color_order(self) -> str:
        return self.triangular_color_order_combo.currentText().strip().upper()

    def selected_classification_surface_mesh(self) -> str:
        return self.classification_surface_mesh_combo.currentText().strip()

    def selected_classification_hemisphere(self) -> str:
        return self.classification_hemisphere_combo.currentText().strip()

    def selected_classification_component(self) -> str:
        value = self.classification_component_combo.currentData()
        if value is None:
            return "1"
        return str(value).strip() or "1"

    def selected_classification_x_axis(self) -> str:
        value = self.classification_x_axis_combo.currentData()
        if value is None:
            return "gradient2"
        return str(value).strip() or "gradient2"

    def selected_classification_y_axis(self) -> str:
        value = self.classification_y_axis_combo.currentData()
        if value is None:
            return "gradient1"
        return str(value).strip() or "gradient1"

    def selected_network_component(self) -> str:
        value = self.network_component_combo.currentData()
        if value is None:
            return "all"
        return str(value).strip() or "all"

    def rotation_presets(self):
        return [combo.currentText().strip() for combo in self._rotation_combos]

    def set_component_count(self, value: int) -> None:
        try:
            value = int(value)
        except Exception:
            value = 4
        value = max(1, min(10, value))
        self.components_spin.blockSignals(True)
        self.components_spin.setValue(value)
        self.components_spin.blockSignals(False)
        self._refresh_rotation_visibility(value)

    def set_colormap_names(self, names, current_colormap="") -> None:
        current = str(current_colormap or "").strip()
        self.colorbar_combo.blockSignals(True)
        self.colorbar_combo.clear()
        self.classification_colorbar_combo.blockSignals(True)
        self.classification_colorbar_combo.clear()
        for name in names:
            text = str(name).strip()
            if text:
                self.colorbar_combo.addItem(text)
                self.classification_colorbar_combo.addItem(text)
        if current and self.colorbar_combo.findText(current) >= 0:
            self.colorbar_combo.setCurrentText(current)
        elif self.colorbar_combo.count() > 0:
            self.colorbar_combo.setCurrentIndex(0)
        if current and self.classification_colorbar_combo.findText(current) >= 0:
            self.classification_colorbar_combo.setCurrentText(current)
        elif self.classification_colorbar_combo.count() > 0:
            self.classification_colorbar_combo.setCurrentIndex(0)
        self.colorbar_combo.blockSignals(False)
        self.classification_colorbar_combo.blockSignals(False)

    def set_parcellation_path(self, path) -> None:
        if path is None:
            self.parcellation_label.setText("Parcellation: none")
            return
        try:
            name = Path(path).name
        except Exception:
            name = str(path)
        self.parcellation_label.setText(f"Parcellation: {name}")

    def set_matrix_source(self, text) -> None:
        label = str(text or "").strip()
        self.matrix_source_label.setText(f"Matrix: {label or 'none'}")

    def set_matrix_options(self, options, selected_entry_id=None) -> None:
        current_data = self.matrix_combo.currentData()
        target_id = selected_entry_id if selected_entry_id is not None else current_data
        self.matrix_combo.blockSignals(True)
        self.matrix_combo.clear()
        selected_index = -1
        for idx, item in enumerate(list(options or [])):
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                entry_id = item.get("id")
            else:
                try:
                    label, entry_id = item
                except Exception:
                    continue
                label = str(label).strip()
            if not label:
                label = str(entry_id or "matrix")
            self.matrix_combo.addItem(label, entry_id)
            if target_id is not None and entry_id == target_id:
                selected_index = idx
        if selected_index < 0 and self.matrix_combo.count() > 0:
            selected_index = 0
        if selected_index >= 0:
            self.matrix_combo.setCurrentIndex(selected_index)
        self.matrix_combo.blockSignals(False)

    def set_progress(self, minimum: int, maximum: int, value: int, text: str) -> None:
        self.progress_bar.setRange(int(minimum), int(maximum))
        self.progress_bar.setValue(int(value))
        self.progress_bar.setFormat(str(text or ""))

    def set_hemisphere_mode(self, value: str) -> None:
        text = str(value or "both").strip().upper()
        if text not in {"BOTH", "LH", "RH"}:
            text = "BOTH"
        display = "Both" if text == "BOTH" else text
        self.hemisphere_combo.blockSignals(True)
        if self.hemisphere_combo.findText(display) >= 0:
            self.hemisphere_combo.setCurrentText(display)
        self.hemisphere_combo.blockSignals(False)

    def set_surface_mesh(self, value: str) -> None:
        text = str(value or "fsaverage4").strip()
        if self.surface_mesh_combo.findText(text) < 0:
            text = "fsaverage4"
        self.surface_mesh_combo.blockSignals(True)
        self.surface_mesh_combo.setCurrentText(text)
        self.surface_mesh_combo.blockSignals(False)

    def set_scatter_rotation(self, value: str) -> None:
        text = str(value or "Default").strip()
        if self.scatter_rotation_combo.findText(text) < 0:
            text = "Default"
        self.scatter_rotation_combo.blockSignals(True)
        self.scatter_rotation_combo.setCurrentText(text)
        self.scatter_rotation_combo.blockSignals(False)

    def set_triangular_rgb(self, enabled: bool) -> None:
        self.triangular_rgb_check.blockSignals(True)
        self.triangular_rgb_check.setChecked(bool(enabled))
        self.triangular_rgb_check.blockSignals(False)

    def set_classification_fit_mode(self, value: str) -> None:
        text = str(value or "triangle").strip().lower()
        if text not in {"triangle", "square"}:
            text = "triangle"
        index = self.classification_fit_mode_combo.findData(text)
        if index < 0:
            index = self.classification_fit_mode_combo.findData("triangle")
        if index < 0:
            index = 0
        self.classification_fit_mode_combo.blockSignals(True)
        self.classification_fit_mode_combo.setCurrentIndex(index)
        self.classification_fit_mode_combo.blockSignals(False)

    def set_triangular_color_order(self, value: str) -> None:
        text = str(value or "RBG").strip().upper()
        if self.triangular_color_order_combo.findText(text) < 0:
            text = "RBG"
        self.triangular_color_order_combo.blockSignals(True)
        self.triangular_color_order_combo.setCurrentText(text)
        self.triangular_color_order_combo.blockSignals(False)

    def set_classification_surface_mesh(self, value: str) -> None:
        text = str(value or "fsaverage4").strip()
        if self.classification_surface_mesh_combo.findText(text) < 0:
            text = "fsaverage4"
        self.classification_surface_mesh_combo.blockSignals(True)
        self.classification_surface_mesh_combo.setCurrentText(text)
        self.classification_surface_mesh_combo.blockSignals(False)

    def set_classification_hemisphere_mode(self, value: str) -> None:
        text = str(value or "both").strip().upper()
        if text not in {"BOTH", "LH", "RH"}:
            text = "BOTH"
        display = "Both" if text == "BOTH" else text
        self.classification_hemisphere_combo.blockSignals(True)
        if self.classification_hemisphere_combo.findText(display) >= 0:
            self.classification_hemisphere_combo.setCurrentText(display)
        self.classification_hemisphere_combo.blockSignals(False)

    def set_classification_adjacency_path(self, path) -> None:
        if path is None or not str(path).strip():
            self._has_classification_adjacency = False
            self.classification_adjacency_label.setText("Adjacency: none")
            return
        try:
            name = Path(path).name
        except Exception:
            name = str(path)
        self._has_classification_adjacency = True
        self.classification_adjacency_label.setText(f"Adjacency: {name}")

    def set_classification_colormap(self, value: str) -> None:
        text = str(value or "").strip()
        self.classification_colorbar_combo.blockSignals(True)
        if text and self.classification_colorbar_combo.findText(text) >= 0:
            self.classification_colorbar_combo.setCurrentText(text)
        elif self.classification_colorbar_combo.count() > 0:
            self.classification_colorbar_combo.setCurrentIndex(0)
        self.classification_colorbar_combo.blockSignals(False)

    def set_classification_component_options(self, component_count: int, selected_component="1") -> None:
        count = max(1, min(10, int(component_count)))
        selected = str(selected_component or "1").strip()
        self.classification_component_combo.blockSignals(True)
        self.classification_component_combo.clear()
        selected_index = 0
        for idx in range(count):
            value = str(idx + 1)
            self.classification_component_combo.addItem(f"C{idx + 1}", value)
            if value == selected:
                selected_index = idx
        self.classification_component_combo.setCurrentIndex(selected_index)
        self.classification_component_combo.blockSignals(False)

    def set_classification_axes(self, x_axis="gradient2", y_axis="gradient1") -> None:
        self._set_classification_axis_combo(self.classification_x_axis_combo, x_axis, "gradient2")
        self._set_classification_axis_combo(self.classification_y_axis_combo, y_axis, "gradient1")

    @staticmethod
    def _set_classification_axis_combo(combo, value, fallback) -> None:
        selected = str(value or fallback).strip().lower()
        if not selected:
            selected = fallback
        index = combo.findData(selected)
        if index < 0:
            index = combo.findData(fallback)
        if index < 0:
            index = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def set_rotation_presets(self, presets) -> None:
        values = [str(value or "Default").strip() or "Default" for value in list(presets or [])]
        while len(values) < len(self._rotation_combos):
            values.append("Default")
        for combo, value in zip(self._rotation_combos, values):
            combo.blockSignals(True)
            if combo.findText(value) >= 0:
                combo.setCurrentText(value)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._refresh_rotation_visibility(self.component_count())

    def set_network_component_options(self, component_count: int, selected_component="all") -> None:
        count = max(1, min(10, int(component_count)))
        selected = str(selected_component or "all").strip().lower()
        ordinals = [
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Sixth",
            "Seventh",
            "Eighth",
            "Ninth",
            "Tenth",
        ]
        self.network_component_combo.blockSignals(True)
        self.network_component_combo.clear()
        self.network_component_combo.addItem("All", "all")
        selected_index = 0
        for idx in range(count):
            value = str(idx + 1)
            self.network_component_combo.addItem(f"{ordinals[idx]} (C{idx + 1})", value)
            if value == selected:
                selected_index = idx + 1
        self.network_component_combo.setCurrentIndex(selected_index)
        self.network_component_combo.blockSignals(False)

    def _refresh_rotation_visibility(self, component_count: int) -> None:
        count = max(1, min(10, int(component_count)))
        for idx, (label, combo) in enumerate(zip(self._rotation_labels, self._rotation_combos)):
            visible = idx < count
            label.setVisible(visible)
            combo.setVisible(visible)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._refresh_action_state()

    def set_can_compute(self, enabled: bool) -> None:
        self._can_compute = bool(enabled)
        self._refresh_action_state()

    def set_has_results(self, enabled: bool) -> None:
        self._has_results = bool(enabled)
        self._refresh_action_state()

    def set_can_classify(self, enabled: bool) -> None:
        self._can_classify = bool(enabled)
        self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        can_interact = not self._busy
        self.compute_button.setEnabled(self._can_compute and can_interact)
        self.matrix_combo.setEnabled(can_interact)
        self.components_spin.setEnabled(can_interact)
        self.parcellation_button.setEnabled(can_interact)
        self.hemisphere_combo.setEnabled(can_interact)
        self.surface_mesh_combo.setEnabled(can_interact)
        self.scatter_rotation_combo.setEnabled(can_interact)
        self.triangular_rgb_check.setEnabled(can_interact)
        self.classification_fit_mode_combo.setEnabled(can_interact)
        self.triangular_color_order_combo.setEnabled(can_interact)
        self.classification_surface_mesh_combo.setEnabled(can_interact)
        self.classification_hemisphere_combo.setEnabled(can_interact)
        self.classification_colorbar_combo.setEnabled(can_interact)
        self.classification_component_combo.setEnabled(can_interact)
        self.classification_x_axis_combo.setEnabled(can_interact)
        self.classification_y_axis_combo.setEnabled(can_interact)
        self.classification_adjacency_button.setEnabled(can_interact)
        self.classification_adjacency_remove_button.setEnabled(
            can_interact and self._has_classification_adjacency
        )
        self.network_component_combo.setEnabled(can_interact)
        for combo in self._rotation_combos:
            combo.setEnabled(can_interact)
        self.save_button.setEnabled(self._has_results and can_interact)
        self.render_3d_button.setEnabled(self._has_results and can_interact)
        self.classify_button.setEnabled(
            self._has_results
            and self._can_classify
            and can_interact
        )
        self.render_network_button.setEnabled(self._has_results and can_interact)

    def set_theme(self, theme_name="Dark"):
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        if theme == "Dark":
            style = (
                "QDialog, QWidget { background: #1f2430; color: #e5e7eb; }"
                "QGroupBox { border: 1px solid #5f6d82; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: 600; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QPushButton, QComboBox, QSpinBox { background: #2d3646; color: #e5e7eb; border: 1px solid #5f6d82; border-radius: 6px; padding: 5px 8px; }"
                "QCheckBox#triangularRgbCheck::indicator { width: 14px; height: 14px; border: 1px solid #5f6d82; border-radius: 2px; background: #2d3646; }"
                "QCheckBox#triangularRgbCheck::indicator:checked { background: #22c55e; border: 1px solid #15803d; }"
                "QTabWidget::pane { border: 1px solid #5f6d82; border-radius: 8px; top: -1px; }"
                "QTabBar::tab { background: #2d3646; color: #e5e7eb; border: 1px solid #5f6d82; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }"
                "QTabBar::tab:selected { background: #2563eb; color: #ffffff; }"
                "QPushButton:hover { background: #374256; }"
                "QProgressBar { border: 1px solid #5f6d82; border-radius: 6px; text-align: center; background: #2d3646; }"
                "QProgressBar::chunk { background: #2563eb; border-radius: 5px; }"
            )
        elif theme == "Teya":
            style = (
                "QDialog, QWidget { background: #ffd0e5; color: #0b7f7a; }"
                "QGroupBox { border: 1px solid #1db8b2; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: 700; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QPushButton, QComboBox, QSpinBox { background: #ffc0dc; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 6px; padding: 5px 8px; }"
                "QCheckBox#triangularRgbCheck::indicator { width: 14px; height: 14px; border: 1px solid #1db8b2; border-radius: 2px; background: #ffe6f1; }"
                "QCheckBox#triangularRgbCheck::indicator:checked { background: #22c55e; border: 1px solid #15803d; }"
                "QTabWidget::pane { border: 1px solid #1db8b2; border-radius: 8px; top: -1px; }"
                "QTabBar::tab { background: #ffc0dc; color: #0b7f7a; border: 1px solid #1db8b2; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }"
                "QTabBar::tab:selected { background: #2ecfc9; color: #073f3c; }"
                "QPushButton:hover { background: #ffb1d5; }"
                "QProgressBar { border: 1px solid #1db8b2; border-radius: 6px; text-align: center; background: #ffc0dc; }"
                "QProgressBar::chunk { background: #2ecfc9; border-radius: 5px; }"
            )
        elif theme == "Donald":
            style = (
                "QDialog, QWidget { background: #d97706; color: #ffffff; }"
                "QGroupBox { border: 1px solid #f3a451; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: 700; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QPushButton, QComboBox, QSpinBox { background: #b85f00; color: #ffffff; border: 1px solid #f3a451; border-radius: 6px; padding: 5px 8px; }"
                "QCheckBox#triangularRgbCheck::indicator { width: 14px; height: 14px; border: 1px solid #f3a451; border-radius: 2px; background: #c96a04; }"
                "QCheckBox#triangularRgbCheck::indicator:checked { background: #22c55e; border: 1px solid #15803d; }"
                "QTabWidget::pane { border: 1px solid #f3a451; border-radius: 8px; top: -1px; }"
                "QTabBar::tab { background: #b85f00; color: #ffffff; border: 1px solid #f3a451; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }"
                "QTabBar::tab:selected { background: #ffd19e; color: #7c2d12; }"
                "QPushButton:hover { background: #c76b06; }"
                "QProgressBar { border: 1px solid #f3a451; border-radius: 6px; text-align: center; background: #b85f00; }"
                "QProgressBar::chunk { background: #ffd19e; border-radius: 5px; }"
            )
        else:
            style = (
                "QDialog, QWidget { background: #f4f6f9; color: #1f2937; }"
                "QGroupBox { border: 1px solid #b7c0cc; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: 600; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QPushButton, QComboBox, QSpinBox { background: #ffffff; color: #1f2937; border: 1px solid #b7c0cc; border-radius: 6px; padding: 5px 8px; }"
                "QCheckBox#triangularRgbCheck::indicator { width: 14px; height: 14px; border: 1px solid #b7c0cc; border-radius: 2px; background: #ffffff; }"
                "QCheckBox#triangularRgbCheck::indicator:checked { background: #22c55e; border: 1px solid #15803d; }"
                "QTabWidget::pane { border: 1px solid #b7c0cc; border-radius: 8px; top: -1px; }"
                "QTabBar::tab { background: #ffffff; color: #1f2937; border: 1px solid #b7c0cc; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }"
                "QTabBar::tab:selected { background: #2563eb; color: #ffffff; }"
                "QPushButton:hover { background: #edf2f7; }"
                "QProgressBar { border: 1px solid #b7c0cc; border-radius: 6px; text-align: center; background: #ffffff; }"
                "QProgressBar::chunk { background: #2563eb; border-radius: 5px; }"
            )
        self.setStyleSheet(style)
