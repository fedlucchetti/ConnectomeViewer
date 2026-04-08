#!/usr/bin/env python3
"""Gradient scatter and classification dialogs."""

import heapq
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path
from warnings import warn

import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from nilearn import surface

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
    )
except Exception:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
    )

try:
    from window.shared.theme import dialog_theme_stylesheet as _shared_dialog_theme_stylesheet
except Exception:
    from mrsi_viewer.window.shared.theme import dialog_theme_stylesheet as _shared_dialog_theme_stylesheet

try:
    from .gradient_free_energy import GradientFreeEnergyDialog
    from .gradient_surface import GradientSurfaceDialog
except Exception:
    from gradient_free_energy import GradientFreeEnergyDialog
    from gradient_surface import GradientSurfaceDialog


def _dialog_theme_stylesheet(theme_name="Dark"):
    return _shared_dialog_theme_stylesheet(theme_name)


class GradientScatterDialog(QDialog):
    """Interactive scatter viewer for arbitrary gradient or spatial axes."""

    def __init__(
        self,
        x_values,
        y_values,
        *,
        color_values=None,
        point_labels=None,
        point_ids=None,
        title="Gradient Scatter",
        x_label="X axis",
        y_label="Y axis",
        color_label="Gradient 1",
        gradient1_values=None,
        path_metric_coords=None,
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        hemisphere_mode="both",
        rotation_preset="Default",
        use_triangular_rgb=False,
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
        edge_pairs=None,
        edge_color="#111827",
        edge_alpha=0.16,
        edge_linewidth=0.45,
        point_group_codes=None,
        show_proximity_circles=False,
        initial_proximity_slider_value=0,
        use_line_proximity_energy=True,
        project_paths_callback=None,
        export_metadata=None,
    ):
        super().__init__(parent)
        self._x = np.asarray(x_values, dtype=float).reshape(-1)
        self._y = np.asarray(y_values, dtype=float).reshape(-1)
        if self._x.shape != self._y.shape:
            raise ValueError("Gradient scatter axes must have matching lengths.")
        color_data = self._y if color_values is None else np.asarray(color_values, dtype=float).reshape(-1)
        if color_data.shape != self._x.shape:
            raise ValueError("Gradient scatter color data must match the axis lengths.")
        if gradient1_values is None:
            gradient1_data = np.asarray(color_data, dtype=float).reshape(-1)
        else:
            gradient1_data = np.asarray(gradient1_values, dtype=float).reshape(-1)
            if gradient1_data.shape != self._x.shape:
                raise ValueError("Gradient scatter Gradient 1 data must match the axis lengths.")
        if point_labels is None:
            label_data = np.asarray([f"Point {idx + 1}" for idx in range(self._x.size)], dtype=object)
        else:
            label_data = np.asarray(point_labels, dtype=object).reshape(-1)
            if label_data.shape != self._x.shape:
                raise ValueError("Gradient scatter point labels must match the axis lengths.")
        if point_ids is None:
            point_id_data = np.arange(1, self._x.size + 1, dtype=int)
        else:
            point_id_data = np.asarray(point_ids, dtype=object).reshape(-1)
            if point_id_data.shape != self._x.shape:
                raise ValueError("Gradient scatter point ids must match the axis lengths.")
        if point_group_codes is None:
            group_data = np.full(self._x.shape, -1, dtype=int)
        else:
            group_data = np.asarray(point_group_codes, dtype=int).reshape(-1)
            if group_data.shape != self._x.shape:
                raise ValueError("Gradient scatter point hemisphere codes must match the axis lengths.")
        finite_mask = np.isfinite(self._x) & np.isfinite(self._y) & np.isfinite(color_data) & np.isfinite(gradient1_data)
        metric_data = None
        if path_metric_coords is not None:
            metric_data = np.asarray(path_metric_coords, dtype=float)
            if metric_data.ndim == 1:
                metric_data = metric_data[:, np.newaxis]
            if metric_data.ndim != 2 or metric_data.shape[0] != self._x.shape[0]:
                raise ValueError("Gradient scatter path metric coordinates must match the axis lengths.")
            finite_mask &= np.all(np.isfinite(metric_data), axis=1)
        if not np.any(finite_mask):
            raise ValueError("Gradient scatter requires finite data points.")
        self._x = self._x[finite_mask]
        self._y = self._y[finite_mask]
        self._color = color_data[finite_mask]
        self._gradient1 = gradient1_data[finite_mask]
        self._path_metric_coords = (
            np.asarray(metric_data[finite_mask, :], dtype=float)
            if metric_data is not None
            else None
        )
        self._point_labels = np.asarray([str(value) for value in label_data[finite_mask].tolist()], dtype=object)
        self._point_ids = np.asarray([str(value) for value in point_id_data[finite_mask].tolist()], dtype=object)
        self._point_group_codes = np.asarray(group_data[finite_mask], dtype=int)
        self._title = str(title or "Gradient Scatter")
        self._x_label = str(x_label or "X axis")
        self._y_label = str(y_label or "Y axis")
        self._color_label = str(color_label or "Gradient 1")
        self._cmap_name = str(cmap_name or "spectrum_fsl")
        self._cmap = cmap if cmap is not None else GradientSurfaceDialog._default_cmap(self._cmap_name)
        self._theme_name = "Dark"
        self._hemisphere_mode = self._normalize_scatter_hemisphere_mode(hemisphere_mode)
        self._rotation_preset = self._normalize_rotation_preset(rotation_preset)
        self._use_triangular_rgb = bool(use_triangular_rgb)
        self._rgb_fit_mode = self._normalize_rgb_fit_mode(rgb_fit_mode)
        self._triangular_color_order = self._normalize_triangular_color_order(triangular_color_order)
        self._edge_pairs = self._normalize_edge_pairs(edge_pairs, self._x.size)
        self._edge_color = str(edge_color or "#111827")
        try:
            self._edge_alpha = float(np.clip(float(edge_alpha), 0.0, 1.0))
        except Exception:
            self._edge_alpha = 0.16
        try:
            self._edge_linewidth = max(0.0, float(edge_linewidth))
        except Exception:
            self._edge_linewidth = 0.45
        self._path_width_scaling_mode = "exp"
        self._path_width_scaling_strength = 2.0
        self._project_paths_callback = project_paths_callback if callable(project_paths_callback) else None
        self._export_metadata = dict(export_metadata or {}) if isinstance(export_metadata, dict) else {}
        self._default_fixed_anchor_indices = self._derive_default_fixed_anchor_indices()
        self._manual_anchor_overrides = {}
        self._endpoint_selection_mode = "adaptive"
        self._manual_endpoint_target = None
        self._project_paths_payload = None
        self._show_proximity_circles = bool(show_proximity_circles)
        self._show_adjacency_edges = True
        self._show_all_ordered_paths = False
        self._use_edge_bundling = False
        self._edge_bundling_note = ""
        self._use_directionality_filter = False
        self._use_line_proximity_energy = bool(use_line_proximity_energy)
        display_x, display_y = self._rotate_points(self._x, self._y, self._rotation_preset)
        self._display_coords = np.column_stack((display_x, display_y))
        self._path_channel_order = self._default_path_channel_order()
        self._proximity_max_radius = self._compute_max_radius(self._display_coords)
        self._proximity_slider_steps = 1000
        self._initial_proximity_slider_value = self._normalize_proximity_slider_value(
            initial_proximity_slider_value,
            self._proximity_slider_steps,
        )
        self._proximity_radius = self._slider_to_radius(self._initial_proximity_slider_value)
        self._edge_distances = self._compute_edge_distances(self._display_coords, self._edge_pairs)
        self._path_metric_edge_distances = self._compute_edge_distances(
            self._path_metric_coords if self._path_metric_coords is not None else self._display_coords,
            self._edge_pairs,
        )
        self._fixed_xlim, self._fixed_ylim = self._compute_fixed_axes(self._display_coords)
        self._point_artist = None
        self._point_artist_entries = []
        self._hover_cid = None
        self._click_cid = None
        self._free_energy_dialog = None
        self.setWindowTitle(self._title)

        self.figure = Figure(figsize=(7.4, 6.4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.info_label = QLabel(self._info_text())
        self.save_button = QPushButton("Save Figure")
        self.save_button.clicked.connect(self._save_figure)

        controls = QHBoxLayout()
        controls.addWidget(self.info_label, 1)
        controls.addWidget(self.save_button, 0)

        proximity_controls = QHBoxLayout()
        self.proximity_check = QCheckBox("Proximity circles")
        self.proximity_check.setChecked(bool(self._show_proximity_circles))
        self.proximity_check.toggled.connect(self._on_proximity_toggled)
        proximity_controls.addWidget(self.proximity_check, 0)
        slider_orientation = Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal
        self.proximity_slider = QSlider(slider_orientation)
        self.proximity_slider.setRange(0, self._proximity_slider_steps)
        self.proximity_slider.setValue(int(self._initial_proximity_slider_value))
        self.proximity_slider.valueChanged.connect(self._on_proximity_slider_changed)
        proximity_controls.addWidget(self.proximity_slider, 1)
        self.proximity_value_label = QLabel(self._proximity_label_text())
        proximity_controls.addWidget(self.proximity_value_label, 0)
        proximity_controls.addWidget(QLabel("Edge width"), 0)
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setDecimals(2)
        self.edge_width_spin.setRange(0.05, 10.0)
        self.edge_width_spin.setSingleStep(0.05)
        self.edge_width_spin.setValue(float(self._edge_linewidth))
        self.edge_width_spin.valueChanged.connect(self._on_edge_width_changed)
        proximity_controls.addWidget(self.edge_width_spin, 0)
        proximity_controls.addWidget(QLabel("Width map"), 0)
        self.path_width_mode_combo = QComboBox()
        self.path_width_mode_combo.addItem("Exp", "exp")
        self.path_width_mode_combo.addItem("Linear", "linear")
        self.path_width_mode_combo.addItem("Log", "log")
        self.path_width_mode_combo.currentIndexChanged.connect(self._on_path_width_mode_changed)
        proximity_controls.addWidget(self.path_width_mode_combo, 0)
        proximity_controls.addWidget(QLabel("Scale"), 0)
        self.path_width_scale_spin = QDoubleSpinBox()
        self.path_width_scale_spin.setDecimals(2)
        self.path_width_scale_spin.setRange(0.05, 1e12)
        self.path_width_scale_spin.setSingleStep(0.5)
        self.path_width_scale_spin.setValue(float(self._path_width_scaling_strength))
        self.path_width_scale_spin.valueChanged.connect(self._on_path_width_scale_changed)
        proximity_controls.addWidget(self.path_width_scale_spin, 0)

        path_controls = QHBoxLayout()
        self.show_adjacency_check = QCheckBox("Show adjacency")
        self.show_adjacency_check.setChecked(True)
        self.show_adjacency_check.toggled.connect(self._on_show_adjacency_toggled)
        path_controls.addWidget(self.show_adjacency_check, 0)
        self.all_paths_check = QCheckBox("All ordered paths")
        self.all_paths_check.toggled.connect(self._on_all_paths_toggled)
        path_controls.addWidget(self.all_paths_check, 0)
        self.edge_bundling_check = QCheckBox("Bundle edges")
        self.edge_bundling_check.toggled.connect(self._on_edge_bundling_toggled)
        path_controls.addWidget(self.edge_bundling_check, 0)
        self.direction_filter_check = QCheckBox("Directional filter")
        self.direction_filter_check.toggled.connect(self._on_direction_filter_toggled)
        path_controls.addWidget(self.direction_filter_check, 0)
        self.line_proximity_energy_check = QCheckBox("Line proximity")
        self.line_proximity_energy_check.setChecked(bool(self._use_line_proximity_energy))
        self.line_proximity_energy_check.toggled.connect(self._on_line_proximity_toggled)
        path_controls.addWidget(self.line_proximity_energy_check, 0)
        path_controls.addWidget(QLabel("lambda"), 0)
        self.free_energy_lambda_spin = QDoubleSpinBox()
        self.free_energy_lambda_spin.setDecimals(3)
        self.free_energy_lambda_spin.setRange(0.001, 1000.0)
        self.free_energy_lambda_spin.setSingleStep(0.1)
        self.free_energy_lambda_spin.setValue(1.0)
        self.free_energy_lambda_spin.valueChanged.connect(self._on_free_energy_lambda_changed)
        path_controls.addWidget(self.free_energy_lambda_spin, 0)
        self.generate_paths_button = QPushButton("Generate paths")
        self.generate_paths_button.clicked.connect(self._on_generate_paths_clicked)
        path_controls.addWidget(self.generate_paths_button, 0)
        self.compute_free_energy_button = QPushButton("Compute free energy")
        self.compute_free_energy_button.setEnabled(False)
        self.compute_free_energy_button.clicked.connect(self._on_compute_free_energy_clicked)
        path_controls.addWidget(self.compute_free_energy_button, 0)
        self.write_free_energy_button = QPushButton("Write free energy")
        self.write_free_energy_button.setEnabled(False)
        self.write_free_energy_button.clicked.connect(self._on_write_free_energy_clicked)
        path_controls.addWidget(self.write_free_energy_button, 0)
        self.project_paths_button = QPushButton("Project to 3D brain")
        self.project_paths_button.setEnabled(False)
        self.project_paths_button.clicked.connect(self._on_project_paths_clicked)
        path_controls.addWidget(self.project_paths_button, 0)
        self.export_paths_button = QPushButton("Export paths")
        self.export_paths_button.setEnabled(False)
        self.export_paths_button.clicked.connect(self._on_export_paths_clicked)
        path_controls.addWidget(self.export_paths_button, 0)
        path_controls.addStretch(1)

        endpoint_controls = QHBoxLayout()
        endpoint_controls.addWidget(QLabel("First"), 0)
        self.path_order_first_combo = QComboBox()
        self.path_order_first_combo.currentIndexChanged.connect(
            lambda _index: self._on_path_order_combo_changed(0)
        )
        endpoint_controls.addWidget(self.path_order_first_combo, 1)
        endpoint_controls.addWidget(QLabel("Second"), 0)
        self.path_order_second_combo = QComboBox()
        self.path_order_second_combo.currentIndexChanged.connect(
            lambda _index: self._on_path_order_combo_changed(1)
        )
        endpoint_controls.addWidget(self.path_order_second_combo, 1)
        endpoint_controls.addWidget(QLabel("Third"), 0)
        self.path_order_third_combo = QComboBox()
        self.path_order_third_combo.currentIndexChanged.connect(
            lambda _index: self._on_path_order_combo_changed(2)
        )
        endpoint_controls.addWidget(self.path_order_third_combo, 1)

        endpoint_mode_controls = QHBoxLayout()
        endpoint_mode_controls.addWidget(QLabel("Endpoints"), 0)
        self.endpoint_mode_combo = QComboBox()
        self.endpoint_mode_combo.addItem("Adaptive", "adaptive")
        self.endpoint_mode_combo.addItem("gradients_avg", "average")
        self.endpoint_mode_combo.addItem("Manual click", "manual")
        self.endpoint_mode_combo.currentIndexChanged.connect(self._on_endpoint_mode_changed)
        endpoint_mode_controls.addWidget(self.endpoint_mode_combo, 0)
        endpoint_mode_controls.addWidget(QLabel("Click target"), 0)
        self.manual_endpoint_target_combo = QComboBox()
        self.manual_endpoint_target_combo.currentIndexChanged.connect(
            self._on_manual_endpoint_target_changed
        )
        endpoint_mode_controls.addWidget(self.manual_endpoint_target_combo, 1)
        self.clear_manual_endpoints_button = QPushButton("Clear manual")
        self.clear_manual_endpoints_button.clicked.connect(self._on_clear_manual_endpoints_clicked)
        endpoint_mode_controls.addWidget(self.clear_manual_endpoints_button, 0)
        endpoint_mode_controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.toolbar)
        layout.addLayout(proximity_controls)
        layout.addLayout(path_controls)
        layout.addLayout(endpoint_controls)
        layout.addLayout(endpoint_mode_controls)
        layout.addWidget(self.canvas, 1)
        self.set_theme(theme_name)
        self._populate_path_order_combos()
        self._populate_manual_endpoint_targets()
        self._sync_proximity_controls()
        self._ensure_hover_callback()

        self._render()

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    def _info_text(self):
        mode = (
            f"{self._rgb_fit_mode.title()} {self._triangular_color_order}"
            if self._use_triangular_rgb
            else f"Cmap: {self._cmap_name}"
        )
        edge_text = ""
        if self._edge_pairs.size:
            edge_text = f" | Adjacency: {self._edge_pairs.shape[0]}"
            edge_text += f" | Visible: {self._active_edge_count()}"
            if self._show_all_ordered_paths and self._use_triangular_rgb:
                edge_text += " | All paths"
            if self._use_edge_bundling:
                edge_text += " | Bundled"
                if self._edge_bundling_note:
                    edge_text += f" | {self._edge_bundling_note}"
            if self._use_triangular_rgb and self._use_directionality_filter:
                edge_text += " | Dir filter"
        endpoint_text = f" | Path: {self._path_channel_order}" if self._use_triangular_rgb else ""
        path_text = self._path_count_summary_text()
        hemisphere_text = f" | Hemi: {self._hemisphere_mode.upper()}"
        endpoint_mode_text = (
            f" | Endpoints: {self._endpoint_mode_display_text()}"
            if self._use_triangular_rgb
            else ""
        )
        manual_target_text = ""
        if self._use_triangular_rgb and self._endpoint_selection_mode == "manual":
            current_target = self.manual_endpoint_target_combo.currentText().strip()
            if current_target:
                manual_target_text = f" | Click: {current_target}"
        return (
            f"Points: {self._x.size}{edge_text}{path_text}{endpoint_text}{endpoint_mode_text}{manual_target_text}"
            f"{hemisphere_text} | Rotation: {self._rotation_preset} | {mode}"
        )

    @staticmethod
    def _normalize_rotation_preset(value):
        text = str(value or "Default").strip()
        valid = {"Default", "+90", "-90", "180"}
        if text not in valid:
            text = "Default"
        return text

    @staticmethod
    def _normalize_scatter_hemisphere_mode(value):
        text = str(value or "both").strip().lower()
        if text not in {"both", "lh", "rh", "separate"}:
            text = "both"
        return text

    @staticmethod
    def _rotate_points(x_values, y_values, preset):
        if preset == "+90":
            return -y_values, x_values
        if preset == "-90":
            return y_values, -x_values
        if preset == "180":
            return -x_values, -y_values
        return x_values, y_values

    @staticmethod
    def _rotate_axis_labels(x_label, y_label, preset):
        if preset == "+90":
            return f"-{y_label}", x_label
        if preset == "-90":
            return y_label, f"-{x_label}"
        if preset == "180":
            return f"-{x_label}", f"-{y_label}"
        return x_label, y_label

    @staticmethod
    def _compute_display_range(values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -1.0, 1.0
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        return vmin, vmax

    @staticmethod
    def _normalize_triangular_color_order(value):
        text = str(value or "RBG").strip().upper()
        valid = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}
        if text not in valid:
            text = "RBG"
        return text

    @staticmethod
    def _normalize_rgb_fit_mode(value):
        text = str(value or "triangle").strip().lower()
        if text not in {"triangle", "square"}:
            text = "triangle"
        return text

    @staticmethod
    def _normalize_path_channel(value):
        text = str(value or "").strip().upper()
        return text if text in {"R", "G", "B"} else "R"

    @classmethod
    def _coerce_path_channel_order(cls, values, fallback="RBG"):
        requested = []
        if isinstance(values, (str, bytes)):
            requested = [char for char in str(values).strip().upper()]
        else:
            requested = [str(value or "").strip().upper() for value in list(values or [])]
        normalized = []
        for value in requested:
            if value in {"R", "G", "B"} and value not in normalized:
                normalized.append(value)
        for value in str(fallback or "RBG").strip().upper():
            if value in {"R", "G", "B"} and value not in normalized:
                normalized.append(value)
        for value in ("R", "G", "B"):
            if value not in normalized:
                normalized.append(value)
        return "".join(normalized[:3])

    def _default_path_channel_order(self):
        fallback = self._coerce_path_channel_order(self._triangular_color_order)
        if self._display_coords.ndim != 2 or self._display_coords.shape[0] < 3:
            return fallback
        try:
            rgb_model = self._rgb_model(
                self._display_coords[:, 0],
                self._display_coords[:, 1],
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )
            anchors = self._rgb_anchor_indices(
                self._display_coords[:, 0],
                self._display_coords[:, 1],
                rgb_model,
            )
        except Exception:
            return fallback
        if not {"R", "G", "B"}.issubset(set(anchors.keys())):
            return fallback
        ranked = []
        fallback_order = [str(channel) for channel in fallback]
        for channel in ("R", "G", "B"):
            anchor_index = int(anchors[channel])
            if anchor_index < 0 or anchor_index >= self._gradient1.shape[0]:
                return fallback
            try:
                fallback_rank = fallback_order.index(channel)
            except Exception:
                fallback_rank = 99
            ranked.append((float(self._gradient1[anchor_index]), -fallback_rank, channel))
        ranked.sort(reverse=True)
        order = "".join(channel for _grad1, _rank, channel in ranked)
        return self._coerce_path_channel_order(order, fallback=fallback)

    @staticmethod
    def _normalize_path_width_scaling_mode(value):
        text = str(value or "exp").strip().lower()
        if text not in {"exp", "linear", "log"}:
            text = "exp"
        return text

    @staticmethod
    def _normalize_edge_pairs(edge_pairs, n_points):
        if edge_pairs is None:
            return np.zeros((0, 2), dtype=int)
        pairs = np.asarray(edge_pairs, dtype=int)
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("Scatter edge pairs must be an Nx2 array.")
        valid = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 0] < int(n_points))
            & (pairs[:, 1] < int(n_points))
            & (pairs[:, 0] != pairs[:, 1])
        )
        pairs = pairs[valid]
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=int)
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    @staticmethod
    def _compute_max_radius(coords):
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return 0.0
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(np.square(deltas), axis=2))
        return float(np.nanmax(distances))

    @staticmethod
    def _compute_edge_distances(coords, edge_pairs):
        points = np.asarray(coords, dtype=float)
        pairs = np.asarray(edge_pairs, dtype=int)
        if points.ndim != 2 or pairs.ndim != 2 or pairs.shape[0] == 0:
            return np.zeros(0, dtype=float)
        deltas = points[pairs[:, 0], :] - points[pairs[:, 1], :]
        return np.sqrt(np.sum(np.square(deltas), axis=1))

    def _slider_to_radius(self, slider_value):
        try:
            slider = int(slider_value)
        except Exception:
            slider = 0
        slider = max(0, min(self._proximity_slider_steps, slider))
        if self._proximity_slider_steps <= 0 or self._proximity_max_radius <= 0.0:
            return 0.0
        return float(self._proximity_max_radius * (slider / float(self._proximity_slider_steps)))

    @staticmethod
    def _normalize_proximity_slider_value(value, max_steps):
        try:
            slider = int(value)
        except Exception:
            slider = 0
        return max(0, min(int(max_steps), slider))

    def _proximity_label_text(self):
        return f"r = {self._proximity_radius:.4f} / {self._proximity_max_radius:.4f}"

    @staticmethod
    def _compute_fixed_axes(coords):
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            return (-1.0, 1.0), (-1.0, 1.0)
        x_values = points[:, 0]
        y_values = points[:, 1]
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        x_span = x_max - x_min
        y_span = y_max - y_min
        if not np.isfinite(x_span) or np.isclose(x_span, 0.0):
            x_pad = max(abs(x_min), abs(x_max), 1.0) * 0.12
        else:
            x_pad = x_span * 0.08
        if not np.isfinite(y_span) or np.isclose(y_span, 0.0):
            y_pad = max(abs(y_min), abs(y_max), 1.0) * 0.12
        else:
            y_pad = y_span * 0.08
        return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

    def _sync_proximity_controls(self):
        enabled = self._display_coords.shape[0] > 0 and self._proximity_max_radius > 0.0
        self.proximity_check.setEnabled(enabled)
        self.proximity_slider.setEnabled(enabled)
        self.edge_width_spin.setEnabled(self._edge_pairs.shape[0] > 0)
        self.path_width_mode_combo.setEnabled(self._use_triangular_rgb)
        self.path_width_scale_spin.setEnabled(self._use_triangular_rgb)
        self.show_adjacency_check.setEnabled(self._edge_pairs.shape[0] > 0)
        self.all_paths_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.edge_bundling_check.setEnabled(self._edge_pairs.shape[0] > 0 or self._use_triangular_rgb)
        self.direction_filter_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.line_proximity_energy_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.generate_paths_button.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.path_order_first_combo.setEnabled(self._use_triangular_rgb)
        self.path_order_second_combo.setEnabled(self._use_triangular_rgb)
        self.path_order_third_combo.setEnabled(self._use_triangular_rgb)
        self.endpoint_mode_combo.setEnabled(self._use_triangular_rgb)
        self.manual_endpoint_target_combo.setEnabled(self._use_triangular_rgb and self._endpoint_selection_mode == "manual")
        self.clear_manual_endpoints_button.setEnabled(self._use_triangular_rgb and self._endpoint_selection_mode == "manual")
        self.free_energy_lambda_spin.setEnabled(self._use_triangular_rgb)
        projectable = False
        exportable = False
        if isinstance(self._project_paths_payload, dict):
            for group_payload in list(self._project_paths_payload.get("group_paths", [])):
                if len(list(group_payload.get("optimal_full_path", []))) >= 2:
                    if self._project_paths_callback is not None:
                        projectable = True
                if len(list(group_payload.get("subc_optimal_path", []))) >= 2:
                    if self._project_paths_callback is not None:
                        projectable = True
                if len(list(group_payload.get("all_full_paths", []))) > 0:
                    exportable = True
                if len(list(group_payload.get("subc_paths", []))) > 0:
                    exportable = True
            if (
                self._project_paths_callback is not None
                and not projectable
                and len(self._project_paths_payload.get("optimal_full_path", [])) >= 2
            ):
                projectable = True
            if not exportable and len(list(self._project_paths_payload.get("all_full_paths", []))) > 0:
                exportable = True
        self.project_paths_button.setEnabled(
            projectable
        )
        self.export_paths_button.setEnabled(exportable)
        self.compute_free_energy_button.setEnabled(exportable)
        free_energy_ready = False
        if isinstance(self._project_paths_payload, dict):
            free_energy_payload = self._project_paths_payload.get("free_energy_payload")
            free_energy_ready = isinstance(free_energy_payload, dict) and bool(list(free_energy_payload.get("groups", [])))
        self.write_free_energy_button.setEnabled(free_energy_ready)
        self.proximity_value_label.setText(self._proximity_label_text())

    def _path_count_summary_text(self, group_name=None):
        if not self._use_triangular_rgb or not isinstance(self._project_paths_payload, dict):
            return ""
        group_payloads = list(self._project_paths_payload.get("group_paths", []))
        if not group_payloads:
            return ""
        parts = []
        target_group = str(group_name or "").strip().lower()
        for group_payload in group_payloads:
            group_name = str(group_payload.get("group", "all")).strip().lower()
            if target_group and group_name != target_group:
                continue
            ctx_count = int(group_payload.get("ctx_path_count", group_payload.get("full_path_count", len(list(group_payload.get("all_full_paths", []))) or 0)))
            subc_count = int(group_payload.get("subc_path_count", len(list(group_payload.get("subc_paths", []))) or 0))
            if group_name == "lh":
                parts.append(f"LH ctx/subc: {ctx_count}/{subc_count}")
            elif group_name == "rh":
                parts.append(f"RH ctx/subc: {ctx_count}/{subc_count}")
            else:
                parts.append(f"Paths ctx/subc: {ctx_count}/{subc_count}")
        return " | " + " | ".join(parts)

    def _ensure_hover_callback(self):
        if self._hover_cid is None:
            self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        if self._click_cid is None:
            self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _hide_hover_annotation(self):
        changed = False
        for entry in list(self._point_artist_entries):
            annotation = entry.get("annotation")
            if annotation is not None and annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def _on_hover(self, event):
        if self._point_artist is None or not self._point_artist_entries:
            self._hide_hover_annotation()
            return
        if event.inaxes is None:
            self._hide_hover_annotation()
            return
        for entry in list(self._point_artist_entries):
            annotation = entry.get("annotation")
            if annotation is None:
                continue
            if event.inaxes != entry.get("axes"):
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            artist = entry.get("artist")
            contains, details = artist.contains(event)
            if not contains:
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            indices = np.asarray(details.get("ind", []), dtype=int).reshape(-1)
            if indices.size == 0:
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            local_index = int(indices[0])
            offsets = np.asarray(artist.get_offsets(), dtype=float)
            index_map = np.asarray(entry.get("indices", []), dtype=int).reshape(-1)
            if (
                local_index < 0
                or local_index >= offsets.shape[0]
                or local_index >= index_map.shape[0]
            ):
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            global_index = int(index_map[local_index])
            x_coord, y_coord = offsets[local_index]
            label = (
                str(self._point_labels[global_index])
                if 0 <= global_index < self._point_labels.shape[0]
                else f"Point {global_index + 1}"
            )
            annotation.xy = (float(x_coord), float(y_coord))
            annotation.set_text(label)
            annotation.set_visible(True)
            self.canvas.draw_idle()
            return
        self._hide_hover_annotation()

    def _on_click(self, event):
        if (
            self._endpoint_selection_mode != "manual"
            or event.inaxes is None
            or getattr(event, "button", None) not in {1, None}
        ):
            return
        target_group, target_channel = self._current_manual_endpoint_target()
        if not target_group or target_channel not in {"R", "G", "B"}:
            return
        candidate_indices = set(self._candidate_indices_for_group(target_group).tolist())
        for entry in list(self._point_artist_entries):
            if event.inaxes != entry.get("axes"):
                continue
            entry_group = str(entry.get("group", "all")).strip().lower()
            if entry_group not in {"", "all"} and target_group not in {"all", entry_group}:
                continue
            artist = entry.get("artist")
            contains, details = artist.contains(event)
            if not contains:
                continue
            indices = np.asarray(details.get("ind", []), dtype=int).reshape(-1)
            if indices.size == 0:
                continue
            local_index = int(indices[0])
            index_map = np.asarray(entry.get("indices", []), dtype=int).reshape(-1)
            if local_index < 0 or local_index >= index_map.shape[0]:
                continue
            global_index = int(index_map[local_index])
            if global_index not in candidate_indices:
                continue
            self._assign_manual_anchor(target_group, target_channel, global_index)
            self._advance_manual_endpoint_target()
            self._invalidate_generated_paths()
            self._sync_proximity_controls()
            self._render()
            return

    def _visible_edge_pairs(self):
        if self._edge_pairs.shape[0] == 0:
            return np.zeros((0, 2), dtype=int)
        if self._proximity_radius <= 0.0:
            return np.zeros((0, 2), dtype=int)
        visible = self._edge_distances <= (2.0 * self._proximity_radius + 1e-12)
        if not np.any(visible):
            return np.zeros((0, 2), dtype=int)
        return np.asarray(self._edge_pairs[visible], dtype=int)

    def _visible_edge_distances(self):
        if self._edge_pairs.shape[0] == 0:
            return np.zeros(0, dtype=float)
        if self._proximity_radius <= 0.0:
            return np.zeros(0, dtype=float)
        visible = self._edge_distances <= (2.0 * self._proximity_radius + 1e-12)
        if not np.any(visible):
            return np.zeros(0, dtype=float)
        return np.asarray(self._path_metric_edge_distances[visible], dtype=float)

    def _active_edge_count(self):
        return int(self._visible_edge_pairs().shape[0])

    @staticmethod
    def _rgb_anchor_indices(x_plot, y_plot, rgb_model, candidate_indices=None):
        points = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        if points.shape[0] == 0:
            return {}
        if candidate_indices is None:
            candidate_indices = np.arange(points.shape[0], dtype=int)
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        if candidate_indices.size < 3:
            return {}
        anchor_points = np.asarray(rgb_model.get("anchor_points"), dtype=float)
        order = [str(channel) for channel in rgb_model.get("order", "RBG")]
        anchors = {}
        used = set()
        for anchor_point, channel in zip(anchor_points, order):
            distances = np.sqrt(np.sum(np.square(points[candidate_indices, :] - anchor_point[np.newaxis, :]), axis=1))
            ranking = np.argsort(distances)
            chosen = int(candidate_indices[int(ranking[0])])
            for rank_idx in ranking.tolist():
                idx = int(candidate_indices[int(rank_idx)])
                if idx not in used:
                    chosen = int(idx)
                    break
            used.add(chosen)
            anchors[channel] = chosen
        return anchors

    def _resolved_rgb_anchor_indices(self, x_plot, y_plot, rgb_model, candidate_indices=None, group_name=None):
        candidate_indices = (
            np.arange(np.asarray(x_plot, dtype=float).shape[0], dtype=int)
            if candidate_indices is None
            else np.asarray(candidate_indices, dtype=int).reshape(-1)
        )
        resolved = dict(self._effective_anchor_indices_for_group(group_name))
        if {"R", "G", "B"}.issubset(set(resolved.keys())):
            candidate_set = {int(value) for value in candidate_indices.tolist()}
            anchor_values = [int(resolved[channel]) for channel in ("R", "G", "B")]
            if (
                all(int(value) in candidate_set for value in anchor_values)
                and len(set(anchor_values)) == 3
            ):
                return {channel: int(resolved[channel]) for channel in ("R", "G", "B")}
        return self._rgb_anchor_indices(
            x_plot,
            y_plot,
            rgb_model,
            candidate_indices=candidate_indices,
        )

    @staticmethod
    def _triangular_anchor_indices(x_plot, y_plot, triangle_model, candidate_indices=None):
        return GradientScatterDialog._rgb_anchor_indices(
            x_plot,
            y_plot,
            triangle_model,
            candidate_indices=candidate_indices,
        )

    def _path_group_specs(self):
        codes = np.asarray(self._point_group_codes, dtype=int).reshape(-1)
        if codes.shape[0] != self._x.shape[0]:
            return [{"name": "all", "eligible_mask": np.ones(self._x.shape, dtype=bool)}]
        has_lh = bool(np.any(codes == 0))
        has_rh = bool(np.any(codes == 1))
        if self._hemisphere_mode == "lh":
            return [{"name": "lh", "eligible_mask": np.asarray((codes == 0) | (codes == 2), dtype=bool)}]
        if self._hemisphere_mode == "rh":
            return [{"name": "rh", "eligible_mask": np.asarray((codes == 1) | (codes == 2), dtype=bool)}]
        if has_lh and has_rh:
            return [
                {"name": "lh", "eligible_mask": np.asarray((codes == 0) | (codes == 2), dtype=bool)},
                {"name": "rh", "eligible_mask": np.asarray((codes == 1) | (codes == 2), dtype=bool)},
            ]
        if has_lh:
            return [{"name": "lh", "eligible_mask": np.asarray((codes == 0) | (codes == 2), dtype=bool)}]
        if has_rh:
            return [{"name": "rh", "eligible_mask": np.asarray((codes == 1) | (codes == 2), dtype=bool)}]
        return [{"name": "all", "eligible_mask": np.ones(codes.shape, dtype=bool)}]

    def _subc_target_names(self, group_name):
        text = str(group_name or "all").strip().lower()
        if text == "lh":
            return ("thal-lh-ventrolateral",)
        if text == "rh":
            return ("thal-rh-ventrolateral",)
        return ("thal-lh-ventrolateral", "thal-rh-ventrolateral")

    @staticmethod
    def _normalize_label_token(text):
        normalized = str(text or "").strip().lower()
        for old, new in (
            ("_", "-"),
            (" ", "-"),
            (".", "-"),
            ("/", "-"),
        ):
            normalized = normalized.replace(old, new)
        while "--" in normalized:
            normalized = normalized.replace("--", "-")
        return normalized

    def _populate_path_order_combos(self):
        combo_specs = (
            (self.path_order_first_combo, self._path_channel_order[0] if len(self._path_channel_order) >= 1 else "R"),
            (self.path_order_second_combo, self._path_channel_order[1] if len(self._path_channel_order) >= 2 else "G"),
            (self.path_order_third_combo, self._path_channel_order[2] if len(self._path_channel_order) >= 3 else "B"),
        )
        for combo, current_value in combo_specs:
            combo.blockSignals(True)
            combo.clear()
            for label in ("R", "G", "B"):
                combo.addItem(label, label)
            index = combo.findData(current_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _group_display_name(self, group_name):
        name = str(group_name or "all").strip().lower()
        if name == "lh":
            return "LH"
        if name == "rh":
            return "RH"
        return "All"

    def _display_group_specs(self):
        path_groups = list(self._path_group_specs())
        if self._hemisphere_mode == "separate" and len(path_groups) > 1:
            specs = []
            for group_spec in path_groups:
                name = str(group_spec.get("name", "all")).strip().lower()
                eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
                if eligible_mask.shape[0] != self._x.shape[0] or not np.any(eligible_mask):
                    continue
                specs.append(
                    {
                        "name": name,
                        "title": self._group_display_name(name),
                        "indices": np.flatnonzero(eligible_mask),
                    }
                )
            return specs or [{"name": "all", "title": self._group_display_name("all"), "indices": np.arange(self._x.shape[0], dtype=int)}]
        return [{"name": "all", "title": "", "indices": np.arange(self._x.shape[0], dtype=int)}]

    def _candidate_indices_for_group(self, group_name):
        target = str(group_name or "all").strip().lower()
        for group_spec in self._path_group_specs():
            if str(group_spec.get("name", "all")).strip().lower() != target:
                continue
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                return np.arange(self._x.shape[0], dtype=int)
            return np.flatnonzero(eligible_mask)
        return np.arange(self._x.shape[0], dtype=int)

    def _anchor_option_label(self, node_index):
        idx = int(node_index)
        if idx < 0 or idx >= self._point_labels.shape[0]:
            return "Unknown"
        return f"{str(self._point_labels[idx])} [{str(self._point_ids[idx])}]"

    def _average_gradient_pair_for_anchor_defaults(self):
        gradients_avg = np.asarray(
            dict(self._export_metadata or {}).get("gradients_avg", np.empty((0, 0), dtype=float)),
            dtype=float,
        )
        if gradients_avg.ndim != 2:
            return None
        n_points = int(self._point_ids.shape[0])
        if gradients_avg.shape[1] == n_points:
            canonical = np.asarray(gradients_avg, dtype=float)
        elif gradients_avg.shape[0] == n_points:
            canonical = np.asarray(gradients_avg.T, dtype=float)
        else:
            return None
        if canonical.shape[0] < 2:
            return None
        return np.asarray(canonical[:2, :].T, dtype=float)

    def _derive_default_fixed_anchor_indices(self):
        gradient_pair = self._average_gradient_pair_for_anchor_defaults()
        if gradient_pair is None or gradient_pair.ndim != 2 or gradient_pair.shape[1] < 2:
            return {}
        gradient1 = np.asarray(gradient_pair[:, 0], dtype=float)
        gradient2 = np.asarray(gradient_pair[:, 1], dtype=float)
        scatter_coords = np.column_stack((gradient2, gradient1))
        finite_mask = np.all(np.isfinite(scatter_coords), axis=1)
        if not np.any(finite_mask):
            return {}
        fixed = {}
        for group_spec in self._path_group_specs():
            group_name = str(group_spec.get("name", "all")).strip().lower()
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                continue
            candidate_indices = np.flatnonzero(eligible_mask & finite_mask)
            if candidate_indices.size < 3:
                continue
            triangle_model = self._rgb_model(
                scatter_coords[candidate_indices, 0],
                scatter_coords[candidate_indices, 1],
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )
            anchors = self._rgb_anchor_indices(
                scatter_coords[:, 0],
                scatter_coords[:, 1],
                triangle_model,
                candidate_indices=candidate_indices,
            )
            if {"R", "G", "B"}.issubset(set(anchors.keys())):
                fixed[group_name] = {str(channel): int(index) for channel, index in anchors.items()}
        return fixed

    def _auto_anchor_indices_for_group(self, group_name):
        candidate_indices = self._candidate_indices_for_group(group_name)
        if candidate_indices.size < 3 or self._display_coords.shape[0] < 3:
            return {}
        coords = np.asarray(self._display_coords[candidate_indices, :], dtype=float)
        finite_mask = np.all(np.isfinite(coords), axis=1)
        if np.count_nonzero(finite_mask) < 3:
            return {}
        triangle_model = self._rgb_model(
            coords[finite_mask, 0],
            coords[finite_mask, 1],
            self._triangular_color_order,
            fit_mode=self._rgb_fit_mode,
        )
        anchors = self._rgb_anchor_indices(
            self._display_coords[:, 0],
            self._display_coords[:, 1],
            triangle_model,
            candidate_indices=candidate_indices,
        )
        return {str(channel): int(index) for channel, index in anchors.items()}

    def _effective_anchor_indices_for_group(self, group_name):
        name = str(group_name or "all").strip().lower()
        auto_base = dict(self._auto_anchor_indices_for_group(name))
        avg_base = dict(self._default_fixed_anchor_indices.get(name, {}))
        mode = str(self._endpoint_selection_mode or "adaptive").strip().lower()
        if mode == "average":
            base = dict(avg_base if {"R", "G", "B"}.issubset(set(avg_base.keys())) else auto_base)
        elif mode == "manual":
            base = dict(auto_base if {"R", "G", "B"}.issubset(set(auto_base.keys())) else avg_base)
        else:
            base = dict(auto_base)
        overrides = dict(self._manual_anchor_overrides.get(name, {}))
        candidate_indices = np.asarray(self._candidate_indices_for_group(name), dtype=int).reshape(-1)
        candidate_set = {int(value) for value in candidate_indices.tolist()}

        for channel, value in list(overrides.items()):
            if value is None:
                continue
            try:
                node_index = int(value)
            except Exception:
                continue
            if candidate_set and node_index not in candidate_set:
                continue
            base[str(channel)] = node_index

        manual_channels = {
            str(channel).strip().upper()
            for channel, value in list(overrides.items())
            if value is not None
        }
        resolved = {}
        used = set()

        for channel in ("R", "G", "B"):
            if channel not in manual_channels:
                continue
            try:
                idx = int(base.get(channel))
            except Exception:
                continue
            if candidate_set and idx not in candidate_set:
                continue
            if idx in used:
                continue
            resolved[channel] = idx
            used.add(idx)

        for channel in ("R", "G", "B"):
            if channel in resolved:
                continue
            candidate_pool = []
            for source in (base, auto_base, avg_base):
                value = source.get(channel)
                try:
                    idx = int(value)
                except Exception:
                    continue
                candidate_pool.append(idx)
            candidate_pool.extend(candidate_indices.tolist())
            chosen = None
            for idx in candidate_pool:
                idx = int(idx)
                if candidate_set and idx not in candidate_set:
                    continue
                if idx in used:
                    continue
                chosen = idx
                break
            if chosen is not None:
                resolved[channel] = int(chosen)
                used.add(int(chosen))

        return {str(channel): int(index) for channel, index in resolved.items()}

    def _average_endpoints_available(self):
        for anchors in list(dict(self._default_fixed_anchor_indices or {}).values()):
            if {"R", "G", "B"}.issubset(set(dict(anchors or {}).keys())):
                return True
        return False

    def _endpoint_mode_display_text(self):
        mode = str(self._endpoint_selection_mode or "adaptive").strip().lower()
        if mode == "manual":
            return "manual"
        if mode == "average":
            return "gradients_avg" if self._average_endpoints_available() else "adaptive (no avg)"
        return "adaptive"

    def _manual_endpoint_target_specs(self):
        specs = []
        for group_spec in self._path_group_specs():
            group_name = str(group_spec.get("name", "all")).strip().lower()
            for channel in ("R", "G", "B"):
                if group_name == "all":
                    label = channel
                else:
                    label = f"{self._group_display_name(group_name)} {channel}"
                specs.append((label, f"{group_name}:{channel}"))
        return specs

    def _populate_manual_endpoint_targets(self):
        current = self._manual_endpoint_target
        self.manual_endpoint_target_combo.blockSignals(True)
        self.manual_endpoint_target_combo.clear()
        for label, value in self._manual_endpoint_target_specs():
            self.manual_endpoint_target_combo.addItem(label, value)
        selected_index = -1
        if current is not None:
            selected_index = self.manual_endpoint_target_combo.findData(current)
        if selected_index < 0 and self.manual_endpoint_target_combo.count() > 0:
            selected_index = 0
        if selected_index >= 0:
            self.manual_endpoint_target_combo.setCurrentIndex(selected_index)
            self._manual_endpoint_target = self.manual_endpoint_target_combo.currentData()
        else:
            self._manual_endpoint_target = None
        self.manual_endpoint_target_combo.blockSignals(False)

    def _current_manual_endpoint_target(self):
        value = self.manual_endpoint_target_combo.currentData()
        if value is None:
            return None, None
        text = str(value).strip()
        if ":" not in text:
            return None, None
        group_name, channel = text.split(":", 1)
        channel = self._normalize_path_channel(channel)
        return str(group_name or "all").strip().lower(), channel

    def _assign_manual_anchor(self, group_name, channel, node_index):
        group_key = str(group_name or "all").strip().lower()
        channel_key = self._normalize_path_channel(channel)
        try:
            node_value = int(node_index)
        except Exception:
            return
        overrides = dict(self._manual_anchor_overrides.get(group_key, {}))
        overrides[channel_key] = node_value
        for other_channel, other_value in list(overrides.items()):
            if other_channel != channel_key and int(other_value) == node_value:
                overrides.pop(other_channel, None)
        self._manual_anchor_overrides[group_key] = overrides

    def _advance_manual_endpoint_target(self):
        count = int(self.manual_endpoint_target_combo.count())
        if count <= 1:
            return
        current_index = int(self.manual_endpoint_target_combo.currentIndex())
        self.manual_endpoint_target_combo.setCurrentIndex((current_index + 1) % count)

    def _on_endpoint_mode_changed(self, _index):
        value = self.endpoint_mode_combo.currentData()
        mode = str(value or "adaptive").strip().lower()
        if mode not in {"adaptive", "average", "manual"}:
            mode = "adaptive"
        self._endpoint_selection_mode = mode
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_manual_endpoint_target_changed(self, _index):
        self._manual_endpoint_target = self.manual_endpoint_target_combo.currentData()
        self._render()

    def _on_clear_manual_endpoints_clicked(self):
        self._manual_anchor_overrides = {}
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _find_subc_anchor_index(self, candidate_indices, group_name):
        indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            return None
        targets = tuple(self._normalize_label_token(target) for target in self._subc_target_names(group_name))
        for idx in indices.tolist():
            if idx < 0 or idx >= self._point_labels.shape[0]:
                continue
            label_text = self._normalize_label_token(self._point_labels[idx])
            if any(target in label_text for target in targets):
                return int(idx)
        return None

    @staticmethod
    def _weighted_adjacency(node_count, edge_pairs, edge_weights, forbidden_nodes=None):
        adjacency = [[] for _ in range(int(node_count))]
        blocked = {int(node) for node in (forbidden_nodes or set())}
        for (node_u, node_v), weight in zip(np.asarray(edge_pairs, dtype=int), np.asarray(edge_weights, dtype=float)):
            w = float(weight)
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            u = int(node_u)
            v = int(node_v)
            if u in blocked or v in blocked:
                continue
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))
        for neighbors in adjacency:
            neighbors.sort(key=lambda item: item[1])
        return adjacency

    @staticmethod
    def _shortest_path(node_count, edge_pairs, edge_weights, start_index, end_index, forbidden_nodes=None):
        if node_count <= 0:
            return None
        start = int(start_index)
        end = int(end_index)
        if start < 0 or end < 0 or start >= node_count or end >= node_count:
            return None
        if start == end:
            return [start]

        blocked = {int(node) for node in (forbidden_nodes or set())}
        blocked.discard(start)
        blocked.discard(end)
        adjacency = GradientScatterDialog._weighted_adjacency(
            node_count,
            edge_pairs,
            edge_weights,
            forbidden_nodes=blocked,
        )

        distances = np.full(int(node_count), np.inf, dtype=float)
        previous = np.full(int(node_count), -1, dtype=int)
        distances[start] = 0.0
        heap = [(0.0, start)]

        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist > distances[node]:
                continue
            if node == end:
                break
            for neighbor, weight in adjacency[node]:
                candidate = current_dist + float(weight)
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(heap, (candidate, neighbor))

        if not np.isfinite(distances[end]):
            return None

        path = [end]
        current = end
        while current != start:
            current = int(previous[current])
            if current < 0:
                return None
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _rgb_basis_color(channel):
        mapping = {
            "R": np.asarray((1.0, 0.0, 0.0), dtype=float),
            "G": np.asarray((0.0, 1.0, 0.0), dtype=float),
            "B": np.asarray((0.0, 0.0, 1.0), dtype=float),
        }
        return np.asarray(mapping.get(str(channel).strip().upper(), (0.5, 0.5, 0.5)), dtype=float)

    @staticmethod
    def _pair_channel_color(first, second):
        pair = frozenset((str(first).strip().upper(), str(second).strip().upper()))
        mapping = {
            frozenset(("R", "B")): np.asarray((0.58, 0.28, 0.82), dtype=float),  # violet
            frozenset(("R", "G")): np.asarray((1.00, 0.55, 0.00), dtype=float),  # orange
            frozenset(("G", "B")): np.asarray((0.10, 0.76, 0.72), dtype=float),  # turquoise
        }
        if pair in mapping:
            return np.asarray(mapping[pair], dtype=float)
        return np.clip(
            0.5 * (
                GradientScatterDialog._rgb_basis_color(first)
                + GradientScatterDialog._rgb_basis_color(second)
            ),
            0.0,
            1.0,
        )

    def _ctx_segment_records_for_full_path(self, path_nodes, anchors, channel_order):
        nodes = [int(node) for node in list(path_nodes or [])]
        anchor_map = {str(key): int(value) for key, value in dict(anchors or {}).items()}
        order = [str(channel) for channel in list(channel_order or []) if str(channel) in anchor_map]
        if len(nodes) < 2 or len(order) < 2:
            return []

        def _segment_record(first, second, segment_nodes):
            color = self._pair_channel_color(first, second)
            return {
                "first": str(first),
                "second": str(second),
                "nodes": [int(node) for node in list(segment_nodes or [])],
                "color": [float(value) for value in color.tolist()],
            }

        if len(order) == 2:
            return [_segment_record(order[0], order[1], nodes)]

        middle_anchor = int(anchor_map[order[1]])
        try:
            split_index = next(
                idx
                for idx, node in enumerate(nodes)
                if int(node) == middle_anchor and 0 < idx < len(nodes) - 1
            )
        except StopIteration:
            return [_segment_record(order[0], order[-1], nodes)]

        return [
            _segment_record(order[0], order[1], nodes[: split_index + 1]),
            _segment_record(order[1], order[2], nodes[split_index:]),
        ]

    @staticmethod
    def _energy_scaling_range(payload, family_type):
        if not isinstance(payload, dict):
            return None
        scaling = dict(payload.get("energy_width_scaling", {})).get(str(family_type), None)
        if not isinstance(scaling, dict):
            return None
        try:
            emin = float(scaling.get("min"))
            emax = float(scaling.get("max"))
        except Exception:
            return None
        if not np.isfinite(emin) or not np.isfinite(emax):
            return None
        return {"min": emin, "max": emax}

    @staticmethod
    def _path_display_width(
        base_width,
        energy=None,
        scaling=None,
        *,
        mode="scatter",
        scaling_mode="exp",
        scaling_strength=2.0,
    ):
        try:
            base = max(0.05, float(base_width))
        except Exception:
            base = 0.45
        if mode == "brain":
            default_width = max(1.2, base * 6.0)
            min_scale = 4.5
            max_scale = 9.0
        else:
            default_width = max(0.7, base * 2.0)
            min_scale = 1.2
            max_scale = 4.0
        if energy is None or scaling is None:
            return default_width
        try:
            energy_value = float(energy)
            emin = float(scaling.get("min"))
            emax = float(scaling.get("max"))
        except Exception:
            return default_width
        if not np.isfinite(energy_value) or not np.isfinite(emin) or not np.isfinite(emax):
            return default_width
        if np.isclose(emax, emin):
            norm = 0.5
        else:
            norm = float(np.clip((energy_value - emin) / (emax - emin), 0.0, 1.0))
        mode_name = GradientScatterDialog._normalize_path_width_scaling_mode(scaling_mode)
        try:
            scale_value = max(0.05, float(scaling_strength))
        except Exception:
            scale_value = 2.0
        if mode_name == "linear":
            mapped = norm
        elif mode_name == "log":
            mapped = float(np.log1p(scale_value * norm) / np.log1p(scale_value))
        else:
            denominator = float(np.expm1(scale_value))
            if np.isclose(denominator, 0.0):
                mapped = norm
            else:
                mapped = float(np.expm1(scale_value * norm) / denominator)
        return max(default_width * 0.7, base * (min_scale + (max_scale - min_scale) * mapped))

    @staticmethod
    def _enumerate_simple_paths(
        node_count,
        edge_pairs,
        edge_weights,
        start_index,
        end_index,
        *,
        forbidden_nodes=None,
        max_paths=96,
        max_depth=24,
    ):
        if node_count <= 0:
            return []
        start = int(start_index)
        end = int(end_index)
        if start < 0 or end < 0 or start >= node_count or end >= node_count:
            return []
        if start == end:
            return [[start]]

        blocked = {int(node) for node in (forbidden_nodes or set())}
        blocked.discard(start)
        blocked.discard(end)
        adjacency = GradientScatterDialog._weighted_adjacency(
            node_count,
            edge_pairs,
            edge_weights,
            forbidden_nodes=blocked,
        )
        max_paths = max(1, int(max_paths))
        max_depth = max(2, int(max_depth))
        results = []

        def dfs(node, path, visited):
            if len(results) >= max_paths:
                return
            if len(path) > max_depth:
                return
            if node == end:
                results.append(list(path))
                return
            for neighbor, _weight in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(int(neighbor))
                dfs(int(neighbor), path, visited)
                path.pop()
                visited.remove(neighbor)
                if len(results) >= max_paths:
                    return

        dfs(start, [start], {start})
        return results

    @staticmethod
    def _ordered_anchor_paths(node_count, edge_pairs, edge_weights, channel_order, anchors, forbidden_nodes=None):
        order = [str(channel) for channel in channel_order if str(channel) in anchors]
        if len(order) < 3:
            return []

        visited_nodes = set(int(node) for node in (forbidden_nodes or set()))
        segments = []
        for idx in range(len(order) - 1):
            first = order[idx]
            second = order[idx + 1]
            start = int(anchors[first])
            end = int(anchors[second])
            forbidden = set(visited_nodes)
            forbidden.discard(start)
            forbidden.discard(end)
            for future_channel in order[idx + 2 :]:
                future_anchor = anchors.get(future_channel)
                if future_anchor is not None and int(future_anchor) not in {start, end}:
                    forbidden.add(int(future_anchor))
            path = GradientScatterDialog._shortest_path(
                node_count,
                edge_pairs,
                edge_weights,
                start,
                end,
                forbidden_nodes=forbidden,
            )
            if path is None or len(path) < 2:
                return []
            segments.append((first, second, path))
            visited_nodes.update(int(node) for node in path)
        return segments

    @staticmethod
    def _ordered_anchor_pair_paths(node_count, edge_pairs, edge_weights, channel_order, anchors, forbidden_nodes=None):
        order = [str(channel) for channel in channel_order if str(channel) in anchors]
        if len(order) < 2:
            return []

        pair_paths = []
        for idx in range(len(order) - 1):
            first = order[idx]
            second = order[idx + 1]
            start = int(anchors[first])
            end = int(anchors[second])
            forbidden = {int(node) for node in (forbidden_nodes or set())}
            for channel in order:
                anchor = anchors.get(channel)
                if anchor is None:
                    continue
                anchor = int(anchor)
                if channel not in {first, second}:
                    forbidden.add(anchor)
            shortest = GradientScatterDialog._shortest_path(
                node_count,
                edge_pairs,
                edge_weights,
                start,
                end,
                forbidden_nodes=forbidden,
            )
            if shortest is None or len(shortest) < 2:
                pair_paths.append((first, second, []))
                continue
            max_depth = min(max(2, len(shortest) + 4), max(2, int(node_count)))
            all_paths = GradientScatterDialog._enumerate_simple_paths(
                node_count,
                edge_pairs,
                edge_weights,
                start,
                end,
                forbidden_nodes=forbidden,
                max_paths=96,
                max_depth=min(max_depth, 24),
            )
            pair_paths.append((first, second, all_paths))
        return pair_paths

    @staticmethod
    def _path_segments(x_plot, y_plot, node_path):
        if node_path is None or len(node_path) < 2:
            return np.zeros((0, 2, 2), dtype=float)
        coords = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        segments = []
        for idx in range(len(node_path) - 1):
            a = int(node_path[idx])
            b = int(node_path[idx + 1])
            segments.append(np.asarray((coords[a], coords[b]), dtype=float))
        if not segments:
            return np.zeros((0, 2, 2), dtype=float)
        return np.asarray(segments, dtype=float)

    @staticmethod
    def _path_metric_length(coords, node_path):
        nodes = np.asarray([int(node) for node in list(node_path or [])], dtype=int)
        points = np.asarray(coords, dtype=float)
        if nodes.size < 2 or points.ndim != 2 or points.shape[1] < 1:
            return None
        if np.any((nodes < 0) | (nodes >= points.shape[0])):
            return None
        path_points = np.asarray(points[nodes, :], dtype=float)
        if not np.all(np.isfinite(path_points)):
            return None
        steps = np.diff(path_points, axis=0)
        if steps.shape[0] == 0:
            return None
        lengths = np.linalg.norm(steps, axis=1)
        if not np.all(np.isfinite(lengths)):
            return None
        return float(np.sum(lengths))

    @staticmethod
    def _path_edge_pairs(node_path):
        nodes = np.asarray([int(node) for node in list(node_path or [])], dtype=int)
        if nodes.size < 2:
            return np.zeros((0, 2), dtype=int)
        return np.column_stack((nodes[:-1], nodes[1:])).astype(int, copy=False)

    @staticmethod
    def _load_edge_bundling_utils():
        try:
            from mrsitoolbox.graphplot.edge_bundling import bundle_edges_2d, polylines_to_segments
            return bundle_edges_2d, polylines_to_segments
        except Exception:
            pass
        try:
            from graphplot.edge_bundling import bundle_edges_2d, polylines_to_segments
            return bundle_edges_2d, polylines_to_segments
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "mrsitoolbox" / "graphplot" / "edge_bundling.py",
            repo_root / "mrsitoolbox_demo" / "mrsitoolbox" / "graphplot" / "edge_bundling.py",
        ]
        for module_path in candidates:
            if not module_path.exists():
                continue
            module_name = f"_donald_edge_bundling_{module_path.stem}_{abs(hash(str(module_path)))}"
            try:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                return module.bundle_edges_2d, module.polylines_to_segments
            except Exception:
                continue

        raise ImportError("Could not locate edge_bundling.py in the installed package or local workspace.")

    def _bundled_segments_from_pairs(self, edge_pairs):
        pairs = np.asarray(edge_pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[0] == 0:
            return np.zeros((0, 2, 2), dtype=float)
        try:
            bundle_edges_2d, polylines_to_segments = self._load_edge_bundling_utils()
            bundled = bundle_edges_2d(self._display_coords, pairs, method="hammer")
            segments = polylines_to_segments(bundled.polylines)
            if segments.size:
                self._edge_bundling_note = ""
                return np.asarray(segments, dtype=float)
            self._edge_bundling_note = "straight fallback"
        except Exception as exc:
            self._edge_bundling_note = "bundle unavailable"
            warn(f"Edge bundling fallback to straight segments: {exc}")
        points = np.asarray(self._display_coords, dtype=float)
        return np.stack((points[pairs[:, 0], :], points[pairs[:, 1], :]), axis=1)

    @staticmethod
    def _path_moves_towards_endpoint(
        coords,
        node_path,
        endpoint_index,
        tol=1e-9,
        max_direction_violations=2,
    ):
        path = [int(node) for node in list(node_path or [])]
        if len(path) < 2:
            return False
        points = np.asarray(coords, dtype=float)
        end = int(endpoint_index)
        if end < 0 or end >= points.shape[0]:
            return False
        endpoint = np.asarray(points[end], dtype=float)
        violations = 0
        for idx in range(len(path) - 1):
            current = int(path[idx])
            nxt = int(path[idx + 1])
            if current < 0 or nxt < 0 or current >= points.shape[0] or nxt >= points.shape[0]:
                return False
            current_point = np.asarray(points[current], dtype=float)
            next_point = np.asarray(points[nxt], dtype=float)
            to_end = endpoint - current_point
            step = next_point - current_point
            segment_is_forward = float(np.dot(step, to_end)) > float(tol)
            current_dist = float(np.linalg.norm(to_end))
            next_dist = float(np.linalg.norm(endpoint - next_point))
            if next_dist >= current_dist - float(tol):
                segment_is_forward = False
            if not segment_is_forward:
                violations += 1
                if violations > int(max(0, max_direction_violations)):
                    return False
        return True

    def _build_triangular_anchor_paths_payload(
        self,
        x_plot,
        y_plot,
        triangle_model,
        path_channel_order,
        point_colors,
        visible_edge_pairs,
        visible_edge_distances,
    ):
        if visible_edge_pairs.shape[0] == 0:
            return None
        scatter_coords = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        metric_coords = (
            np.asarray(self._path_metric_coords, dtype=float)
            if self._path_metric_coords is not None
            else np.asarray(scatter_coords, dtype=float)
        )
        project_group_paths = []
        valid_optimal_paths = []

        for group_spec in self._path_group_specs():
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                continue
            candidate_indices = np.flatnonzero(eligible_mask)
            anchors = self._resolved_rgb_anchor_indices(
                x_plot,
                y_plot,
                triangle_model,
                candidate_indices=candidate_indices,
                group_name=group_spec.get("name", "all"),
            )
            if not {"R", "G", "B"}.issubset(set(anchors.keys())):
                continue
            forbidden_nodes = set(np.flatnonzero(~eligible_mask).tolist())
            group_pair_paths = []
            path_order = [str(channel) for channel in list(str(path_channel_order or self._triangular_color_order))]
            pair_paths = self._ordered_anchor_pair_paths(
                self._x.size,
                visible_edge_pairs,
                visible_edge_distances,
                path_order,
                anchors,
                forbidden_nodes=forbidden_nodes,
            )
            valid_pair_records = []
            for first, second, paths in pair_paths:
                color = tuple(self._pair_channel_color(first, second).tolist())
                valid_paths = []
                for path in paths:
                    if self._use_directionality_filter and not self._path_moves_towards_endpoint(
                        scatter_coords,
                        path,
                        anchors[second],
                    ):
                        continue
                    record = self._path_record(first, second, path, color)
                    record["group"] = str(group_spec.get("name", "all"))
                    group_pair_paths.append(record)
                    valid_paths.append([int(node) for node in list(path)])
                valid_pair_records.append((first, second, valid_paths))

            order_channels = [str(channel) for channel in path_order]
            subc_paths = []
            subc_optimal_path = []
            subc_color = None
            subc_anchor_index = self._find_subc_anchor_index(
                candidate_indices,
                group_spec.get("name", "all"),
            )
            if (
                subc_anchor_index is not None
                and len(order_channels) >= 2
                and order_channels[1] in anchors
            ):
                start_channel = str(order_channels[1])
                start_index = int(anchors[start_channel])
                start_color = (
                    np.asarray(point_colors[start_index], dtype=float)
                    if 0 <= start_index < np.asarray(point_colors, dtype=float).shape[0]
                    else np.asarray((0.0, 0.0, 0.0), dtype=float)
                )
                target_color = (
                    np.asarray(point_colors[subc_anchor_index], dtype=float)
                    if 0 <= subc_anchor_index < np.asarray(point_colors, dtype=float).shape[0]
                    else np.asarray((0.0, 0.0, 0.0), dtype=float)
                )
                subc_color = np.clip(0.5 * (start_color + target_color), 0.0, 1.0)
                subc_forbidden_nodes = set(forbidden_nodes)
                first_ctx_anchor = anchors.get(order_channels[0])
                if first_ctx_anchor is not None and int(first_ctx_anchor) not in {start_index, int(subc_anchor_index)}:
                    subc_forbidden_nodes.add(int(first_ctx_anchor))
                shortest_subc = self._shortest_path(
                    self._x.size,
                    visible_edge_pairs,
                    visible_edge_distances,
                    start_index,
                    int(subc_anchor_index),
                    forbidden_nodes=subc_forbidden_nodes,
                )
                if shortest_subc is not None and len(shortest_subc) >= 2:
                    max_depth = min(max(2, len(shortest_subc) + 4), max(2, int(self._x.size)))
                    all_subc_paths = self._enumerate_simple_paths(
                        self._x.size,
                        visible_edge_pairs,
                        visible_edge_distances,
                        start_index,
                        int(subc_anchor_index),
                        forbidden_nodes=subc_forbidden_nodes,
                        max_paths=96,
                        max_depth=min(max_depth, 24),
                    )
                    for path in all_subc_paths:
                        if self._use_directionality_filter and not self._path_moves_towards_endpoint(
                            scatter_coords,
                            path,
                            subc_anchor_index,
                        ):
                            continue
                        subc_paths.append([int(node) for node in list(path)])
                    if not subc_paths:
                        subc_paths = [[int(node) for node in list(shortest_subc)]]
                    if subc_paths:
                        subc_optimal_path = list(subc_paths[0])

            ordered_segments = self._ordered_anchor_paths(
                self._x.size,
                visible_edge_pairs,
                visible_edge_distances,
                path_order,
                anchors,
                forbidden_nodes=forbidden_nodes,
            )
            valid_ordered_segments = []
            for first, second, path in ordered_segments:
                if self._use_directionality_filter and not self._path_moves_towards_endpoint(
                    scatter_coords,
                    path,
                    anchors[second],
                ):
                    continue
                segments = self._path_segments(x_plot, y_plot, path)
                if segments.shape[0] == 0:
                    continue
                valid_ordered_segments.append((first, second, [int(node) for node in list(path)]))

            all_full_paths = self._combine_ordered_path_records(valid_pair_records, max_full_paths=256)
            optimal_full_path = []
            optimal_full_energy = None
            if all_full_paths:
                ctx_length_candidates = []
                for path_nodes in list(all_full_paths):
                    path_length = self._path_metric_length(metric_coords, path_nodes)
                    if path_length is None or not np.isfinite(path_length):
                        continue
                    ctx_length_candidates.append(
                        (
                            float(path_length),
                            len(list(path_nodes or [])),
                            [int(node) for node in list(path_nodes or [])],
                        )
                    )
                if ctx_length_candidates:
                    ctx_length_candidates.sort(key=lambda item: (item[0], item[1]))
                    optimal_full_path = [int(node) for node in list(ctx_length_candidates[0][2])]
                    highlight_energy = self._ctx_full_path_energy(
                        scatter_coords,
                        optimal_full_path,
                        anchors,
                        path_order,
                    )
                    if highlight_energy is not None and np.isfinite(highlight_energy):
                        optimal_full_energy = float(highlight_energy)
            if not optimal_full_path:
                fallback_path = self._combine_ordered_segments(valid_ordered_segments)
                if len(fallback_path) >= 2:
                    optimal_full_path = [int(node) for node in list(fallback_path)]
                    fallback_energy = self._ctx_full_path_energy(
                        scatter_coords,
                        optimal_full_path,
                        anchors,
                        path_order,
                    )
                    if fallback_energy is not None and np.isfinite(fallback_energy):
                        optimal_full_energy = float(fallback_energy)

            optimal_segment_records = self._ctx_segment_records_for_full_path(
                optimal_full_path,
                anchors,
                path_order,
            )

            optimal_subc_energy = None
            if subc_paths:
                start_channel = str(order_channels[1]) if len(order_channels) >= 2 else ""
                start_index = int(anchors[start_channel]) if start_channel in anchors else None
                if start_index is not None and subc_anchor_index is not None:
                    subc_ref_unit = self._reference_unit_vector(start_index, int(subc_anchor_index))
                    if subc_ref_unit is not None:
                        subc_length_candidates = []
                        for path_nodes in list(subc_paths):
                            path_length = self._path_metric_length(metric_coords, path_nodes)
                            if path_length is None or not np.isfinite(path_length):
                                continue
                            subc_length_candidates.append(
                                (
                                    float(path_length),
                                    len(list(path_nodes or [])),
                                    [int(node) for node in list(path_nodes or [])],
                                )
                            )
                        if subc_length_candidates:
                            subc_length_candidates.sort(key=lambda item: (item[0], item[1]))
                            subc_optimal_path = [int(node) for node in list(subc_length_candidates[0][2])]
                            highlight_energy = self._path_directionality_energy(
                                scatter_coords,
                                subc_optimal_path,
                                subc_ref_unit,
                                line_start=np.asarray(scatter_coords[start_index, :], dtype=float),
                                line_end=np.asarray(scatter_coords[int(subc_anchor_index), :], dtype=float),
                                include_line_proximity=self._use_line_proximity_energy,
                            )
                            if highlight_energy is not None and np.isfinite(highlight_energy):
                                optimal_subc_energy = float(highlight_energy)

            group_payload = {
                "group": str(group_spec.get("name", "all")),
                "anchors": {str(key): int(value) for key, value in anchors.items()},
                "optimal_segments": [
                    self._path_record(
                        str(record.get("first", "")),
                        str(record.get("second", "")),
                        record.get("nodes", []),
                        tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist()),
                    )
                    for record in list(optimal_segment_records)
                ],
                "optimal_full_path": [int(node) for node in optimal_full_path] if len(optimal_full_path) >= 2 else [],
                "all_full_paths": all_full_paths,
                "full_path_count": 0,
                "ctx_path_count": 0,
                "all_pair_paths": group_pair_paths,
                "subc_anchor": int(subc_anchor_index) if subc_anchor_index is not None else None,
                "subc_paths": [list(path) for path in subc_paths],
                "subc_optimal_path": [int(node) for node in subc_optimal_path] if len(subc_optimal_path) >= 2 else [],
                "ctx_optimal_path_energy": float(optimal_full_energy) if optimal_full_energy is not None else None,
                "subc_optimal_path_energy": float(optimal_subc_energy) if optimal_subc_energy is not None else None,
                "subc_color": [float(value) for value in np.asarray(subc_color, dtype=float).tolist()] if subc_color is not None else [],
                "subc_path_count": int(len(subc_paths)),
                "use_line_proximity_energy": bool(self._use_line_proximity_energy),
            }
            group_payload["full_path_count"] = int(len(group_payload["all_full_paths"]))
            group_payload["ctx_path_count"] = int(group_payload["full_path_count"])
            if group_payload["optimal_full_path"]:
                valid_optimal_paths.append(group_payload["optimal_full_path"])
            project_group_paths.append(group_payload)

        if not project_group_paths:
            return None
        return {
            "channel_order": str(path_channel_order or self._triangular_color_order),
            "color_order": str(triangle_model.get("order", self._triangular_color_order)),
            "fit_mode": str(triangle_model.get("fit_mode", self._rgb_fit_mode)),
            "group_paths": project_group_paths,
            "optimal_full_path": list(valid_optimal_paths[0]) if valid_optimal_paths else [],
            "show_all_ordered_paths": bool(self._show_all_ordered_paths),
            "rotation_preset": str(self._rotation_preset),
            "radius": float(self._proximity_radius),
        }

    def _draw_triangular_anchor_paths(self, ax, x_plot, y_plot, point_colors, path_payload, group_name=None):
        if not isinstance(path_payload, dict):
            return
        target_group = str(group_name or "").strip().lower()
        group_payloads = []
        for group_payload in list(path_payload.get("group_paths", [])):
            payload = dict(group_payload or {})
            payload_group = str(payload.get("group", "all")).strip().lower()
            if target_group and payload_group != target_group:
                continue
            group_payloads.append(payload)
        if not group_payloads:
            return

        width_mode = self._normalize_path_width_scaling_mode(
            path_payload.get("width_scaling_mode", self._path_width_scaling_mode)
        )
        try:
            width_strength = max(
                0.05,
                float(path_payload.get("width_scaling_strength", self._path_width_scaling_strength)),
            )
        except Exception:
            width_strength = float(self._path_width_scaling_strength)
        all_segments = []
        all_colors = []
        all_widths = []
        highlighted_segments = []
        highlighted_colors = []
        highlighted_widths = []
        anchor_positions = []
        anchor_colors = []
        bundled_all_groups = {}
        bundled_highlight_groups = {}
        channel_order = [str(channel) for channel in str(path_payload.get("channel_order", self._triangular_color_order or ""))]
        ctx_scaling = self._energy_scaling_range(path_payload, "ctx")
        subc_scaling = self._energy_scaling_range(path_payload, "subc")

        for group_payload in group_payloads:
            anchors = dict(group_payload.get("anchors", {}))
            ctx_all_paths = [
                [int(node) for node in list(path_nodes or [])]
                for path_nodes in list(group_payload.get("all_full_paths", []))
            ]
            ctx_all_energies = [
                float(value) if np.isfinite(value) else float("nan")
                for value in np.asarray(group_payload.get("ctx_path_energies", []), dtype=float).reshape(-1).tolist()
            ]
            if self._show_all_ordered_paths:
                for idx, path_nodes in enumerate(ctx_all_paths):
                    energy = ctx_all_energies[idx] if idx < len(ctx_all_energies) else None
                    path_width = self._path_display_width(
                        self._edge_linewidth,
                        energy,
                        ctx_scaling,
                        mode="scatter",
                        scaling_mode=width_mode,
                        scaling_strength=width_strength,
                    )
                    for record in self._ctx_segment_records_for_full_path(path_nodes, anchors, channel_order):
                        color = tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist())
                        node_pairs = self._path_edge_pairs(record.get("nodes", []))
                        if self._use_edge_bundling and node_pairs.size:
                            bundle_key = (
                                tuple(np.round(np.asarray(color, dtype=float), 6).tolist()),
                                round(float(path_width), 2),
                            )
                            bundled_all_groups.setdefault(bundle_key, []).append(node_pairs)
                        else:
                            segments = self._path_segments(x_plot, y_plot, record.get("nodes", []))
                            for segment in segments:
                                all_segments.append(segment)
                                all_colors.append(color)
                                all_widths.append(path_width)
                subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
                if subc_color.shape != (3,):
                    subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
                subc_energies = [
                    float(value) if np.isfinite(value) else float("nan")
                    for value in np.asarray(group_payload.get("subc_path_energies", []), dtype=float).reshape(-1).tolist()
                ]
                for idx, path_nodes in enumerate(list(group_payload.get("subc_paths", []))):
                    nodes = [int(node) for node in list(path_nodes or [])]
                    path_width = self._path_display_width(
                        self._edge_linewidth,
                        subc_energies[idx] if idx < len(subc_energies) else None,
                        subc_scaling,
                        mode="scatter",
                        scaling_mode=width_mode,
                        scaling_strength=width_strength,
                    )
                    subc_color_tuple = tuple(subc_color.tolist())
                    node_pairs = self._path_edge_pairs(nodes)
                    if self._use_edge_bundling and node_pairs.size:
                        bundle_key = (
                            tuple(np.round(np.asarray(subc_color_tuple, dtype=float), 6).tolist()),
                            round(float(path_width), 2),
                        )
                        bundled_all_groups.setdefault(bundle_key, []).append(node_pairs)
                    else:
                        segments = self._path_segments(x_plot, y_plot, nodes)
                        for segment in segments:
                            all_segments.append(segment)
                            all_colors.append(subc_color_tuple)
                            all_widths.append(path_width)

            optimal_full_path = [int(node) for node in list(group_payload.get("optimal_full_path", []))]
            optimal_ctx_width = self._path_display_width(
                self._edge_linewidth,
                group_payload.get("ctx_optimal_path_energy"),
                ctx_scaling,
                mode="scatter",
                scaling_mode=width_mode,
                scaling_strength=width_strength,
            )
            for record in self._ctx_segment_records_for_full_path(optimal_full_path, anchors, channel_order):
                color = tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist())
                node_pairs = self._path_edge_pairs(record.get("nodes", []))
                if self._use_edge_bundling and node_pairs.size:
                    bundle_key = (
                        tuple(np.round(np.asarray(color, dtype=float), 6).tolist()),
                        round(float(optimal_ctx_width), 2),
                    )
                    bundled_highlight_groups.setdefault(bundle_key, []).append(node_pairs)
                else:
                    segments = self._path_segments(x_plot, y_plot, record.get("nodes", []))
                    for segment in segments:
                        highlighted_segments.append(segment)
                        highlighted_colors.append(color)
                        highlighted_widths.append(optimal_ctx_width)

            subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
            if subc_color.shape != (3,):
                subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
            subc_segments = self._path_segments(x_plot, y_plot, list(group_payload.get("subc_optimal_path", [])))
            subc_optimal_width = self._path_display_width(
                self._edge_linewidth,
                group_payload.get("subc_optimal_path_energy"),
                subc_scaling,
                mode="scatter",
                scaling_mode=width_mode,
                scaling_strength=width_strength,
            )
            if self._use_edge_bundling:
                node_pairs = self._path_edge_pairs(list(group_payload.get("subc_optimal_path", [])))
                if node_pairs.size:
                    bundle_key = (
                        tuple(np.round(np.asarray(subc_color, dtype=float), 6).tolist()),
                        round(float(subc_optimal_width), 2),
                    )
                    bundled_highlight_groups.setdefault(bundle_key, []).append(node_pairs)
            else:
                for segment in subc_segments:
                    highlighted_segments.append(segment)
                    highlighted_colors.append(tuple(subc_color.tolist()))
                    highlighted_widths.append(subc_optimal_width)

            for channel in ("R", "G", "B"):
                index = dict(group_payload.get("anchors", {})).get(channel)
                if index is None:
                    continue
                index = int(index)
                if index < 0 or index >= len(x_plot):
                    continue
                anchor_positions.append((float(x_plot[index]), float(y_plot[index])))
                if channel == "R":
                    anchor_colors.append("#ef4444")
                elif channel == "G":
                    anchor_colors.append("#22c55e")
                else:
                    anchor_colors.append("#3b82f6")

            subc_anchor_index = group_payload.get("subc_anchor")
            if subc_anchor_index is not None:
                subc_anchor_index = int(subc_anchor_index)
                if 0 <= subc_anchor_index < len(x_plot):
                    anchor_positions.append((float(x_plot[subc_anchor_index]), float(y_plot[subc_anchor_index])))
                    if (
                        0 <= subc_anchor_index < np.asarray(point_colors, dtype=float).shape[0]
                        and np.all(np.isfinite(np.asarray(point_colors[subc_anchor_index], dtype=float)))
                    ):
                        anchor_colors.append(tuple(np.asarray(point_colors[subc_anchor_index], dtype=float).tolist()))
                    else:
                        anchor_colors.append("#111827")

        if self._use_edge_bundling and bundled_all_groups:
            for (color_key, width_key), group_pairs in bundled_all_groups.items():
                merged_pairs = np.vstack(group_pairs) if group_pairs else np.zeros((0, 2), dtype=int)
                bundled_segments = self._bundled_segments_from_pairs(merged_pairs)
                for segment in bundled_segments:
                    all_segments.append(segment)
                    all_colors.append(tuple(color_key))
                    all_widths.append(float(width_key))
        if self._use_edge_bundling and bundled_highlight_groups:
            for (color_key, width_key), group_pairs in bundled_highlight_groups.items():
                merged_pairs = np.vstack(group_pairs) if group_pairs else np.zeros((0, 2), dtype=int)
                bundled_segments = self._bundled_segments_from_pairs(merged_pairs)
                for segment in bundled_segments:
                    highlighted_segments.append(segment)
                    highlighted_colors.append(tuple(color_key))
                    highlighted_widths.append(float(width_key))

        if all_segments:
            ax.add_collection(
                LineCollection(
                    np.asarray(all_segments, dtype=float),
                    colors=all_colors,
                    linewidths=all_widths,
                    alpha=0.35,
                    zorder=3,
                )
            )
        if highlighted_segments:
            ax.add_collection(
                LineCollection(
                    np.asarray(highlighted_segments, dtype=float),
                    colors=highlighted_colors,
                    linewidths=highlighted_widths,
                    alpha=0.95,
                    zorder=4,
                )
            )
        if anchor_positions:
            anchor_positions = np.asarray(anchor_positions, dtype=float)
            ax.scatter(
                anchor_positions[:, 0],
                anchor_positions[:, 1],
                s=92,
                c=anchor_colors,
                edgecolors="#111827",
                linewidths=1.1,
                zorder=5,
            )

    def _draw_active_anchor_markers(self, ax, x_plot, y_plot, group_name=None):
        if not self._use_triangular_rgb:
            return
        target_groups = []
        if group_name:
            target_groups = [str(group_name).strip().lower()]
        else:
            target_groups = [str(spec.get("name", "all")).strip().lower() for spec in self._path_group_specs()]
        positions = []
        colors = []
        for current_group in target_groups:
            anchors = dict(self._effective_anchor_indices_for_group(current_group))
            for channel in ("R", "G", "B"):
                if channel not in anchors:
                    continue
                idx = int(anchors[channel])
                if idx < 0 or idx >= len(x_plot):
                    continue
                positions.append((float(x_plot[idx]), float(y_plot[idx])))
                colors.append(tuple(self._rgb_basis_color(channel).tolist()))
        if not positions:
            return
        coords = np.asarray(positions, dtype=float)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=90,
            c=colors,
            edgecolors="#111827",
            linewidths=1.1,
            zorder=5,
        )

    @staticmethod
    def _edge_subset_for_indices(edge_pairs, edge_distances, allowed_indices):
        pairs = np.asarray(edge_pairs, dtype=int)
        distances = np.asarray(edge_distances, dtype=float).reshape(-1)
        allowed = {int(value) for value in np.asarray(allowed_indices, dtype=int).reshape(-1).tolist()}
        if pairs.ndim != 2 or pairs.shape[0] == 0 or not allowed:
            return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
        keep_mask = np.asarray(
            [(int(pair[0]) in allowed and int(pair[1]) in allowed) for pair in pairs.tolist()],
            dtype=bool,
        )
        if not np.any(keep_mask):
            return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
        subset_pairs = np.asarray(pairs[keep_mask], dtype=int)
        subset_distances = np.asarray(distances[keep_mask], dtype=float) if distances.shape[0] == pairs.shape[0] else np.zeros(subset_pairs.shape[0], dtype=float)
        return subset_pairs, subset_distances

    def _draw_proximity_overlay(self, ax, x_plot, y_plot):
        if not self._show_proximity_circles or self._proximity_radius <= 0.0:
            return
        for x_coord, y_coord in zip(x_plot, y_plot):
            ax.add_patch(
                Circle(
                    (float(x_coord), float(y_coord)),
                    radius=float(self._proximity_radius),
                    facecolor="#9ca3af",
                    edgecolor="#6b7280",
                    linewidth=0.45,
                    alpha=0.11,
                    zorder=0,
                )
            )

    def _on_proximity_toggled(self, checked):
        self._show_proximity_circles = bool(checked)
        self._sync_proximity_controls()
        self._render()

    def _on_proximity_slider_changed(self, value):
        self._proximity_radius = self._slider_to_radius(value)
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_edge_width_changed(self, value):
        try:
            self._edge_linewidth = max(0.05, float(value))
        except Exception:
            self._edge_linewidth = 0.45
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
        self._render()

    def _on_path_width_mode_changed(self, _index):
        self._path_width_scaling_mode = self._normalize_path_width_scaling_mode(
            self.path_width_mode_combo.currentData()
        )
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
        self._render()

    def _on_path_width_scale_changed(self, value):
        try:
            self._path_width_scaling_strength = max(0.05, float(value))
        except Exception:
            self._path_width_scaling_strength = 2.0
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
        self._render()

    def _on_all_paths_toggled(self, checked):
        self._show_all_ordered_paths = bool(checked)
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
        self._render()

    def _on_show_adjacency_toggled(self, checked):
        self._show_adjacency_edges = bool(checked)
        self._render()

    def _on_edge_bundling_toggled(self, checked):
        self._use_edge_bundling = bool(checked)
        self._render()

    def _on_direction_filter_toggled(self, checked):
        self._use_directionality_filter = bool(checked)
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_line_proximity_toggled(self, checked):
        self._use_line_proximity_energy = bool(checked)
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_path_order_combo_changed(self, index):
        requested = [
            self.path_order_first_combo.currentData(),
            self.path_order_second_combo.currentData(),
            self.path_order_third_combo.currentData(),
        ]
        current = self._path_channel_order or self._triangular_color_order
        self._path_channel_order = self._coerce_path_channel_order(requested, fallback=current)
        self._populate_path_order_combos()
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_project_paths_clicked(self):
        if self._project_paths_callback is None or not isinstance(self._project_paths_payload, dict):
            return
        has_projectable_path = False
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            if len(list(group_payload.get("optimal_full_path", []))) >= 2:
                has_projectable_path = True
                break
            if len(list(group_payload.get("subc_optimal_path", []))) >= 2:
                has_projectable_path = True
                break
        if not has_projectable_path and len(list(self._project_paths_payload.get("optimal_full_path", []))) < 2:
            return
        self._project_paths_callback(dict(self._project_paths_payload))

    def _path_export_node(self, node_index):
        idx = int(node_index)
        node_id = str(self._point_ids[idx]) if 0 <= idx < self._point_ids.shape[0] else str(idx)
        node_name = str(self._point_labels[idx]) if 0 <= idx < self._point_labels.shape[0] else f"Point {idx + 1}"
        if 0 <= idx < self._display_coords.shape[0]:
            x_coord = float(self._display_coords[idx, 0])
            y_coord = float(self._display_coords[idx, 1])
        else:
            x_coord = float("nan")
            y_coord = float("nan")
        return {
            "scatter_index": idx,
            "node_label": node_id,
            "node_name": node_name,
            "x_coord": x_coord,
            "y_coord": y_coord,
        }

    def _invalidate_generated_paths(self):
        self._project_paths_payload = None

    def _clear_free_energy_payload(self):
        if not isinstance(self._project_paths_payload, dict):
            return
        self._project_paths_payload.pop("free_energy_payload", None)
        self._project_paths_payload.pop("energy_width_scaling", None)
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            if not isinstance(group_payload, dict):
                continue
            group_payload.pop("ctx_path_energies", None)
            group_payload.pop("subc_path_energies", None)
            group_payload.pop("ctx_optimal_path_energy", None)
            group_payload.pop("subc_optimal_path_energy", None)

    def _on_free_energy_lambda_changed(self, _value):
        if not isinstance(self._project_paths_payload, dict):
            return
        self._clear_free_energy_payload()
        self._sync_proximity_controls()
        self._render()

    def _on_generate_paths_clicked(self):
        if not self._use_triangular_rgb:
            return
        x_plot, y_plot = self._rotate_points(self._x, self._y, self._rotation_preset)
        triangle_model = self._rgb_model(
            x_plot,
            y_plot,
            self._triangular_color_order,
            fit_mode=self._rgb_fit_mode,
        )
        point_colors = self._rgb_colors_from_model(x_plot, y_plot, triangle_model)
        visible_edge_pairs = self._visible_edge_pairs()
        visible_edge_distances = self._visible_edge_distances()
        payload = self._build_triangular_anchor_paths_payload(
            x_plot,
            y_plot,
            triangle_model,
            self._path_channel_order,
            point_colors,
            visible_edge_pairs,
            visible_edge_distances,
        )
        if isinstance(payload, dict):
            payload["point_colors"] = np.asarray(point_colors, dtype=float).tolist()
            payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
            payload["edge_linewidth"] = float(self._edge_linewidth)
            payload["width_scaling_mode"] = self._path_width_scaling_mode
            payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
        self._project_paths_payload = payload
        self._render()

    @staticmethod
    def _normalize_free_energy_lambda(value):
        try:
            lam = float(value)
        except Exception:
            lam = 1.0
        return max(1e-6, lam)

    @staticmethod
    def _point_distances_to_line(sample_points, line_start, line_end):
        points = np.asarray(sample_points, dtype=float)
        start = np.asarray(line_start, dtype=float).reshape(-1)
        end = np.asarray(line_end, dtype=float).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 2 or start.shape != (2,) or end.shape != (2,):
            return None, None
        line_vector = np.asarray(end - start, dtype=float)
        line_length = float(np.linalg.norm(line_vector))
        if not np.isfinite(line_length) or line_length <= 1e-12:
            return None, None
        line_unit = line_vector / line_length
        rel = points - start[np.newaxis, :]
        cross_vals = rel[:, 0] * line_unit[1] - rel[:, 1] * line_unit[0]
        return np.abs(np.asarray(cross_vals, dtype=float)), float(line_length)

    @staticmethod
    def _path_directionality_energy(
        coords,
        path_nodes,
        reference_unit_vector,
        line_start=None,
        line_end=None,
        *,
        include_line_proximity=True,
    ):
        nodes = np.asarray([int(node) for node in list(path_nodes or [])], dtype=int)
        points = np.asarray(coords, dtype=float)
        ref = np.asarray(reference_unit_vector, dtype=float).reshape(-1)
        if nodes.size < 2 or points.ndim != 2 or points.shape[1] != 2 or ref.shape != (2,):
            return None
        if np.any((nodes < 0) | (nodes >= points.shape[0])):
            return None
        if not np.all(np.isfinite(points[nodes, :])) or not np.all(np.isfinite(ref)):
            return None
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm <= 0.0:
            return None
        ref_unit = ref / ref_norm
        steps = np.diff(points[nodes, :], axis=0)
        if steps.shape[0] == 0:
            return None
        step_norms = np.linalg.norm(steps, axis=1)
        valid = np.isfinite(step_norms) & (step_norms > 1e-12)
        if not np.any(valid):
            return None
        step_units = steps[valid, :] / step_norms[valid, np.newaxis]
        alignment = np.clip(step_units @ ref_unit, -1.0, 1.0)
        direction_penalties = 1.0 - alignment

        line_start_point = (
            np.asarray(line_start, dtype=float).reshape(-1)
            if line_start is not None
            else np.asarray(points[nodes[0], :], dtype=float).reshape(-1)
        )
        line_end_point = (
            np.asarray(line_end, dtype=float).reshape(-1)
            if line_end is not None
            else np.asarray(points[nodes[-1], :], dtype=float).reshape(-1)
        )
        midpoints = 0.5 * (points[nodes[:-1], :] + points[nodes[1:], :])
        midpoint_distances, reference_length = GradientScatterDialog._point_distances_to_line(
            midpoints[valid, :],
            line_start_point,
            line_end_point,
        )
        if (
            not include_line_proximity
            or midpoint_distances is None
            or reference_length is None
        ):
            proximity_penalties = np.zeros_like(direction_penalties, dtype=float)
        else:
            proximity_penalties = np.asarray(midpoint_distances, dtype=float) / float(reference_length)

        penalties = direction_penalties + proximity_penalties
        return float(np.sum(penalties))

    @staticmethod
    def _stable_free_energy(energies, lam):
        values = np.asarray(energies, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        lam = GradientScatterDialog._normalize_free_energy_lambda(lam)
        scaled = -lam * values
        max_scaled = float(np.max(scaled))
        return float(-(1.0 / lam) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))))

    def _reference_unit_vector(self, start_index, end_index):
        coords = np.asarray(self._display_coords, dtype=float)
        start = int(start_index)
        end = int(end_index)
        if (
            coords.ndim != 2
            or coords.shape[1] != 2
            or start < 0
            or end < 0
            or start >= coords.shape[0]
            or end >= coords.shape[0]
        ):
            return None
        vector = np.asarray(coords[end, :] - coords[start, :], dtype=float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return vector / norm

    def _ctx_full_path_energy(
        self,
        coords,
        full_path_nodes,
        anchors,
        channel_order,
        *,
        include_line_proximity=None,
    ):
        order = [str(channel) for channel in list(channel_order or []) if str(channel) in anchors]
        if len(order) < 2:
            return None
        if include_line_proximity is None:
            include_line_proximity = self._use_line_proximity_energy
        if len(order) == 2:
            start_anchor = int(anchors[order[0]])
            end_anchor = int(anchors[order[1]])
            ref_unit = self._reference_unit_vector(start_anchor, end_anchor)
            if ref_unit is None:
                return None
            return self._path_directionality_energy(
                coords,
                full_path_nodes,
                ref_unit,
                line_start=np.asarray(coords[start_anchor, :], dtype=float),
                line_end=np.asarray(coords[end_anchor, :], dtype=float),
                include_line_proximity=include_line_proximity,
            )

        nodes = [int(node) for node in list(full_path_nodes or [])]
        if len(nodes) < 2:
            return None
        start_anchor = int(anchors[order[0]])
        middle_anchor = int(anchors[order[1]])
        end_anchor = int(anchors[order[2]])
        if nodes[0] != start_anchor or nodes[-1] != end_anchor:
            return None
        try:
            split_index = next(
                idx
                for idx, node in enumerate(nodes)
                if int(node) == middle_anchor and idx > 0 and idx < len(nodes) - 1
            )
        except StopIteration:
            return None

        first_segment = nodes[: split_index + 1]
        second_segment = nodes[split_index:]
        ref_first = self._reference_unit_vector(start_anchor, middle_anchor)
        ref_second = self._reference_unit_vector(middle_anchor, end_anchor)
        if ref_first is None or ref_second is None:
            return None
        energy_first = self._path_directionality_energy(
            coords,
            first_segment,
            ref_first,
            line_start=np.asarray(coords[start_anchor, :], dtype=float),
            line_end=np.asarray(coords[middle_anchor, :], dtype=float),
            include_line_proximity=include_line_proximity,
        )
        energy_second = self._path_directionality_energy(
            coords,
            second_segment,
            ref_second,
            line_start=np.asarray(coords[middle_anchor, :], dtype=float),
            line_end=np.asarray(coords[end_anchor, :], dtype=float),
            include_line_proximity=include_line_proximity,
        )
        if energy_first is None or energy_second is None:
            return None
        return float(energy_first + energy_second)

    def _compute_free_energy_payload(self, lam):
        if not isinstance(self._project_paths_payload, dict):
            return None
        lam = self._normalize_free_energy_lambda(lam)
        channel_order = str(self._project_paths_payload.get("channel_order", self._triangular_color_order or "")).strip()
        if len(channel_order) < 2:
            return None
        coords = np.asarray(self._display_coords, dtype=float)
        groups = []
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            group = dict(group_payload or {})
            anchors = {str(key): int(value) for key, value in dict(group.get("anchors", {})).items()}
            families = []
            order = [str(channel) for channel in channel_order if str(channel) in anchors]
            if len(order) >= 2:
                ctx_records = []
                for full_path in list(group.get("all_full_paths", [])):
                    energy = self._ctx_full_path_energy(coords, full_path, anchors, order)
                    if energy is not None and np.isfinite(energy):
                        ctx_records.append(
                            {
                                "nodes": [int(node) for node in list(full_path or [])],
                                "energy": float(energy),
                            }
                        )
                if ctx_records:
                    ctx_energies = [float(record["energy"]) for record in ctx_records]
                    ctx_colors = []
                    for record in list(group.get("optimal_segments", [])):
                        color = np.asarray(dict(record).get("color", (0.2, 0.2, 0.2)), dtype=float).reshape(-1)
                        if color.shape == (3,):
                            ctx_colors.append(color)
                    if ctx_colors:
                        ctx_color = np.mean(np.asarray(ctx_colors, dtype=float), axis=0)
                    else:
                        ctx_color = np.asarray((0.2, 0.2, 0.2), dtype=float)
                    reference_vectors = []
                    segment_labels = []
                    for idx in range(len(order) - 1):
                        ref_unit = self._reference_unit_vector(anchors[order[idx]], anchors[order[idx + 1]])
                        if ref_unit is not None:
                            reference_vectors.append([float(value) for value in np.asarray(ref_unit, dtype=float).tolist()])
                        segment_labels.append(f"{order[idx]}{order[idx + 1]}")
                    families.append(
                        {
                            "label": "CTX",
                            "family_type": "ctx",
                            "segment_labels": segment_labels,
                            "reference_vectors": reference_vectors,
                            "energies": [float(value) for value in ctx_energies],
                            "path_energies": ctx_records,
                            "free_energy": self._stable_free_energy(ctx_energies, lam),
                            "color": [float(value) for value in np.asarray(ctx_color, dtype=float).tolist()],
                            "n_paths": int(len(ctx_energies)),
                        }
                    )

            if len(channel_order) >= 2 and channel_order[1] in anchors and group.get("subc_anchor") is not None:
                start_index = int(anchors[channel_order[1]])
                subc_index = int(group.get("subc_anchor"))
                ref_unit = self._reference_unit_vector(start_index, subc_index)
                if ref_unit is not None:
                    subc_records = []
                    for path_nodes in list(group.get("subc_paths", [])):
                        energy = self._path_directionality_energy(
                            coords,
                            path_nodes,
                            ref_unit,
                            line_start=np.asarray(coords[start_index, :], dtype=float),
                            line_end=np.asarray(coords[subc_index, :], dtype=float),
                            include_line_proximity=self._use_line_proximity_energy,
                        )
                        if energy is not None and np.isfinite(energy):
                            subc_records.append(
                                {
                                    "nodes": [int(node) for node in list(path_nodes or [])],
                                    "energy": float(energy),
                                }
                            )
                    if subc_records:
                        energies = [float(record["energy"]) for record in subc_records]
                        subc_color = np.asarray(group.get("subc_color", (0.1, 0.1, 0.1)), dtype=float).reshape(-1)
                        if subc_color.shape != (3,):
                            subc_color = np.asarray((0.1, 0.1, 0.1), dtype=float)
                        subc_name = ""
                        if 0 <= subc_index < self._point_labels.shape[0]:
                            subc_name = str(self._point_labels[subc_index]).strip()
                        families.append(
                            {
                                "label": "SUBC",
                                "family_type": "subc",
                                "segment_labels": [f"{channel_order[1]}->{subc_name or 'thal'}"],
                                "reference_vector": [float(value) for value in ref_unit.tolist()],
                                "energies": [float(value) for value in energies],
                                "path_energies": subc_records,
                                "free_energy": self._stable_free_energy(energies, lam),
                                "color": [float(value) for value in subc_color.tolist()],
                                "n_paths": int(len(energies)),
                            }
                        )

            if families:
                groups.append(
                    {
                        "group": str(group.get("group", "all")),
                        "families": families,
                    }
                )

        if not groups:
            return None
        return {
            "title": self._title,
            "lambda": float(lam),
            "rotation": str(self._rotation_preset),
            "x_axis_label": self._rotate_axis_labels(self._x_label, self._y_label, self._rotation_preset)[0],
            "y_axis_label": self._rotate_axis_labels(self._x_label, self._y_label, self._rotation_preset)[1],
            "use_line_proximity_energy": bool(self._use_line_proximity_energy),
            "groups": groups,
        }

    def _apply_free_energy_scaling(self, free_energy_payload):
        if not isinstance(self._project_paths_payload, dict) or not isinstance(free_energy_payload, dict):
            return

        family_energy_ranges = {}
        for family_type in ("ctx", "subc"):
            energies = []
            for group in list(free_energy_payload.get("groups", [])):
                for family in list(dict(group).get("families", [])):
                    if str(dict(family).get("family_type", "")).strip().lower() != family_type:
                        continue
                    for record in list(dict(family).get("path_energies", [])):
                        try:
                            value = float(dict(record).get("energy"))
                        except Exception:
                            continue
                        if np.isfinite(value):
                            energies.append(value)
            if energies:
                energy_values = np.asarray(energies, dtype=float)
                family_energy_ranges[family_type] = {
                    "min": float(np.min(energy_values)),
                    "max": float(np.max(energy_values)),
                }

        group_payloads = {}
        for group in list(self._project_paths_payload.get("group_paths", [])):
            if isinstance(group, dict):
                group_payloads[str(group.get("group", "all")).strip().lower()] = group
        for free_group in list(free_energy_payload.get("groups", [])):
            free_group_dict = dict(free_group or {})
            group_name = str(free_group_dict.get("group", "all")).strip().lower()
            target_group = group_payloads.get(group_name)
            if target_group is None:
                continue
            ctx_lookup = {}
            subc_lookup = {}
            for family in list(free_group_dict.get("families", [])):
                family_dict = dict(family or {})
                family_type = str(family_dict.get("family_type", "")).strip().lower()
                for record in list(family_dict.get("path_energies", [])):
                    record_dict = dict(record or {})
                    path_key = tuple(int(node) for node in list(record_dict.get("nodes", [])))
                    try:
                        energy = float(record_dict.get("energy"))
                    except Exception:
                        continue
                    if family_type == "ctx":
                        ctx_lookup[path_key] = energy
                    elif family_type == "subc":
                        subc_lookup[path_key] = energy
            target_group["ctx_path_energies"] = [
                float(ctx_lookup.get(tuple(int(node) for node in list(path or [])), float("nan")))
                for path in list(target_group.get("all_full_paths", []))
            ]
            target_group["subc_path_energies"] = [
                float(subc_lookup.get(tuple(int(node) for node in list(path or [])), float("nan")))
                for path in list(target_group.get("subc_paths", []))
            ]
            optimal_ctx_key = tuple(int(node) for node in list(target_group.get("optimal_full_path", [])))
            optimal_subc_key = tuple(int(node) for node in list(target_group.get("subc_optimal_path", [])))
            target_group["ctx_optimal_path_energy"] = (
                float(ctx_lookup[optimal_ctx_key]) if optimal_ctx_key in ctx_lookup else None
            )
            target_group["subc_optimal_path_energy"] = (
                float(subc_lookup[optimal_subc_key]) if optimal_subc_key in subc_lookup else None
            )

        self._project_paths_payload["energy_width_scaling"] = family_energy_ranges
        self._project_paths_payload["free_energy_payload"] = dict(free_energy_payload)
        self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
        self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
        self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)

    def _on_compute_free_energy_clicked(self):
        if not isinstance(self._project_paths_payload, dict):
            return
        payload = self._compute_free_energy_payload(self.free_energy_lambda_spin.value())
        if not isinstance(payload, dict):
            return
        self._apply_free_energy_scaling(payload)
        self._sync_proximity_controls()
        self._render()
        dialog = GradientFreeEnergyDialog(
            payload,
            parent=self,
            theme_name=self._theme_name,
        )
        self._free_energy_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @staticmethod
    def _safe_json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        try:
            if value is not None and np.isnan(value):
                return None
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _safe_name_fragment(text):
        token = str(text or "").strip()
        if not token:
            return "gradient"
        cleaned = []
        for ch in token:
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        out = "".join(cleaned).strip("_")
        while "__" in out:
            out = out.replace("__", "_")
        return out or "gradient"

    @staticmethod
    def _derive_parc_scheme(path_text):
        text = str(path_text or "").strip()
        if not text:
            return ""
        path = Path(text)
        name = path.name
        stem = name[:-7] if name.endswith(".nii.gz") else path.stem
        base = stem
        scale = ""
        if "-" in stem:
            left, right = stem.rsplit("-", 1)
            if right.isdigit():
                base = left
                scale = f"_scale{right}"
        filtered = []
        for ch in base:
            if ch.isalnum():
                filtered.append(ch)
        return "".join(filtered) + scale

    def _numeric_point_ids(self):
        values = []
        for value in self._point_ids.tolist():
            try:
                values.append(int(str(value).strip()))
            except Exception:
                return np.asarray(self._point_ids, dtype=object)
        return np.asarray(values, dtype=int)

    def _gradient_component_arrays_for_export(self):
        metadata = dict(self._export_metadata or {})
        gradient1 = np.asarray(metadata.get("gradient1_values", self._gradient1), dtype=float).reshape(-1)
        if gradient1.shape[0] != self._x.shape[0]:
            gradient1 = np.asarray(self._gradient1, dtype=float).reshape(-1)

        gradient2_raw = metadata.get("gradient2_values", None)
        if gradient2_raw is not None:
            gradient2 = np.asarray(gradient2_raw, dtype=float).reshape(-1)
        else:
            gradient2 = np.full(self._x.shape, np.nan, dtype=float)
            x_text = str(self._x_label or "").strip().lower()
            y_text = str(self._y_label or "").strip().lower()
            if x_text == "gradient 2":
                gradient2 = np.asarray(self._x, dtype=float).reshape(-1)
            elif y_text == "gradient 2":
                gradient2 = np.asarray(self._y, dtype=float).reshape(-1)
        if gradient2.shape[0] != self._x.shape[0]:
            gradient2 = np.full(self._x.shape, np.nan, dtype=float)
        return gradient1, gradient2

    def _batch_export_node_payload(self, node_index, gradient1_values, gradient2_values):
        idx = int(node_index)
        node_label = str(self._point_ids[idx]) if 0 <= idx < self._point_ids.shape[0] else str(idx)
        node_name = str(self._point_labels[idx]) if 0 <= idx < self._point_labels.shape[0] else f"Point {idx + 1}"
        gradient1_coord = float(gradient1_values[idx]) if 0 <= idx < gradient1_values.shape[0] else float("nan")
        gradient2_coord = float(gradient2_values[idx]) if 0 <= idx < gradient2_values.shape[0] else float("nan")
        scatter_x = float(self._display_coords[idx, 0]) if 0 <= idx < self._display_coords.shape[0] else float("nan")
        scatter_y = float(self._display_coords[idx, 1]) if 0 <= idx < self._display_coords.shape[0] else float("nan")
        return {
            "node_index": idx,
            "node_label": node_label,
            "node_name": node_name,
            "gradient1_coord": gradient1_coord,
            "gradient2_coord": gradient2_coord,
            "scatter_x": scatter_x,
            "scatter_y": scatter_y,
        }

    def _batch_export_path_record(self, path_nodes, energy, gradient1_values, gradient2_values, **extra):
        nodes = [int(node) for node in list(path_nodes or [])]
        record = {
            "nodes": nodes,
            "node_labels": [str(self._point_ids[idx]) for idx in nodes],
            "node_names": [str(self._point_labels[idx]) for idx in nodes],
            "gradient1_coords": [float(gradient1_values[idx]) for idx in nodes],
            "gradient2_coords": [float(gradient2_values[idx]) for idx in nodes],
            "scatter_coords": [
                [float(self._display_coords[idx, 0]), float(self._display_coords[idx, 1])]
                for idx in nodes
            ],
            "energy": float(energy) if energy is not None and np.isfinite(float(energy)) else float("nan"),
        }
        record.update(extra)
        return record

    def _free_energy_export_payload(self):
        if not isinstance(self._project_paths_payload, dict):
            return None
        free_energy_payload = self._project_paths_payload.get("free_energy_payload")
        if not isinstance(free_energy_payload, dict):
            return None

        metadata = dict(self._export_metadata or {})
        gradient1_values, gradient2_values = self._gradient_component_arrays_for_export()
        channel_order = str(self._project_paths_payload.get("channel_order", self._triangular_color_order or "")).strip()
        color_order = str(self._project_paths_payload.get("color_order", self._triangular_color_order or "")).strip()
        path_group_lookup = {
            str(group.get("group", "all")).strip().lower(): dict(group)
            for group in list(self._project_paths_payload.get("group_paths", []))
            if isinstance(group, dict)
        }
        group_exports = []
        fixed_endpoints = {}

        for free_group in list(free_energy_payload.get("groups", [])):
            free_group_dict = dict(free_group or {})
            group_name = str(free_group_dict.get("group", "all")).strip().lower()
            path_group = path_group_lookup.get(group_name)
            if path_group is None:
                continue
            anchors = {str(key): int(value) for key, value in dict(path_group.get("anchors", {})).items()}
            group_record = {
                "group": group_name,
                "color_order": str(color_order),
                "path_order": str(channel_order),
                "anchors": {
                    channel: self._batch_export_node_payload(index, gradient1_values, gradient2_values)
                    for channel, index in anchors.items()
                },
                "subc_anchor": (
                    self._batch_export_node_payload(int(path_group.get("subc_anchor")), gradient1_values, gradient2_values)
                    if path_group.get("subc_anchor") is not None
                    else None
                ),
                "ctx_segment_labels": [
                    f"{channel_order[idx]}{channel_order[idx + 1]}"
                    for idx in range(max(0, len(channel_order) - 1))
                ],
                "ctx_reference_vectors": [],
                "subc_reference_vector": None,
                "ctx_paths": [],
                "ctx_optimal_path": None,
                "ctx_path_count": 0,
                "ctx_free_energy": float("nan"),
                "subc_paths": [],
                "subc_optimal_path": None,
                "subc_path_count": 0,
                "subc_free_energy": float("nan"),
            }
            fixed_endpoints[group_name] = {
                "path_order": str(channel_order),
                "anchors": {
                    channel: {
                        "node_label": str(node.get("node_label", "")),
                        "node_name": str(node.get("node_name", "")),
                    }
                    for channel, node in group_record["anchors"].items()
                },
            }

            ctx_lookup = {}
            subc_lookup = {}
            for family in list(free_group_dict.get("families", [])):
                family_dict = dict(family or {})
                family_type = str(family_dict.get("family_type", "")).strip().lower()
                if family_type == "ctx":
                    group_record["ctx_reference_vectors"] = [
                        [float(value) for value in np.asarray(vector, dtype=float).reshape(-1).tolist()]
                        for vector in list(family_dict.get("reference_vectors", []))
                    ]
                    group_record["ctx_free_energy"] = float(family_dict.get("free_energy", float("nan")))
                    for record in list(family_dict.get("path_energies", [])):
                        record_dict = dict(record or {})
                        nodes = [int(node) for node in list(record_dict.get("nodes", []))]
                        if len(nodes) < 2:
                            continue
                        energy = record_dict.get("energy")
                        ctx_lookup[tuple(nodes)] = float(energy) if energy is not None else float("nan")
                        group_record["ctx_paths"].append(
                            self._batch_export_path_record(
                                nodes,
                                energy,
                                gradient1_values,
                                gradient2_values,
                                family="CTX",
                                path_label=str(channel_order),
                                segment_labels=list(group_record["ctx_segment_labels"]),
                            )
                        )
                    group_record["ctx_path_count"] = int(family_dict.get("n_paths", len(group_record["ctx_paths"])))
                elif family_type == "subc":
                    reference = family_dict.get("reference_vector")
                    if reference is not None:
                        group_record["subc_reference_vector"] = [
                            float(value) for value in np.asarray(reference, dtype=float).reshape(-1).tolist()
                        ]
                    group_record["subc_free_energy"] = float(family_dict.get("free_energy", float("nan")))
                    for record in list(family_dict.get("path_energies", [])):
                        record_dict = dict(record or {})
                        nodes = [int(node) for node in list(record_dict.get("nodes", []))]
                        if len(nodes) < 2:
                            continue
                        energy = record_dict.get("energy")
                        subc_lookup[tuple(nodes)] = float(energy) if energy is not None else float("nan")
                        group_record["subc_paths"].append(
                            self._batch_export_path_record(
                                nodes,
                                energy,
                                gradient1_values,
                                gradient2_values,
                                family="SUBC",
                                path_label=f"{channel_order[1]}->thal" if len(channel_order) >= 2 else "->thal",
                                segment_labels=[f"{channel_order[1]}->thal"] if len(channel_order) >= 2 else ["->thal"],
                            )
                        )
                    group_record["subc_path_count"] = int(family_dict.get("n_paths", len(group_record["subc_paths"])))

            optimal_ctx_nodes = [int(node) for node in list(path_group.get("optimal_full_path", []))]
            if len(optimal_ctx_nodes) >= 2:
                group_record["ctx_optimal_path"] = self._batch_export_path_record(
                    optimal_ctx_nodes,
                    ctx_lookup.get(tuple(optimal_ctx_nodes), path_group.get("ctx_optimal_path_energy")),
                    gradient1_values,
                    gradient2_values,
                    family="CTX",
                    path_label=str(channel_order),
                    segment_labels=list(group_record["ctx_segment_labels"]),
                )

            optimal_subc_nodes = [int(node) for node in list(path_group.get("subc_optimal_path", []))]
            if len(optimal_subc_nodes) >= 2:
                group_record["subc_optimal_path"] = self._batch_export_path_record(
                    optimal_subc_nodes,
                    subc_lookup.get(tuple(optimal_subc_nodes), path_group.get("subc_optimal_path_energy")),
                    gradient1_values,
                    gradient2_values,
                    family="SUBC",
                    path_label=f"{channel_order[1]}->thal" if len(channel_order) >= 2 else "->thal",
                    segment_labels=[f"{channel_order[1]}->thal"] if len(channel_order) >= 2 else ["->thal"],
                )

            group_exports.append(group_record)

        if not group_exports:
            return None

        subject_id = str(metadata.get("subject_id", "") or "").strip()
        session_id = str(metadata.get("session_id", "") or "").strip()
        group_name = str(metadata.get("group", "") or "").strip()
        modality = str(metadata.get("modality", "") or "").strip()
        parc_path = str(metadata.get("parc_path", metadata.get("template_path", "")) or "").strip()
        parc_scheme = str(metadata.get("parc_scheme", "") or "").strip() or self._derive_parc_scheme(parc_path)
        adjacency_path = str(metadata.get("adjacency_path", "") or "").strip()
        covars_row = dict(metadata.get("covars_row", {}) or {})
        gradients_pair = np.asarray(
            metadata.get(
                "gradients_pair",
                np.column_stack((gradient1_values, gradient2_values)),
            ),
            dtype=float,
        )
        gradients_avg = np.asarray(metadata.get("gradients_avg", np.empty((0, 0), dtype=float)), dtype=float)

        def _group_metric(group_key, metric_key, default_value):
            normalized = str(group_key).strip().lower()
            for group in group_exports:
                if str(group.get("group", "")).strip().lower() == normalized:
                    return group.get(metric_key, default_value)
            if normalized in {"lh", "rh"}:
                for group in group_exports:
                    if str(group.get("group", "")).strip().lower() == "all":
                        return group.get(metric_key, default_value)
            return default_value

        ctx_path_count_lh = int(_group_metric("lh", "ctx_path_count", 0))
        ctx_path_count_rh = int(_group_metric("rh", "ctx_path_count", 0))
        subc_path_count_lh = int(_group_metric("lh", "subc_path_count", 0))
        subc_path_count_rh = int(_group_metric("rh", "subc_path_count", 0))
        ctx_free_energy_lh = float(_group_metric("lh", "ctx_free_energy", float("nan")))
        ctx_free_energy_rh = float(_group_metric("rh", "ctx_free_energy", float("nan")))
        subc_free_energy_lh = float(_group_metric("lh", "subc_free_energy", float("nan")))
        subc_free_energy_rh = float(_group_metric("rh", "subc_free_energy", float("nan")))

        summary = {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": group_name,
            "modality": modality,
            "parc_scheme": parc_scheme,
            "parc_path": parc_path,
            "adjacency_path": adjacency_path,
            "lambda_value": float(free_energy_payload.get("lambda", self.free_energy_lambda_spin.value())),
            "color_order": str(color_order),
            "path_order_override": str(channel_order),
            "axes": {
                "x": str(free_energy_payload.get("x_axis_label", self._x_label)),
                "y": str(free_energy_payload.get("y_axis_label", self._y_label)),
            },
            "directional_filter": bool(self._use_directionality_filter),
            "line_proximity_energy": bool(
                free_energy_payload.get("use_line_proximity_energy", self._use_line_proximity_energy)
            ),
            "endpoint_selection_mode": str(self._endpoint_selection_mode or "adaptive"),
            "max_direction_violations": 2,
            "ctx_path_count_lh": ctx_path_count_lh,
            "ctx_path_count_rh": ctx_path_count_rh,
            "subc_path_count_lh": subc_path_count_lh,
            "subc_path_count_rh": subc_path_count_rh,
            "ctx_free_energy_lh": ctx_free_energy_lh,
            "ctx_free_energy_rh": ctx_free_energy_rh,
            "subc_free_energy_lh": subc_free_energy_lh,
            "subc_free_energy_rh": subc_free_energy_rh,
            "fixed_endpoint_source": f"gui_{str(self._endpoint_selection_mode or 'adaptive')}",
            "fixed_endpoint_file": "",
            "fixed_endpoints": fixed_endpoints,
            "groups": group_exports,
        }
        summary_json = json.dumps(summary, default=self._safe_json_default, indent=2)

        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": group_name,
            "modality": modality,
            "parc_scheme": parc_scheme,
            "parc_path": parc_path,
            "adjacency_path": adjacency_path,
            "lambda_value": float(free_energy_payload.get("lambda", self.free_energy_lambda_spin.value())),
            "color_order": str(color_order),
            "path_order_override": str(channel_order),
            "x_axis_label": str(free_energy_payload.get("x_axis_label", self._x_label)),
            "y_axis_label": str(free_energy_payload.get("y_axis_label", self._y_label)),
            "directional_filter": bool(self._use_directionality_filter),
            "line_proximity_energy": bool(
                free_energy_payload.get("use_line_proximity_energy", self._use_line_proximity_energy)
            ),
            "endpoint_selection_mode": str(self._endpoint_selection_mode or "adaptive"),
            "max_direction_violations": 2,
            "ctx_path_count_lh": ctx_path_count_lh,
            "ctx_path_count_rh": ctx_path_count_rh,
            "subc_path_count_lh": subc_path_count_lh,
            "subc_path_count_rh": subc_path_count_rh,
            "ctx_free_energy_lh": ctx_free_energy_lh,
            "ctx_free_energy_rh": ctx_free_energy_rh,
            "subc_free_energy_lh": subc_free_energy_lh,
            "subc_free_energy_rh": subc_free_energy_rh,
            "parcel_labels": self._numeric_point_ids(),
            "parcel_names": np.asarray(self._point_labels, dtype=object),
            "hemisphere_codes": np.asarray(self._point_group_codes, dtype=int),
            "gradients_pair": np.asarray(gradients_pair, dtype=float),
            "gradients_avg": np.asarray(gradients_avg, dtype=float),
            "covars_row": dict(covars_row),
            "groups": group_exports,
            "summary_json": summary_json,
        }

    def _on_write_free_energy_clicked(self):
        export_payload = self._free_energy_export_payload()
        if export_payload is None:
            return
        subject_id = str(export_payload.get("subject_id", "") or "").strip()
        session_id = str(export_payload.get("session_id", "") or "").strip()
        order_tag = str(export_payload.get("path_order_override", self._path_channel_order or "RGB") or "RGB").strip()
        source_dir = Path(str(dict(self._export_metadata or {}).get("source_dir", Path.cwd())))
        if subject_id and session_id:
            default_name = f"sub-{subject_id}_ses-{session_id}_order-{order_tag}_desc-free_energy_paths.npz"
        else:
            source_name = str(dict(self._export_metadata or {}).get("source_name", self._title))
            default_name = f"{self._safe_name_fragment(source_name)}_order-{order_tag}_desc-free_energy_paths.npz"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Write free energy to NPZ",
            str(source_dir / default_name),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        covars_row = dict(export_payload.get("covars_row", {}) or {})
        try:
            np.savez_compressed(
                str(output_path),
                subject_id=np.asarray(str(export_payload.get("subject_id", ""))),
                session_id=np.asarray(str(export_payload.get("session_id", ""))),
                group=np.asarray(str(export_payload.get("group", ""))),
                modality=np.asarray(str(export_payload.get("modality", ""))),
                parc_scheme=np.asarray(str(export_payload.get("parc_scheme", ""))),
                parc_path=np.asarray(str(export_payload.get("parc_path", ""))),
                adjacency_path=np.asarray(str(export_payload.get("adjacency_path", ""))),
                lambda_value=np.asarray(float(export_payload.get("lambda_value", 1.0))),
                color_order=np.asarray(str(export_payload.get("color_order", ""))),
                path_order_override=np.asarray(str(export_payload.get("path_order_override", ""))),
                x_axis_label=np.asarray(str(export_payload.get("x_axis_label", ""))),
                y_axis_label=np.asarray(str(export_payload.get("y_axis_label", ""))),
                directional_filter=np.asarray(bool(export_payload.get("directional_filter", False))),
                line_proximity_energy=np.asarray(bool(export_payload.get("line_proximity_energy", True))),
                endpoint_selection_mode=np.asarray(str(export_payload.get("endpoint_selection_mode", "adaptive"))),
                max_direction_violations=np.asarray(int(export_payload.get("max_direction_violations", 2))),
                ctx_path_count_lh=np.asarray(int(export_payload.get("ctx_path_count_lh", 0)), dtype=int),
                ctx_path_count_rh=np.asarray(int(export_payload.get("ctx_path_count_rh", 0)), dtype=int),
                subc_path_count_lh=np.asarray(int(export_payload.get("subc_path_count_lh", 0)), dtype=int),
                subc_path_count_rh=np.asarray(int(export_payload.get("subc_path_count_rh", 0)), dtype=int),
                ctx_free_energy_lh=np.asarray(float(export_payload.get("ctx_free_energy_lh", float("nan"))), dtype=float),
                ctx_free_energy_rh=np.asarray(float(export_payload.get("ctx_free_energy_rh", float("nan"))), dtype=float),
                subc_free_energy_lh=np.asarray(float(export_payload.get("subc_free_energy_lh", float("nan"))), dtype=float),
                subc_free_energy_rh=np.asarray(float(export_payload.get("subc_free_energy_rh", float("nan"))), dtype=float),
                parcel_labels=np.asarray(export_payload.get("parcel_labels"), dtype=object),
                parcel_names=np.asarray(export_payload.get("parcel_names"), dtype=object),
                hemisphere_codes=np.asarray(export_payload.get("hemisphere_codes"), dtype=int),
                gradients_pair=np.asarray(export_payload.get("gradients_pair"), dtype=float),
                gradients_avg=np.asarray(export_payload.get("gradients_avg"), dtype=float),
                covars_row=np.asarray([covars_row], dtype=object),
                covars_row_json=np.asarray(json.dumps(covars_row, default=self._safe_json_default, indent=2)),
                fixed_endpoint_source=np.asarray(
                    f"gui_{str(export_payload.get('endpoint_selection_mode', 'adaptive'))}"
                ),
                fixed_endpoint_file=np.asarray(""),
                fixed_endpoints_json=np.asarray(
                    json.dumps(
                        {
                            str(group.get("group", "all")): {
                                "path_order": str(group.get("path_order", "")),
                                "anchors": {
                                    str(channel): {
                                        "node_label": str(node.get("node_label", "")),
                                        "node_name": str(node.get("node_name", "")),
                                    }
                                    for channel, node in dict(group.get("anchors", {})).items()
                                },
                            }
                            for group in list(export_payload.get("groups", []))
                        },
                        default=self._safe_json_default,
                        indent=2,
                    )
                ),
                groups=np.asarray(export_payload.get("groups", []), dtype=object),
                summary_json=np.asarray(str(export_payload.get("summary_json", ""))),
            )
        except Exception as exc:
            warn(f"Failed to write free energy NPZ `{output_path}`: {exc}")
            return
        warn(f"Wrote free energy NPZ to {output_path}")

    def _export_paths_payload(self):
        if not isinstance(self._project_paths_payload, dict):
            return None
        groups = []
        ctx_path_label = str(self._project_paths_payload.get("channel_order", self._triangular_color_order or "")).strip()
        ctx_segment_labels = [
            f"{ctx_path_label[idx]}{ctx_path_label[idx + 1]}"
            for idx in range(max(0, len(ctx_path_label) - 1))
        ]
        subc_from_endpoint = ctx_path_label[1] if len(ctx_path_label) >= 2 else ""
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            ctx_full_paths = []
            for path in list(group_payload.get("all_full_paths", [])):
                nodes = [int(node) for node in list(path or [])]
                if len(nodes) < 2:
                    continue
                ctx_full_paths.append([self._path_export_node(node) for node in nodes])
            subc_full_paths = []
            for path in list(group_payload.get("subc_paths", [])):
                nodes = [int(node) for node in list(path or [])]
                if len(nodes) < 2:
                    continue
                subc_full_paths.append([self._path_export_node(node) for node in nodes])
            ctx_optimal_nodes = [
                self._path_export_node(node)
                for node in list(group_payload.get("optimal_full_path", []))
            ]
            subc_optimal_nodes = [
                self._path_export_node(node)
                for node in list(group_payload.get("subc_optimal_path", []))
            ]
            ctx_endpoints = {}
            for channel, node_index in dict(group_payload.get("anchors", {})).items():
                try:
                    ctx_endpoints[str(channel)] = self._path_export_node(int(node_index))
                except Exception:
                    continue
            ordered_ctx_endpoints = []
            for channel in [str(channel) for channel in str(self._triangular_color_order or "RGB")]:
                if channel in ctx_endpoints:
                    ordered_ctx_endpoints.append(ctx_endpoints[channel])
            subc_endpoint = None
            if group_payload.get("subc_anchor") is not None:
                try:
                    subc_endpoint = self._path_export_node(int(group_payload.get("subc_anchor")))
                except Exception:
                    subc_endpoint = None
            subc_target_name = ""
            subc_target_label = ""
            if isinstance(subc_endpoint, dict):
                subc_target_name = str(subc_endpoint.get("node_name", "")).strip()
                subc_target_label = str(subc_endpoint.get("node_label", "")).strip()
            subc_path_label = (
                f"{subc_from_endpoint}->{subc_target_name or subc_target_label}"
                if subc_from_endpoint and (subc_target_name or subc_target_label)
                else str(subc_target_name or subc_target_label or subc_from_endpoint)
            )

            def _export_segment_record(record):
                rec = dict(record or {})
                first = str(rec.get("first", "")).strip()
                second = str(rec.get("second", "")).strip()
                nodes = [self._path_export_node(node) for node in list(rec.get("nodes", []))]
                return {
                    "pair_label": f"{first}{second}" if first or second else "",
                    "from_endpoint": first,
                    "to_endpoint": second,
                    "nodes": nodes,
                }

            ctx_optimal_segments_detail = [
                _export_segment_record(record)
                for record in list(group_payload.get("optimal_segments", []))
            ]
            ctx_segment_paths_detail = [
                _export_segment_record(record)
                for record in list(group_payload.get("all_pair_paths", []))
            ]
            ctx_optimal_path_detail = {
                "path_label": ctx_path_label,
                "segment_labels": list(ctx_segment_labels),
                "nodes": ctx_optimal_nodes,
            }
            ctx_paths_detail = [
                {
                    "path_label": ctx_path_label,
                    "segment_labels": list(ctx_segment_labels),
                    "nodes": path_nodes,
                }
                for path_nodes in ctx_full_paths
            ]
            subc_optimal_path_detail = {
                "path_label": subc_path_label,
                "from_endpoint": subc_from_endpoint,
                "to_endpoint_label": subc_target_label,
                "to_endpoint_name": subc_target_name,
                "nodes": subc_optimal_nodes,
            }
            subc_paths_detail = [
                {
                    "path_label": subc_path_label,
                    "from_endpoint": subc_from_endpoint,
                    "to_endpoint_label": subc_target_label,
                    "to_endpoint_name": subc_target_name,
                    "nodes": path_nodes,
                }
                for path_nodes in subc_full_paths
            ]
            groups.append(
                {
                    "group": str(group_payload.get("group", "all")),
                    "ctx_path_count": int(group_payload.get("ctx_path_count", len(ctx_full_paths))),
                    "subc_path_count": int(group_payload.get("subc_path_count", len(subc_full_paths))),
                    "ctx_path_label": ctx_path_label,
                    "ctx_segment_labels": list(ctx_segment_labels),
                    "ctx_endpoints": ctx_endpoints,
                    "ctx_ordered_endpoints": ordered_ctx_endpoints,
                    "subc_endpoint": subc_endpoint,
                    "subc_path_label": subc_path_label,
                    "subc_from_endpoint": subc_from_endpoint,
                    "ctx_optimal_path": ctx_optimal_nodes,
                    "ctx_paths": ctx_full_paths,
                    "ctx_optimal_path_detail": ctx_optimal_path_detail,
                    "ctx_paths_detail": ctx_paths_detail,
                    "ctx_optimal_segments": ctx_optimal_segments_detail,
                    "ctx_segment_paths": ctx_segment_paths_detail,
                    "subc_optimal_path": subc_optimal_nodes,
                    "subc_paths": subc_full_paths,
                    "subc_optimal_path_detail": subc_optimal_path_detail,
                    "subc_paths_detail": subc_paths_detail,
                }
            )
        return {
            "title": self._title,
            "x_axis_label": self._rotate_axis_labels(
                self._x_label,
                self._y_label,
                self._rotation_preset,
            )[0],
            "y_axis_label": self._rotate_axis_labels(
                self._x_label,
                self._y_label,
                self._rotation_preset,
            )[1],
            "rotation": self._rotation_preset,
            "radius": float(self._proximity_radius),
            "fit_mode": self._rgb_fit_mode,
            "color_order": self._triangular_color_order,
            "groups": groups,
        }

    def _on_export_paths_clicked(self):
        export_payload = self._export_paths_payload()
        if export_payload is None:
            return
        has_paths = any(
            int(group.get("ctx_path_count", 0)) > 0 or int(group.get("subc_path_count", 0)) > 0
            for group in export_payload["groups"]
        )
        if not has_paths:
            return
        default_name = "gradient_paths.json"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export classification paths",
            str(Path.cwd() / default_name),
            "JSON (*.json);;Text (*.txt)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in {".json", ".txt"}:
            if "Text" in selected_filter:
                output_path = output_path.with_suffix(".txt")
            else:
                output_path = output_path.with_suffix(".json")
        if output_path.suffix.lower() == ".txt":
            lines = [
                f"Title: {export_payload['title']}",
                f"X axis: {export_payload['x_axis_label']}",
                f"Y axis: {export_payload['y_axis_label']}",
                f"Rotation: {export_payload['rotation']}",
                f"Radius: {export_payload['radius']:.4f}",
                f"Fit mode: {export_payload['fit_mode']}",
                f"Color order: {export_payload['color_order']}",
                "",
            ]
            for group in export_payload["groups"]:
                lines.append(
                    f"[{str(group['group']).upper()}] ctx={int(group['ctx_path_count'])} subc={int(group['subc_path_count'])}"
                )
                if group["ctx_optimal_path"]:
                    lines.append("CTX optimal:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in group["ctx_optimal_path"]
                    )
                for idx, path_nodes in enumerate(group["ctx_paths"], start=1):
                    lines.append(f"CTX path {idx}:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in path_nodes
                    )
                if group["subc_optimal_path"]:
                    lines.append("SUBC optimal:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in group["subc_optimal_path"]
                    )
                for idx, path_nodes in enumerate(group["subc_paths"], start=1):
                    lines.append(f"SUBC path {idx}:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in path_nodes
                    )
                lines.append("")
            output_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            output_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")

    @staticmethod
    def _combine_ordered_segments(segments):
        combined = []
        for _first, _second, path in list(segments or []):
            nodes = [int(node) for node in list(path or [])]
            if len(nodes) < 2:
                return []
            if not combined:
                combined.extend(nodes)
            elif combined[-1] == nodes[0]:
                combined.extend(nodes[1:])
            else:
                combined.extend(nodes)
        return combined

    @staticmethod
    def _path_record(first, second, path_nodes, color):
        return {
            "first": str(first),
            "second": str(second),
            "nodes": [int(node) for node in list(path_nodes or [])],
            "color": [float(value) for value in np.asarray(color, dtype=float).reshape(3).tolist()],
        }

    @staticmethod
    def _combine_ordered_path_records(segment_records, max_full_paths=256):
        records = list(segment_records or [])
        if not records:
            return []
        max_full_paths = max(1, int(max_full_paths))
        combined = []

        def extend(segment_index, current_path):
            if len(combined) >= max_full_paths:
                return
            if segment_index >= len(records):
                if len(current_path) >= 2:
                    combined.append(list(current_path))
                return
            _first, _second, candidate_paths = records[segment_index]
            for path_nodes in list(candidate_paths or []):
                nodes = [int(node) for node in list(path_nodes or [])]
                if len(nodes) < 2:
                    continue
                if not current_path:
                    extend(segment_index + 1, nodes)
                else:
                    if current_path[-1] != nodes[0]:
                        continue
                    used = set(int(node) for node in current_path[:-1])
                    if any(int(node) in used for node in nodes[1:]):
                        continue
                    extend(segment_index + 1, list(current_path) + nodes[1:])
                if len(combined) >= max_full_paths:
                    break

        extend(0, [])
        return combined

    @staticmethod
    def _fallback_triangle_vertices(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            return np.array([[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=float)
        x_valid = x_values[finite_mask]
        y_valid = y_values[finite_mask]
        x_min, x_max, y_min, y_max = GradientScatterDialog._triangular_rgb_bounds(x_valid, y_valid)
        return np.array(
            [
                [0.5 * (x_min + x_max), y_max],
                [x_min, y_min],
                [x_max, y_min],
            ],
            dtype=float,
        )

    @staticmethod
    def _fit_square_outline(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            center_x = 0.0
            center_y = 0.0
            half = 1.0
        else:
            x_valid = x_values[finite_mask]
            y_valid = y_values[finite_mask]
            x_min, x_max, y_min, y_max = GradientScatterDialog._triangular_rgb_bounds(x_valid, y_valid)
            center_x = 0.5 * (x_min + x_max)
            center_y = 0.5 * (y_min + y_max)
            half = max(0.5 * (x_max - x_min), 0.5 * (y_max - y_min), 1e-6)
        left = center_x - half
        right = center_x + half
        bottom = center_y - half
        top = center_y + half
        outline = np.asarray(
            (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ),
            dtype=float,
        )
        anchor_points = np.asarray(
            (
                (center_x, top),
                (left, center_y),
                (right, center_y),
            ),
            dtype=float,
        )
        return outline, anchor_points

    @staticmethod
    def _triangular_rgb_bounds(x_values, y_values):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        x_min, x_max = np.nanmin(x_valid), np.nanmax(x_valid)
        y_min, y_max = np.nanmin(y_valid), np.nanmax(y_valid)
        return float(x_min), float(x_max), float(y_min), float(y_max)

    @staticmethod
    def _triangle_area2(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @staticmethod
    def _fit_triangle_vertices(x_values, y_values):
        points = np.column_stack(
            (
                np.asarray(x_values, dtype=float).reshape(-1),
                np.asarray(y_values, dtype=float).reshape(-1),
            )
        )
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.shape[0] < 3:
            return GradientScatterDialog._fallback_triangle_vertices(x_values, y_values)

        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] < 3:
            return GradientScatterDialog._fallback_triangle_vertices(x_values, y_values)

        hull_points = unique_points
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(unique_points)
            hull_points = unique_points[hull.vertices]
        except Exception:
            pass

        candidate_points = hull_points
        if candidate_points.shape[0] > 96:
            step = max(1, int(np.ceil(candidate_points.shape[0] / 96.0)))
            candidate_points = candidate_points[::step]
            extrema = np.unique(
                np.vstack(
                    (
                        hull_points[np.argmin(hull_points[:, 0])],
                        hull_points[np.argmax(hull_points[:, 0])],
                        hull_points[np.argmin(hull_points[:, 1])],
                        hull_points[np.argmax(hull_points[:, 1])],
                    )
                ),
                axis=0,
            )
            candidate_points = np.unique(np.vstack((candidate_points, extrema)), axis=0)

        best_vertices = None
        best_area = -1.0
        for idx_a, idx_b, idx_c in combinations(range(candidate_points.shape[0]), 3):
            a = candidate_points[idx_a]
            b = candidate_points[idx_b]
            c = candidate_points[idx_c]
            area2 = GradientScatterDialog._triangle_area2(a, b, c)
            if area2 > best_area:
                best_area = area2
                best_vertices = np.asarray((a, b, c), dtype=float)

        if best_vertices is None or best_area <= 1e-8:
            return GradientScatterDialog._fallback_triangle_vertices(x_values, y_values)

        apex_index = int(np.argmax(best_vertices[:, 1]))
        apex = best_vertices[apex_index]
        base = np.delete(best_vertices, apex_index, axis=0)
        base = base[np.argsort(base[:, 0])]
        return np.asarray((apex, base[0], base[1]), dtype=float)

    @staticmethod
    def _rgb_model(x_values, y_values, color_order="RBG", fit_mode="triangle"):
        order = GradientScatterDialog._normalize_triangular_color_order(color_order)
        mode = GradientScatterDialog._normalize_rgb_fit_mode(fit_mode)
        if mode == "square":
            outline, anchor_points = GradientScatterDialog._fit_square_outline(x_values, y_values)
            vertices = np.asarray(outline, dtype=float)
        else:
            vertices = GradientScatterDialog._fit_triangle_vertices(x_values, y_values)
            anchor_points = np.asarray(vertices, dtype=float)
        rgb_basis = {
            "R": np.array((1.0, 0.0, 0.0), dtype=float),
            "G": np.array((0.0, 1.0, 0.0), dtype=float),
            "B": np.array((0.0, 0.0, 1.0), dtype=float),
        }
        vertex_colors = np.asarray([rgb_basis[channel] for channel in order], dtype=float)
        return {
            "vertices": np.asarray(vertices, dtype=float),
            "anchor_points": np.asarray(anchor_points, dtype=float),
            "vertex_colors": vertex_colors,
            "order": order,
            "fit_mode": mode,
        }

    @staticmethod
    def _normalize_rgb_chroma(values):
        colors = np.asarray(values, dtype=float)
        if colors.ndim != 2 or colors.shape[1] != 3:
            return np.clip(colors, 0.0, 1.0)
        colors = np.clip(colors, 0.0, 1.0)
        scale = np.max(colors, axis=1, keepdims=True)
        scale[scale <= 1e-9] = 1.0
        return np.clip(colors / scale, 0.0, 1.0)

    @staticmethod
    def _rgb_colors_from_model(x_values, y_values, model):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        colors = np.full((x_valid.shape[0], 3), 0.65, dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
        if not np.any(finite_mask):
            return colors

        fit_mode = GradientScatterDialog._normalize_rgb_fit_mode(model.get("fit_mode", "triangle"))
        vertex_colors = np.asarray(model["vertex_colors"], dtype=float)
        if fit_mode == "square":
            anchor_points = np.asarray(model["anchor_points"], dtype=float)
            points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
            deltas = points[:, np.newaxis, :] - anchor_points[np.newaxis, :, :]
            distances = np.sqrt(np.sum(np.square(deltas), axis=2))
            weights = 1.0 / np.maximum(distances, 1e-9)
            close_mask = distances <= 1e-9
            if np.any(close_mask):
                for row_idx in np.flatnonzero(np.any(close_mask, axis=1)).tolist():
                    weights[row_idx, :] = close_mask[row_idx, :].astype(float)
            weight_sum = weights.sum(axis=1, keepdims=True)
            weight_sum[weight_sum <= 0] = 1.0
            weights /= weight_sum
            colors[finite_mask] = GradientScatterDialog._normalize_rgb_chroma(weights @ vertex_colors)
            return np.clip(colors, 0.0, 1.0)

        vertices = np.asarray(model["vertices"], dtype=float)
        v0, v1, v2 = vertices
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if np.isclose(denom, 0.0):
            fallback = GradientScatterDialog._rgb_model(
                x_valid[finite_mask],
                y_valid[finite_mask],
                model.get("order", "RBG"),
                fit_mode="triangle",
            )
            vertices = np.asarray(fallback["vertices"], dtype=float)
            vertex_colors = np.asarray(fallback["vertex_colors"], dtype=float)
            v0, v1, v2 = vertices
            denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if np.isclose(denom, 0.0):
                return colors

        points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
        w0 = (
            (v1[1] - v2[1]) * (points[:, 0] - v2[0])
            + (v2[0] - v1[0]) * (points[:, 1] - v2[1])
        ) / denom
        w1 = (
            (v2[1] - v0[1]) * (points[:, 0] - v2[0])
            + (v0[0] - v2[0]) * (points[:, 1] - v2[1])
        ) / denom
        w2 = 1.0 - w0 - w1
        weights = np.column_stack((w0, w1, w2))
        weights = np.clip(weights, 0.0, 1.0)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum[weight_sum <= 0] = 1.0
        weights /= weight_sum
        colors[finite_mask] = GradientScatterDialog._normalize_rgb_chroma(weights @ vertex_colors)
        return np.clip(colors, 0.0, 1.0)

    @staticmethod
    def _triangular_rgb_model(x_values, y_values, color_order="RBG"):
        return GradientScatterDialog._rgb_model(
            x_values,
            y_values,
            color_order=color_order,
            fit_mode="triangle",
        )

    @staticmethod
    def _triangular_rgb_colors_from_model(x_values, y_values, model):
        return GradientScatterDialog._rgb_colors_from_model(x_values, y_values, model)

    def _render(self):
        self.figure.clear()
        self._point_artist = None
        self._point_artist_entries = []
        x_plot, y_plot = self._rotate_points(self._x, self._y, self._rotation_preset)
        x_label, y_label = self._rotate_axis_labels(
            self._x_label,
            self._y_label,
            self._rotation_preset,
        )
        visible_edge_pairs = self._visible_edge_pairs()
        visible_edge_distances = self._visible_edge_distances()
        subplot_specs = self._display_group_specs()
        if len(subplot_specs) <= 1:
            axes = [self.figure.add_subplot(111)]
        else:
            grid = self.figure.add_gridspec(1, len(subplot_specs), wspace=0.16)
            axes = [self.figure.add_subplot(grid[0, idx]) for idx in range(len(subplot_specs))]
            self.figure.suptitle(self._title, fontsize=13)
        shared_scatter = None

        global_triangle_model = None
        point_colors = None
        if self._use_triangular_rgb:
            global_triangle_model = self._rgb_model(
                x_plot,
                y_plot,
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )
            point_colors = self._rgb_colors_from_model(x_plot, y_plot, global_triangle_model)
            if isinstance(self._project_paths_payload, dict):
                self._project_paths_payload["point_colors"] = np.asarray(point_colors, dtype=float).tolist()
                self._project_paths_payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
                self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
                self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
                self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
        else:
            vmin, vmax = self._compute_display_range(self._color)

        for ax, subplot_spec in zip(axes, subplot_specs):
            local_indices = np.asarray(subplot_spec.get("indices", np.arange(self._x.shape[0], dtype=int)), dtype=int).reshape(-1)
            if local_indices.size == 0:
                ax.set_axis_off()
                continue

            local_pairs, local_distances = self._edge_subset_for_indices(
                visible_edge_pairs,
                visible_edge_distances,
                local_indices,
            )
            self._draw_proximity_overlay(ax, x_plot[local_indices], y_plot[local_indices])
            if self._show_adjacency_edges and local_pairs.size:
                if self._use_edge_bundling:
                    segments = self._bundled_segments_from_pairs(local_pairs)
                else:
                    segments = np.stack(
                        (
                            np.column_stack((x_plot[local_pairs[:, 0]], y_plot[local_pairs[:, 0]])),
                            np.column_stack((x_plot[local_pairs[:, 1]], y_plot[local_pairs[:, 1]])),
                        ),
                        axis=1,
                    )
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=self._edge_color,
                        linewidths=self._edge_linewidth,
                        alpha=self._edge_alpha,
                        zorder=1,
                    )
                )

            if self._use_triangular_rgb:
                if isinstance(self._project_paths_payload, dict):
                    self._draw_triangular_anchor_paths(
                        ax,
                        x_plot,
                        y_plot,
                        point_colors,
                        self._project_paths_payload,
                        group_name=subplot_spec.get("name") if len(subplot_specs) > 1 else None,
                    )
                else:
                    self._draw_active_anchor_markers(
                        ax,
                        x_plot,
                        y_plot,
                        group_name=subplot_spec.get("name") if len(subplot_specs) > 1 else None,
                    )
                scatter = ax.scatter(
                    x_plot[local_indices],
                    y_plot[local_indices],
                    c=np.asarray(point_colors[local_indices], dtype=float),
                    s=38,
                    alpha=0.92,
                    linewidths=0.2,
                    edgecolors="#111827",
                    zorder=2,
                )
                if local_indices.size >= 3:
                    local_model = self._rgb_model(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        self._triangular_color_order,
                        fit_mode=self._rgb_fit_mode,
                    )
                    vertices = np.asarray(local_model["vertices"], dtype=float)
                    outline = np.vstack((vertices, vertices[0]))
                    ax.plot(
                        outline[:, 0],
                        outline[:, 1],
                        linestyle="--",
                        linewidth=1.1,
                        color="#111827",
                        alpha=0.6,
                        zorder=3,
                    )
            else:
                scatter = ax.scatter(
                    x_plot[local_indices],
                    y_plot[local_indices],
                    c=np.asarray(self._color[local_indices], dtype=float),
                    cmap=self._cmap,
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    s=38,
                    alpha=0.9,
                    linewidths=0.2,
                    edgecolors="#111827",
                    zorder=2,
                )
                if shared_scatter is None:
                    shared_scatter = scatter

            if self._point_artist is None:
                self._point_artist = scatter
            annotation = ax.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#6b7280"},
                arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
            )
            annotation.set_visible(False)
            self._point_artist_entries.append(
                {
                    "axes": ax,
                    "artist": scatter,
                    "indices": np.asarray(local_indices, dtype=int),
                    "group": str(subplot_spec.get("name", "all")).strip().lower(),
                    "annotation": annotation,
                }
            )

            title_text = self._title if len(subplot_specs) == 1 else str(subplot_spec.get("title", ""))
            if title_text:
                ax.set_title(title_text, fontsize=13 if len(subplot_specs) == 1 else 11)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(*self._fixed_xlim)
            ax.set_ylim(*self._fixed_ylim)
            ax.set_autoscale_on(False)
            try:
                ax.set_box_aspect(1.0)
            except Exception:
                ax.set_aspect("auto")
            if self._use_triangular_rgb:
                ax.text(
                    0.02,
                    0.98,
                    f"{self._rgb_fit_mode.title()} RGB",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
                if self._show_proximity_circles:
                    ax.text(
                        0.02,
                        0.84,
                        f"Radius: {self._proximity_radius:.4f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )
                ax.text(
                    0.02,
                    0.91,
                    f"Color: {self._triangular_color_order} | Path: {self._path_channel_order}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
                path_summary = self._path_count_summary_text(
                    subplot_spec.get("name") if len(subplot_specs) > 1 else None
                ).replace(" | ", "\n").strip()
                if path_summary:
                    ax.text(
                        0.02,
                        0.77,
                        path_summary,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )

        if not self._use_triangular_rgb and shared_scatter is not None:
            cbar = self.figure.colorbar(shared_scatter, ax=axes)
            cbar.set_label(self._color_label, fontsize=10)
        self._sync_proximity_controls()
        self.info_label.setText(self._info_text())
        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = "gradient_scatter.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient scatter figure",
            str(Path.cwd() / default_name),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            if "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            else:
                output_path = output_path.with_suffix(".png")
        self.figure.savefig(str(output_path), dpi=200, bbox_inches="tight")


class GradientClassificationDialog(QDialog):
    """RGB-classified fsaverage viewer using the selected scatter axes."""

    def __init__(
        self,
        x_volume_img,
        y_volume_img,
        x_values,
        y_values,
        *,
        support_mask_img=None,
        title="Gradient Classification",
        x_label="Gradient 2",
        y_label="Gradient 1",
        parent=None,
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        rotation_preset="Default",
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
    ):
        super().__init__(parent)
        self._x_img = nib.as_closest_canonical(x_volume_img)
        self._y_img = nib.as_closest_canonical(y_volume_img)
        self._support_img = None if support_mask_img is None else nib.as_closest_canonical(support_mask_img)
        self._x_data = np.asarray(self._x_img.get_fdata(), dtype=float)
        self._y_data = np.asarray(self._y_img.get_fdata(), dtype=float)
        if self._x_data.ndim != 3 or self._y_data.ndim != 3:
            raise ValueError("Classification requires 3D projected volumes for the selected axes.")
        if self._x_data.shape != self._y_data.shape:
            raise ValueError("Classification axis volumes must have matching shapes.")
        if self._support_img is not None:
            support_data = np.asarray(self._support_img.get_fdata(), dtype=float)
            if support_data.shape != self._x_data.shape:
                raise ValueError("Classification support mask must match the axis volume shape.")

        self._x_values = np.asarray(x_values, dtype=float).reshape(-1)
        self._y_values = np.asarray(y_values, dtype=float).reshape(-1)
        if self._x_values.shape != self._y_values.shape:
            raise ValueError("Classification axis arrays must have matching lengths.")
        finite_mask = np.isfinite(self._x_values) & np.isfinite(self._y_values)
        if not np.any(finite_mask):
            raise ValueError("Classification requires finite axis values.")
        self._x_values = self._x_values[finite_mask]
        self._y_values = self._y_values[finite_mask]
        scatter_x, scatter_y = GradientScatterDialog._rotate_points(
            self._x_values,
            self._y_values,
            GradientScatterDialog._normalize_rotation_preset(rotation_preset),
        )
        self._rgb_fit_mode = GradientScatterDialog._normalize_rgb_fit_mode(rgb_fit_mode)
        self._triangular_color_order = GradientScatterDialog._normalize_triangular_color_order(
            triangular_color_order
        )
        self._triangular_model = GradientScatterDialog._rgb_model(
            scatter_x,
            scatter_y,
            self._triangular_color_order,
            fit_mode=self._rgb_fit_mode,
        )

        self._title = str(title or "Gradient Classification")
        self._x_label = str(x_label or "Gradient 2")
        self._y_label = str(y_label or "Gradient 1")
        self._theme_name = "Dark"
        self._hemisphere_mode = GradientSurfaceDialog._normalize_hemisphere_mode(hemisphere_mode)
        self._fsaverage_mesh = GradientSurfaceDialog._normalize_fsaverage_mesh(fsaverage_mesh)
        self._rotation_preset = GradientScatterDialog._normalize_rotation_preset(rotation_preset)
        self.setWindowTitle(self._title)

        if self._hemisphere_mode == "both":
            fig_width, fig_height = 13.8, 9.4
        else:
            fig_width, fig_height = 10.6, 5.1
        self.figure = Figure(figsize=(15.5, 8.6), constrained_layout=True)
        self.figure.set_size_inches(fig_width, fig_height, forward=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.info_label = QLabel(
            f"Axes: {self._x_label} / {self._y_label} | Mesh: {self._fsaverage_mesh} | "
            f"Hemisphere: {self._hemisphere_mode.upper()} | Rotation: {self._rotation_preset} | "
            f"{self._rgb_fit_mode.title()} {self._triangular_color_order}"
        )
        self.save_button = QPushButton("Save Figure")
        self.save_button.clicked.connect(self._save_figure)

        controls = QHBoxLayout()
        controls.addWidget(self.info_label, 1)
        controls.addWidget(self.save_button, 0)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.set_theme(theme_name)

        self._render()

    @classmethod
    def from_array(
        cls,
        x_volume_data,
        y_volume_data,
        *,
        affine=None,
        x_values,
        y_values,
        support_mask_data=None,
        title="Gradient Classification",
        x_label="Gradient 2",
        y_label="Gradient 1",
        parent=None,
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        rotation_preset="Default",
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
    ):
        x_arr = np.asarray(x_volume_data, dtype=float)
        y_arr = np.asarray(y_volume_data, dtype=float)
        if x_arr.ndim != 3 or y_arr.ndim != 3:
            raise ValueError(
                f"Expected 3D arrays for classification axes. Got shapes {x_arr.shape} and {y_arr.shape}."
            )
        if x_arr.shape != y_arr.shape:
            raise ValueError("Classification axis arrays must have matching shapes.")
        if affine is None:
            affine = np.eye(4)
        x_img = nib.Nifti1Image(x_arr, affine)
        y_img = nib.Nifti1Image(y_arr, affine)
        support_img = None
        if support_mask_data is not None:
            support_arr = np.asarray(support_mask_data, dtype=float)
            if support_arr.shape != x_arr.shape:
                raise ValueError("Classification support mask must match the axis volume shape.")
            support_img = nib.Nifti1Image(support_arr, affine)
        return cls(
            x_img,
            y_img,
            x_values,
            y_values,
            support_mask_img=support_img,
            title=title,
            x_label=x_label,
            y_label=y_label,
            parent=parent,
            theme_name=theme_name,
            hemisphere_mode=hemisphere_mode,
            fsaverage_mesh=fsaverage_mesh,
            rotation_preset=rotation_preset,
            rgb_fit_mode=rgb_fit_mode,
            triangular_color_order=triangular_color_order,
        )

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    def _surface_views_layout(self):
        assets = GradientSurfaceDialog._get_surface_assets(self._fsaverage_mesh)
        if self._hemisphere_mode == "lh":
            return [
                [
                    ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                    ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
                ]
            ]
        if self._hemisphere_mode == "rh":
            return [
                [
                    ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                    ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
                ]
            ]
        return [
            [
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
            ],
            [
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            ],
        ]

    @staticmethod
    def _mesh_arrays(mesh):
        if hasattr(mesh, "coordinates") and hasattr(mesh, "faces"):
            return np.asarray(mesh.coordinates, dtype=float), np.asarray(mesh.faces, dtype=int)
        if isinstance(mesh, (tuple, list)) and len(mesh) >= 2:
            return np.asarray(mesh[0], dtype=float), np.asarray(mesh[1], dtype=int)
        raise ValueError("Unsupported surface mesh format.")

    @staticmethod
    def _view_angles(hemi, view):
        mapping = {
            ("left", "lateral"): (0.0, 180.0),
            ("left", "medial"): (0.0, 0.0),
            ("right", "lateral"): (0.0, 0.0),
            ("right", "medial"): (0.0, 180.0),
        }
        return mapping.get((hemi, view), (0.0, 180.0))

    @staticmethod
    def _background_face_gray(bg_map, faces):
        n_faces = int(np.asarray(faces).shape[0])
        if bg_map is None:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg = np.asarray(bg_map, dtype=float).reshape(-1)
        if bg.size < np.max(faces) + 1:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg_finite = bg[np.isfinite(bg)]
        if bg_finite.size == 0:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg_min = float(np.nanmin(bg_finite))
        bg_max = float(np.nanmax(bg_finite))
        if np.isclose(bg_min, bg_max):
            bg_norm = np.full(bg.shape, 0.72, dtype=float)
        else:
            bg_norm = 0.55 + 0.35 * ((bg - bg_min) / (bg_max - bg_min))
        face_gray = np.asarray(bg_norm[faces].mean(axis=1), dtype=float)
        return np.repeat(face_gray[:, np.newaxis], 3, axis=1)

    def _plot_rgb_surface(self, ax, mesh, vertex_colors, vertex_alpha, bg_map, *, hemi, view, title):
        coords, faces = self._mesh_arrays(mesh)
        bg_face_rgb = self._background_face_gray(bg_map, faces)

        base = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=faces,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        base_rgba = np.concatenate((bg_face_rgb, np.ones((bg_face_rgb.shape[0], 1), dtype=float)), axis=1)
        base.set_facecolors(base_rgba)
        base.set_edgecolors("none")

        overlay = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=faces,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        face_rgb = np.asarray(vertex_colors[faces].mean(axis=1), dtype=float)
        face_rgb = np.clip(0.92 * face_rgb + 0.08 * bg_face_rgb, 0.0, 1.0)
        face_alpha = np.asarray(vertex_alpha[faces].mean(axis=1), dtype=float)
        face_alpha = np.clip(face_alpha, 0.0, 1.0)
        overlay_rgba = np.concatenate((face_rgb, face_alpha[:, np.newaxis]), axis=1)
        overlay.set_facecolors(overlay_rgba)
        overlay.set_edgecolors("none")
        elev, azim = self._view_angles(hemi, view)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pad = 0.04 * span
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        try:
            ax.set_box_aspect(tuple(span.tolist()))
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, pad=4)

    def _surface_vertex_colors(self, x_img, y_img, mesh):
        x_texture = surface.vol_to_surf(
            x_img,
            mesh,
            radius=0.5,
            interpolation="linear",
        )
        y_texture = surface.vol_to_surf(
            y_img,
            mesh,
            radius=0.5,
            interpolation="linear",
        )
        if self._support_img is not None:
            support_texture = surface.vol_to_surf(
                self._support_img,
                mesh,
                radius=0.5,
                interpolation="nearest",
            )
            vertex_alpha = np.clip(np.asarray(support_texture, dtype=float), 0.0, 1.0)
        else:
            vertex_alpha = np.asarray(np.isfinite(x_texture) & np.isfinite(y_texture), dtype=float)
        x_rot, y_rot = GradientScatterDialog._rotate_points(
            np.asarray(x_texture, dtype=float),
            np.asarray(y_texture, dtype=float),
            self._rotation_preset,
        )
        vertex_colors = GradientScatterDialog._rgb_colors_from_model(
            x_rot,
            y_rot,
            self._triangular_model,
        )
        return vertex_colors, vertex_alpha

    def _render(self):
        self.figure.clear()
        view_rows = self._surface_views_layout()
        n_rows = len(view_rows)
        n_cols = max(len(row) for row in view_rows)
        gs = self.figure.add_gridspec(n_rows, n_cols, wspace=0.02, hspace=0.08)

        for row_idx, row_views in enumerate(view_rows):
            for col_idx, (hemi, view, mesh, bg_map, title) in enumerate(row_views):
                ax = self.figure.add_subplot(gs[row_idx, col_idx], projection="3d")
                vertex_colors, vertex_alpha = self._surface_vertex_colors(self._x_img, self._y_img, mesh)
                self._plot_rgb_surface(
                    ax,
                    mesh,
                    vertex_colors,
                    vertex_alpha,
                    bg_map,
                    hemi=hemi,
                    view=view,
                    title=title,
                )

        self.figure.suptitle(self._title, fontsize=15)
        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = "gradient_classification.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient classification figure",
            str(Path.cwd() / default_name),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            if "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            else:
                output_path = output_path.with_suffix(".png")
        self.figure.savefig(str(output_path), dpi=200, bbox_inches="tight")


MSModeSurfaceDialog = GradientSurfaceDialog


__all__ = [
    "GradientSurfaceDialog",
    "GradientScatterDialog",
    "GradientClassificationDialog",
    "MSModeSurfaceDialog",
]
