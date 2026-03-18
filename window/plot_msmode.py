#!/usr/bin/env python3
"""Nilearn surface rendering dialogs for gradient volume maps.

This module is designed for GUI integration (e.g., connectome_viewer).
It renders each component on one row with four views:
left-lateral, left-medial, right-medial, right-lateral,
and a per-component colorbar on the right.
"""

import heapq
import json
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.cm import ScalarMappable
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

from nilearn import datasets, plotting, surface

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

def _dialog_theme_stylesheet(theme_name="Dark"):
    theme = str(theme_name or "Dark").strip().title()
    if theme not in {"Light", "Dark", "Teya", "Donald"}:
        theme = "Dark"
    if theme == "Dark":
        return theme, (
            "QDialog, QWidget { background: #1f2430; color: #e5e7eb; }"
            "QPushButton { background: #2d3646; color: #e5e7eb; border: 1px solid #5f6d82; border-radius: 6px; padding: 5px 10px; }"
            "QPushButton:hover { background: #374256; }"
        )
    if theme == "Teya":
        return theme, (
            "QDialog, QWidget { background: #ffd0e5; color: #0b7f7a; }"
            "QPushButton { background: #ffc0dc; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 6px; padding: 5px 10px; }"
            "QPushButton:hover { background: #ffb1d5; }"
        )
    if theme == "Donald":
        return theme, (
            "QDialog, QWidget { background: #d97706; color: #ffffff; }"
            "QPushButton { background: #b85f00; color: #ffffff; border: 1px solid #f3a451; border-radius: 6px; padding: 5px 10px; }"
            "QPushButton:hover { background: #c76b06; }"
        )
    return theme, (
        "QDialog, QWidget { background: #f4f6f9; color: #1f2937; }"
        "QPushButton { background: #ffffff; color: #1f2937; border: 1px solid #b7c0cc; border-radius: 6px; padding: 5px 10px; }"
        "QPushButton:hover { background: #edf2f7; }"
    )


class GradientSurfaceDialog(QDialog):
    """Interactive Nilearn surface viewer for 3D/4D gradient maps."""
    _surface_assets_cache = None

    def __init__(
        self,
        volume_img,
        title="Gradient Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
    ):
        super().__init__(parent)
        self._img = nib.as_closest_canonical(volume_img)
        self._data = np.asarray(self._img.get_fdata(), dtype=float)
        if self._data.ndim == 3:
            self._data = self._data[..., np.newaxis]
        if self._data.ndim != 4:
            raise ValueError(f"Expected 3D or 4D volume data. Got shape {self._data.shape}.")

        self._n_components = self._data.shape[3]
        self._title = title
        self._cmap_name = str(cmap_name or "spectrum_fsl")
        self._cmap = cmap if cmap is not None else self._default_cmap(self._cmap_name)
        self._theme_name = "Dark"
        self._hemisphere_mode = self._normalize_hemisphere_mode(hemisphere_mode)
        self._fsaverage_mesh = self._normalize_fsaverage_mesh(fsaverage_mesh)
        self.setWindowTitle(title)

        self.figure = Figure(
            figsize=(20, max(3.2 * self._n_components, 4.0)),
            constrained_layout=True,
        )
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.info_label = QLabel(
            "Views per component: "
            + self._hemisphere_label_text()
            + f" | Mesh: {self._fsaverage_mesh}"
            + f" | Cmap: {self._cmap_name}"
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

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    @staticmethod
    def _default_cmap(cmap_name):
        try:
            try:
                from graphplot.colorbar import ColorBar
            except Exception:
                from mrsitoolbox.graphplot.colorbar import ColorBar

            colorbar = ColorBar()
            if cmap_name == "spectrum_fsl":
                try:
                    return colorbar.load_fsl_cmap(cmap_name)
                except Exception:
                    return colorbar.bars(cmap_name)
            try:
                return colorbar.load_fsl_cmap(cmap_name)
            except Exception:
                return colorbar.bars(cmap_name)
        except Exception:
            return "viridis"

    @classmethod
    def from_array(
        cls,
        volume_data,
        affine=None,
        title="Gradient Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
    ):
        """Create dialog from ndarray + affine."""
        arr = np.asarray(volume_data, dtype=float)
        if arr.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D array. Got shape {arr.shape}.")
        if affine is None:
            affine = np.eye(4)
        img = nib.Nifti1Image(arr, affine)
        return cls(
            img,
            title=title,
            parent=parent,
            cmap=cmap,
            cmap_name=cmap_name,
            theme_name=theme_name,
            hemisphere_mode=hemisphere_mode,
            fsaverage_mesh=fsaverage_mesh,
        )

    @staticmethod
    def _normalize_hemisphere_mode(value):
        text = str(value or "both").strip().lower()
        if text not in {"both", "lh", "rh"}:
            text = "both"
        return text

    @staticmethod
    def _normalize_fsaverage_mesh(value):
        text = str(value or "fsaverage4").strip().lower()
        valid = {"fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"}
        if text not in valid:
            text = "fsaverage4"
        return text

    def _hemisphere_label_text(self):
        if self._hemisphere_mode == "lh":
            return "LH lateral | LH medial"
        if self._hemisphere_mode == "rh":
            return "RH medial | RH lateral"
        return "LH lateral | LH medial | RH medial | RH lateral"

    @classmethod
    def _get_surface_assets(cls, fsaverage_mesh="fsaverage4"):
        """Load and cache fsaverage meshes/background maps once."""
        mesh_name = cls._normalize_fsaverage_mesh(fsaverage_mesh)
        if cls._surface_assets_cache is None:
            cls._surface_assets_cache = {}
        if mesh_name in cls._surface_assets_cache:
            return cls._surface_assets_cache[mesh_name]

        try:
            fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh_name)
        except Exception:
            try:
                fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage4")
                mesh_name = "fsaverage4"
            except Exception:
                fsaverage = datasets.fetch_surf_fsaverage()
                mesh_name = "fsaverage4"

        mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
        mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
        sulc_left = surface.load_surf_data(fsaverage.sulc_left)
        sulc_right = surface.load_surf_data(fsaverage.sulc_right)

        cls._surface_assets_cache[mesh_name] = {
            "mesh_left": mesh_left,
            "mesh_right": mesh_right,
            "sulc_left": sulc_left,
            "sulc_right": sulc_right,
        }
        return cls._surface_assets_cache[mesh_name]

    @staticmethod
    def _compute_display_range(data_4d):
        finite = data_4d[np.isfinite(data_4d)]
        if finite.size == 0:
            return -1.0, 1.0, None
        nonzero = finite[finite != 0]
        scale_data = nonzero if nonzero.size else finite
        vmin = float(np.percentile(scale_data, 2))
        vmax = float(np.percentile(scale_data, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin = float(np.nanmin(scale_data))
            vmax = float(np.nanmax(scale_data))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        threshold = None
        if nonzero.size:
            thr = float(np.percentile(np.abs(nonzero), 2))
            if np.isfinite(thr) and thr > 0:
                threshold = thr
        return vmin, vmax, threshold

    def _render(self):
        self.figure.clear()
        assets = self._get_surface_assets(self._fsaverage_mesh)

        vmin, vmax, threshold = self._compute_display_range(self._data)
        row_norm = Normalize(vmin=vmin, vmax=vmax)

        if self._hemisphere_mode == "lh":
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
            )
        elif self._hemisphere_mode == "rh":
            views = (
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            )
        else:
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            )
        n_view_cols = len(views)
        gs = self.figure.add_gridspec(
            self._n_components,
            n_view_cols + 1,
            width_ratios=[1.0] * n_view_cols + [0.05],
            wspace=0.02,
            hspace=0.08,
        )

        row_axes_all = []
        for comp_idx in range(self._n_components):
            comp_img = nib.Nifti1Image(self._data[..., comp_idx], self._img.affine)
            texture_left = surface.vol_to_surf(
                comp_img,
                assets["mesh_left"],
                radius=0.5,
                interpolation="linear",
            )
            texture_right = surface.vol_to_surf(
                comp_img,
                assets["mesh_right"],
                radius=0.5,
                interpolation="linear",
            )
            textures = {"left": texture_left, "right": texture_right}
            row_axes = []

            for view_idx, (hemi, view, mesh, bg_map, view_label) in enumerate(views):
                ax = self.figure.add_subplot(gs[comp_idx, view_idx], projection="3d")
                plotting.plot_surf_stat_map(
                    mesh,
                    textures[hemi],
                    hemi=hemi,
                    view=view,
                    bg_map=bg_map,
                    cmap=self._cmap,
                    colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                    threshold=threshold,
                    axes=ax,
                )
                ax.set_title(view_label, fontsize=9, pad=2)
                row_axes.append(ax)
            row_axes_all.append(row_axes)

            cax = self.figure.add_subplot(gs[comp_idx, n_view_cols])
            mappable = ScalarMappable(norm=row_norm, cmap=self._cmap)
            mappable.set_array([])
            cbar = self.figure.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=8, length=2)
            cbar.set_label("Gradient", fontsize=10)

        # Center component titles per row at a safe height in the gap above each row.
        prev_row_bottom = 0.99
        for comp_idx, row_axes in enumerate(row_axes_all):
            left_pos = row_axes[0].get_position()
            right_pos = row_axes[-1].get_position()
            center_x = (left_pos.x0 + right_pos.x1) / 2.0
            row_top = max(ax.get_position().y1 for ax in row_axes)
            row_bottom = min(ax.get_position().y0 for ax in row_axes)
            gap_above = max(0.0, prev_row_bottom - row_top)
            y_text = row_top + (0.28 * gap_above if gap_above > 0 else 0.006)
            y_text = min(y_text, 0.972)
            self.figure.text(
                center_x,
                y_text,
                f"Gradient {comp_idx + 1}",
                ha="center",
                va="bottom",
                fontsize=16,
            )
            prev_row_bottom = row_bottom

        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = f"gradient_components_{self._n_components}.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient surface figure",
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
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        rotation_preset="Default",
        use_triangular_rgb=False,
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
        edge_pairs=None,
        edge_color="#111827",
        edge_alpha=0.16,
        edge_linewidth=0.45,
        point_group_codes=None,
        project_paths_callback=None,
    ):
        super().__init__(parent)
        self._x = np.asarray(x_values, dtype=float).reshape(-1)
        self._y = np.asarray(y_values, dtype=float).reshape(-1)
        if self._x.shape != self._y.shape:
            raise ValueError("Gradient scatter axes must have matching lengths.")
        color_data = self._y if color_values is None else np.asarray(color_values, dtype=float).reshape(-1)
        if color_data.shape != self._x.shape:
            raise ValueError("Gradient scatter color data must match the axis lengths.")
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
        finite_mask = np.isfinite(self._x) & np.isfinite(self._y) & np.isfinite(color_data)
        if not np.any(finite_mask):
            raise ValueError("Gradient scatter requires finite data points.")
        self._x = self._x[finite_mask]
        self._y = self._y[finite_mask]
        self._color = color_data[finite_mask]
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
        self._project_paths_payload = None
        self._show_proximity_circles = False
        self._show_all_ordered_paths = False
        self._use_directionality_filter = False
        display_x, display_y = self._rotate_points(self._x, self._y, self._rotation_preset)
        self._display_coords = np.column_stack((display_x, display_y))
        self._proximity_max_radius = self._compute_max_radius(self._display_coords)
        self._proximity_slider_steps = 1000
        self._proximity_radius = 0.0
        self._edge_distances = self._compute_edge_distances(self._display_coords, self._edge_pairs)
        self._fixed_xlim, self._fixed_ylim = self._compute_fixed_axes(self._display_coords)
        self._point_artist = None
        self._hover_annotation = None
        self._hover_axes = None
        self._hover_cid = None
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
        self.proximity_check.toggled.connect(self._on_proximity_toggled)
        proximity_controls.addWidget(self.proximity_check, 0)
        slider_orientation = Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal
        self.proximity_slider = QSlider(slider_orientation)
        self.proximity_slider.setRange(0, self._proximity_slider_steps)
        self.proximity_slider.setValue(0)
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
        self.path_width_scale_spin.setRange(0.05, 200.0)
        self.path_width_scale_spin.setSingleStep(0.5)
        self.path_width_scale_spin.setValue(float(self._path_width_scaling_strength))
        self.path_width_scale_spin.valueChanged.connect(self._on_path_width_scale_changed)
        proximity_controls.addWidget(self.path_width_scale_spin, 0)

        path_controls = QHBoxLayout()
        self.all_paths_check = QCheckBox("All ordered paths")
        self.all_paths_check.toggled.connect(self._on_all_paths_toggled)
        path_controls.addWidget(self.all_paths_check, 0)
        self.direction_filter_check = QCheckBox("Directional filter")
        self.direction_filter_check.toggled.connect(self._on_direction_filter_toggled)
        path_controls.addWidget(self.direction_filter_check, 0)
        path_controls.addWidget(QLabel("lambda"), 0)
        self.free_energy_lambda_spin = QDoubleSpinBox()
        self.free_energy_lambda_spin.setDecimals(3)
        self.free_energy_lambda_spin.setRange(0.001, 1000.0)
        self.free_energy_lambda_spin.setSingleStep(0.1)
        self.free_energy_lambda_spin.setValue(1.0)
        path_controls.addWidget(self.free_energy_lambda_spin, 0)
        self.generate_paths_button = QPushButton("Generate paths")
        self.generate_paths_button.clicked.connect(self._on_generate_paths_clicked)
        path_controls.addWidget(self.generate_paths_button, 0)
        self.compute_free_energy_button = QPushButton("Compute free energy")
        self.compute_free_energy_button.setEnabled(False)
        self.compute_free_energy_button.clicked.connect(self._on_compute_free_energy_clicked)
        path_controls.addWidget(self.compute_free_energy_button, 0)
        self.project_paths_button = QPushButton("Project to 3D brain")
        self.project_paths_button.setEnabled(False)
        self.project_paths_button.clicked.connect(self._on_project_paths_clicked)
        path_controls.addWidget(self.project_paths_button, 0)
        self.export_paths_button = QPushButton("Export paths")
        self.export_paths_button.setEnabled(False)
        self.export_paths_button.clicked.connect(self._on_export_paths_clicked)
        path_controls.addWidget(self.export_paths_button, 0)
        path_controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.toolbar)
        layout.addLayout(proximity_controls)
        layout.addLayout(path_controls)
        layout.addWidget(self.canvas, 1)
        self.set_theme(theme_name)
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
            if self._use_triangular_rgb and self._use_directionality_filter:
                edge_text += " | Dir filter"
        path_text = self._path_count_summary_text()
        return f"Points: {self._x.size}{edge_text}{path_text} | Rotation: {self._rotation_preset} | {mode}"

    @staticmethod
    def _normalize_rotation_preset(value):
        text = str(value or "Default").strip()
        valid = {"Default", "+90", "-90", "180"}
        if text not in valid:
            text = "Default"
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
        self.all_paths_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.direction_filter_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.generate_paths_button.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
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
        self.proximity_value_label.setText(self._proximity_label_text())

    def _path_count_summary_text(self):
        if not self._use_triangular_rgb or not isinstance(self._project_paths_payload, dict):
            return ""
        group_payloads = list(self._project_paths_payload.get("group_paths", []))
        if not group_payloads:
            return ""
        parts = []
        for group_payload in group_payloads:
            group_name = str(group_payload.get("group", "all")).strip().lower()
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

    def _hide_hover_annotation(self):
        if self._hover_annotation is not None and self._hover_annotation.get_visible():
            self._hover_annotation.set_visible(False)
            self.canvas.draw_idle()

    def _on_hover(self, event):
        if (
            self._point_artist is None
            or self._hover_annotation is None
            or self._hover_axes is None
            or event.inaxes != self._hover_axes
        ):
            self._hide_hover_annotation()
            return
        contains, details = self._point_artist.contains(event)
        if not contains:
            self._hide_hover_annotation()
            return
        indices = np.asarray(details.get("ind", []), dtype=int).reshape(-1)
        if indices.size == 0:
            self._hide_hover_annotation()
            return
        index = int(indices[0])
        offsets = np.asarray(self._point_artist.get_offsets(), dtype=float)
        if index < 0 or index >= offsets.shape[0]:
            self._hide_hover_annotation()
            return
        x_coord, y_coord = offsets[index]
        label = str(self._point_labels[index]) if index < self._point_labels.shape[0] else f"Point {index + 1}"
        self._hover_annotation.xy = (float(x_coord), float(y_coord))
        self._hover_annotation.set_text(label)
        self._hover_annotation.set_visible(True)
        self.canvas.draw_idle()

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
        return np.asarray(self._edge_distances[visible], dtype=float)

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

    def _ctx_segment_records_for_full_path(self, path_nodes, anchors, channel_order):
        nodes = [int(node) for node in list(path_nodes or [])]
        anchor_map = {str(key): int(value) for key, value in dict(anchors or {}).items()}
        order = [str(channel) for channel in list(channel_order or []) if str(channel) in anchor_map]
        if len(nodes) < 2 or len(order) < 2:
            return []

        def _segment_record(first, second, segment_nodes):
            color = np.clip(
                0.5 * (self._rgb_basis_color(first) + self._rgb_basis_color(second)),
                0.0,
                1.0,
            )
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
        point_colors,
        visible_edge_pairs,
        visible_edge_distances,
    ):
        if visible_edge_pairs.shape[0] == 0:
            return None
        scatter_coords = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        rgb_basis = {
            "R": np.array((1.0, 0.0, 0.0), dtype=float),
            "G": np.array((0.0, 1.0, 0.0), dtype=float),
            "B": np.array((0.0, 0.0, 1.0), dtype=float),
        }
        project_group_paths = []
        valid_optimal_paths = []

        for group_spec in self._path_group_specs():
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                continue
            candidate_indices = np.flatnonzero(eligible_mask)
            anchors = self._rgb_anchor_indices(
                x_plot,
                y_plot,
                triangle_model,
                candidate_indices=candidate_indices,
            )
            if not {"R", "G", "B"}.issubset(set(anchors.keys())):
                continue
            forbidden_nodes = set(np.flatnonzero(~eligible_mask).tolist())
            group_pair_paths = []
            pair_paths = self._ordered_anchor_pair_paths(
                self._x.size,
                visible_edge_pairs,
                visible_edge_distances,
                triangle_model.get("order", "RBG"),
                anchors,
                forbidden_nodes=forbidden_nodes,
            )
            valid_pair_records = []
            for first, second, paths in pair_paths:
                color = tuple(np.clip(0.5 * (rgb_basis[first] + rgb_basis[second]), 0.0, 1.0).tolist())
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

            order_channels = [str(channel) for channel in list(triangle_model.get("order", "RBG"))]
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
                triangle_model.get("order", "RBG"),
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

            optimal_full_path = self._combine_ordered_segments(valid_ordered_segments)
            group_payload = {
                "group": str(group_spec.get("name", "all")),
                "anchors": {str(key): int(value) for key, value in anchors.items()},
                "optimal_segments": [
                    self._path_record(
                        first,
                        second,
                        path_nodes,
                        tuple(np.clip(0.5 * (rgb_basis[first] + rgb_basis[second]), 0.0, 1.0).tolist()),
                    )
                    for first, second, path_nodes in valid_ordered_segments
                ],
                "optimal_full_path": [int(node) for node in optimal_full_path] if len(optimal_full_path) >= 2 else [],
                "all_full_paths": self._combine_ordered_path_records(valid_pair_records, max_full_paths=256),
                "full_path_count": 0,
                "ctx_path_count": 0,
                "all_pair_paths": group_pair_paths,
                "subc_anchor": int(subc_anchor_index) if subc_anchor_index is not None else None,
                "subc_paths": [list(path) for path in subc_paths],
                "subc_optimal_path": [int(node) for node in subc_optimal_path] if len(subc_optimal_path) >= 2 else [],
                "subc_color": [float(value) for value in np.asarray(subc_color, dtype=float).tolist()] if subc_color is not None else [],
                "subc_path_count": int(len(subc_paths)),
            }
            group_payload["full_path_count"] = int(len(group_payload["all_full_paths"]))
            group_payload["ctx_path_count"] = int(group_payload["full_path_count"])
            if group_payload["optimal_full_path"]:
                valid_optimal_paths.append(group_payload["optimal_full_path"])
            project_group_paths.append(group_payload)

        if not project_group_paths:
            return None
        return {
            "channel_order": str(triangle_model.get("order", "RBG")),
            "fit_mode": str(triangle_model.get("fit_mode", self._rgb_fit_mode)),
            "group_paths": project_group_paths,
            "optimal_full_path": list(valid_optimal_paths[0]) if valid_optimal_paths else [],
            "show_all_ordered_paths": bool(self._show_all_ordered_paths),
            "rotation_preset": str(self._rotation_preset),
            "radius": float(self._proximity_radius),
        }

    def _draw_triangular_anchor_paths(self, ax, x_plot, y_plot, point_colors, path_payload):
        if not isinstance(path_payload, dict):
            return
        group_payloads = [dict(group_payload) for group_payload in list(path_payload.get("group_paths", []))]
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
                        segments = self._path_segments(x_plot, y_plot, record.get("nodes", []))
                        color = tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist())
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
                    segments = self._path_segments(x_plot, y_plot, nodes)
                    for segment in segments:
                        all_segments.append(segment)
                        all_colors.append(tuple(subc_color.tolist()))
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
                segments = self._path_segments(x_plot, y_plot, record.get("nodes", []))
                color = tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist())
                for segment in segments:
                    highlighted_segments.append(segment)
                    highlighted_colors.append(color)
                    highlighted_widths.append(optimal_ctx_width)

            subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
            if subc_color.shape != (3,):
                subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
            subc_segments = self._path_segments(x_plot, y_plot, list(group_payload.get("subc_optimal_path", [])))
            for segment in subc_segments:
                highlighted_segments.append(segment)
                highlighted_colors.append(tuple(subc_color.tolist()))
                highlighted_widths.append(
                    self._path_display_width(
                        self._edge_linewidth,
                        group_payload.get("subc_optimal_path_energy"),
                        subc_scaling,
                        mode="scatter",
                        scaling_mode=width_mode,
                        scaling_strength=width_strength,
                    )
                )

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

    def _on_direction_filter_toggled(self, checked):
        self._use_directionality_filter = bool(checked)
        self._invalidate_generated_paths()
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
    def _path_directionality_energy(coords, path_nodes, reference_unit_vector):
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
        penalties = 1.0 - alignment
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

    def _ctx_full_path_energy(self, coords, full_path_nodes, anchors, channel_order):
        order = [str(channel) for channel in list(channel_order or []) if str(channel) in anchors]
        if len(order) < 2:
            return None
        if len(order) == 2:
            ref_unit = self._reference_unit_vector(anchors[order[0]], anchors[order[1]])
            if ref_unit is None:
                return None
            return self._path_directionality_energy(coords, full_path_nodes, ref_unit)

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
        energy_first = self._path_directionality_energy(coords, first_segment, ref_first)
        energy_second = self._path_directionality_energy(coords, second_segment, ref_second)
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
                        energy = self._path_directionality_energy(coords, path_nodes, ref_unit)
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
            colors[finite_mask] = weights @ vertex_colors
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
        colors[finite_mask] = weights @ vertex_colors
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
        ax = self.figure.add_subplot(111)
        self._hover_axes = ax
        self._point_artist = None
        x_plot, y_plot = self._rotate_points(self._x, self._y, self._rotation_preset)
        x_label, y_label = self._rotate_axis_labels(
            self._x_label,
            self._y_label,
            self._rotation_preset,
        )
        self._draw_proximity_overlay(ax, x_plot, y_plot)
        visible_edge_pairs = self._visible_edge_pairs()
        visible_edge_distances = self._visible_edge_distances()
        if visible_edge_pairs.size:
            segments = np.stack(
                (
                    np.column_stack((x_plot[visible_edge_pairs[:, 0]], y_plot[visible_edge_pairs[:, 0]])),
                    np.column_stack((x_plot[visible_edge_pairs[:, 1]], y_plot[visible_edge_pairs[:, 1]])),
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
            triangle_model = self._rgb_model(
                x_plot,
                y_plot,
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )
            point_colors = self._rgb_colors_from_model(x_plot, y_plot, triangle_model)
            if isinstance(self._project_paths_payload, dict):
                self._project_paths_payload["point_colors"] = np.asarray(point_colors, dtype=float).tolist()
                self._project_paths_payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
                self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
                self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
                self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
                self._draw_triangular_anchor_paths(
                    ax,
                    x_plot,
                    y_plot,
                    point_colors,
                    self._project_paths_payload,
                )
            scatter = ax.scatter(
                x_plot,
                y_plot,
                c=point_colors,
                s=38,
                alpha=0.92,
                linewidths=0.2,
                edgecolors="#111827",
                zorder=2,
            )
            vertices = np.asarray(triangle_model["vertices"], dtype=float)
            if triangle_model.get("fit_mode", "triangle") == "square":
                outline = np.vstack((vertices, vertices[0]))
            else:
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
            anchor_points = np.asarray(triangle_model.get("anchor_points", vertices[:3]), dtype=float)
            for vertex, channel in zip(anchor_points, triangle_model["order"]):
                ax.text(
                    float(vertex[0]),
                    float(vertex[1]),
                    channel,
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
                    zorder=4,
                )
        else:
            vmin, vmax = self._compute_display_range(self._color)
            scatter = ax.scatter(
                x_plot,
                y_plot,
                c=self._color,
                cmap=self._cmap,
                norm=Normalize(vmin=vmin, vmax=vmax),
                s=38,
                alpha=0.9,
                linewidths=0.2,
                edgecolors="#111827",
                zorder=2,
            )
        self._point_artist = scatter
        ax.set_title(self._title, fontsize=13)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(*self._fixed_xlim)
        ax.set_ylim(*self._fixed_ylim)
        ax.set_autoscale_on(False)
        ax.set_aspect("equal", adjustable="box")
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
                f"Order: {self._triangular_color_order}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            path_summary = self._path_count_summary_text().replace(" | ", "\n").strip()
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
        else:
            cbar = self.figure.colorbar(scatter, ax=ax)
            cbar.set_label(self._color_label, fontsize=10)
        self._hover_annotation = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#6b7280"},
            arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
        )
        self._hover_annotation.set_visible(False)
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


class GradientFreeEnergyDialog(QDialog):
    """Order-plot viewer for path directionality energies and free energies."""

    def __init__(self, payload, *, parent=None, theme_name="Dark"):
        super().__init__(parent)
        self._payload = dict(payload or {})
        self._theme_name = "Dark"
        self.setWindowTitle(f"Free Energy - {str(self._payload.get('title', 'Gradients'))}")

        self.figure = Figure(figsize=(11.2, 6.8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lambda_value = float(self._payload.get("lambda", 1.0))
        axis_text = f"{str(self._payload.get('y_axis_label', 'Y'))} vs {str(self._payload.get('x_axis_label', 'X'))}"
        self.info_label = QLabel(
            f"lambda = {lambda_value:.3f} | Energy = sum(1 - step_unit dot ref_unit) | {axis_text}"
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

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    def _render(self):
        self.figure.clear()
        groups = [dict(group) for group in list(self._payload.get("groups", []))]
        if not groups:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No free-energy paths available.", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        n_rows = len(groups)
        n_cols = max(1, max(len(list(dict(group).get("families", []))) for group in groups))
        axes = np.asarray(self.figure.subplots(n_rows, n_cols, squeeze=False), dtype=object)
        title_text = str(self._payload.get("title", "Gradients"))
        rotation_text = str(self._payload.get("rotation", "Default"))
        self.figure.suptitle(f"Free Energy Order Plot - {title_text} | Rotation: {rotation_text}", fontsize=12)

        for row_idx, group in enumerate(groups):
            families = [dict(family) for family in list(group.get("families", []))]
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(families):
                    ax.axis("off")
                    continue
                family = families[col_idx]
                energies = np.sort(np.asarray(family.get("energies", []), dtype=float).reshape(-1))
                if energies.size == 0:
                    ax.text(0.5, 0.5, "No paths", ha="center", va="center")
                    ax.axis("off")
                    continue
                x_order = np.arange(1, energies.size + 1, dtype=int)
                color = np.asarray(family.get("color", (0.2, 0.2, 0.2)), dtype=float).reshape(-1)
                if color.shape != (3,):
                    color = np.asarray((0.2, 0.2, 0.2), dtype=float)
                free_energy = float(family.get("free_energy", float("nan")))
                ax.plot(
                    x_order,
                    energies,
                    color=tuple(color.tolist()),
                    marker="o",
                    markersize=3.8,
                    linewidth=1.2,
                    alpha=0.95,
                )
                if np.isfinite(free_energy):
                    ax.axhline(
                        free_energy,
                        color=tuple(color.tolist()),
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.7,
                    )
                ax.set_title(
                    f"{str(group.get('group', 'all')).upper()} | {str(family.get('label', 'path'))}\n"
                    f"F={free_energy:.4f} | n={int(family.get('n_paths', energies.size))}",
                    fontsize=10,
                )
                ax.set_xlabel("Ordered path rank")
                ax.set_ylabel("Energy")
                ax.grid(True, alpha=0.25)

        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = "gradient_free_energy.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save free-energy figure",
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
            fig_width, fig_height = 19.2, 5.4
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
        assets = GradientSurfaceDialog._get_surface_assets(self._fsaverage_mesh)

        if self._hemisphere_mode == "lh":
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
            )
        elif self._hemisphere_mode == "rh":
            views = (
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            )
        else:
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            )

        gs = self.figure.add_gridspec(1, len(views), wspace=0.02)
        brain_axes = [self.figure.add_subplot(gs[0, idx], projection="3d") for idx in range(len(views))]

        for ax, (hemi, view, mesh, bg_map, title) in zip(brain_axes, views):
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
