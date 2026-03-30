#!/usr/bin/env python3
"""Shared dialog theme helpers."""

from __future__ import annotations

_VALID_THEMES = ("Light", "Dark", "Teya", "Donald")

_PANEL_PALETTES = {
    "Dark": {
        "bg": "#1f2430",
        "fg": "#e5e7eb",
        "control_bg": "#2d3646",
        "border": "#5f6d82",
        "hover": "#374256",
        "selected_bg": "#2563eb",
        "selected_fg": "#ffffff",
        "checkbox_bg": "#2d3646",
        "progress_bg": "#2d3646",
        "progress_chunk": "#2563eb",
        "group_weight": "600",
    },
    "Teya": {
        "bg": "#ffd0e5",
        "fg": "#0b7f7a",
        "control_bg": "#ffc0dc",
        "border": "#1db8b2",
        "hover": "#ffb1d5",
        "selected_bg": "#2ecfc9",
        "selected_fg": "#073f3c",
        "checkbox_bg": "#ffe6f1",
        "progress_bg": "#ffc0dc",
        "progress_chunk": "#2ecfc9",
        "group_weight": "700",
    },
    "Donald": {
        "bg": "#d97706",
        "fg": "#ffffff",
        "control_bg": "#b85f00",
        "border": "#f3a451",
        "hover": "#c76b06",
        "selected_bg": "#ffd19e",
        "selected_fg": "#7c2d12",
        "checkbox_bg": "#c96a04",
        "progress_bg": "#b85f00",
        "progress_chunk": "#ffd19e",
        "group_weight": "700",
    },
    "Light": {
        "bg": "#f4f6f9",
        "fg": "#1f2937",
        "control_bg": "#ffffff",
        "border": "#b7c0cc",
        "hover": "#edf2f7",
        "selected_bg": "#2563eb",
        "selected_fg": "#ffffff",
        "checkbox_bg": "#ffffff",
        "progress_bg": "#ffffff",
        "progress_chunk": "#2563eb",
        "group_weight": "600",
    },
}

_WORKFLOW_PALETTES = {
    "Dark": {
        "bg": "#1f2430",
        "fg": "#e5e7eb",
        "control_bg": "#2a3140",
        "border": "#556070",
        "hover": "#344054",
        "checked_bg": "#2563eb",
        "checked_border": "#60a5fa",
        "checked_fg": "#ffffff",
        "checked_weight": "600",
        "header_bg": "#2d3646",
        "table_selected_bg": "#3b82f6",
        "table_selected_fg": "#ffffff",
    },
    "Teya": {
        "bg": "#ffd0e5",
        "fg": "#0b7f7a",
        "control_bg": "#ffe6f1",
        "border": "#1db8b2",
        "hover": "#ffd9ea",
        "checked_bg": "#2ecfc9",
        "checked_border": "#0b7f7a",
        "checked_fg": "#073f3c",
        "checked_weight": "700",
        "header_bg": "#ffc4df",
        "table_selected_bg": "#2ecfc9",
        "table_selected_fg": "#073f3c",
    },
    "Donald": {
        "bg": "#d97706",
        "fg": "#ffffff",
        "control_bg": "#c96a04",
        "border": "#f3a451",
        "hover": "#c76b06",
        "checked_bg": "#b85f00",
        "checked_border": "#ffd19e",
        "checked_fg": "#ffffff",
        "checked_weight": "700",
        "header_bg": "#c96a04",
        "table_selected_bg": "#2563eb",
        "table_selected_fg": "#ffffff",
    },
    "Light": {
        "bg": "#f4f6f9",
        "fg": "#1f2937",
        "control_bg": "#ffffff",
        "border": "#c9d0da",
        "hover": "#edf2f7",
        "checked_bg": "#2563eb",
        "checked_border": "#1d4ed8",
        "checked_fg": "#ffffff",
        "checked_weight": "600",
        "header_bg": "#eef2f7",
        "table_selected_bg": "#2563eb",
        "table_selected_fg": "#ffffff",
    },
}

_SWITCH_CHECKBOX_PALETTES = {
    "Dark": ("#485569", "#22c55e", "#556070"),
    "Teya": ("#f6a9cb", "#2ecfc9", "#1db8b2"),
    "Donald": ("#d97706", "#2563eb", "#f3a451"),
    "Light": ("#d1d5db", "#2563eb", "#c9d0da"),
}


def normalize_theme_name(theme_name="Dark") -> str:
    theme = str(theme_name or "Dark").strip().title()
    return theme if theme in _VALID_THEMES else "Dark"


def dialog_theme_stylesheet(theme_name="Dark"):
    theme = normalize_theme_name(theme_name)
    palette = _PANEL_PALETTES[theme]
    style = (
        f"QDialog, QWidget {{ background: {palette['bg']}; color: {palette['fg']}; }}"
        f"QPushButton {{ background: {palette['control_bg']}; color: {palette['fg']}; border: 1px solid {palette['border']}; border-radius: 6px; padding: 5px 10px; }}"
        f"QPushButton:hover {{ background: {palette['hover']}; }}"
    )
    return theme, style


def panel_dialog_stylesheet(
    theme_name="Dark",
    *,
    control_selector="QPushButton",
    include_groupbox=False,
    include_tabs=False,
    include_progress=False,
    checkbox_indicator_selector="",
):
    theme = normalize_theme_name(theme_name)
    palette = _PANEL_PALETTES[theme]
    parts = [
        f"QDialog, QWidget {{ background: {palette['bg']}; color: {palette['fg']}; }}",
    ]
    if include_groupbox:
        parts.append(
            f"QGroupBox {{ border: 1px solid {palette['border']}; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: {palette['group_weight']}; }}"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
    parts.append(
        f"{control_selector} {{ background: {palette['control_bg']}; color: {palette['fg']}; border: 1px solid {palette['border']}; border-radius: 6px; padding: 5px 8px; }}"
        f"QPushButton:hover {{ background: {palette['hover']}; }}"
    )
    if checkbox_indicator_selector:
        parts.append(
            f"{checkbox_indicator_selector}::indicator {{ width: 14px; height: 14px; border: 1px solid {palette['border']}; border-radius: 2px; background: {palette['checkbox_bg']}; }}"
            f"{checkbox_indicator_selector}::indicator:checked {{ background: #22c55e; border: 1px solid #15803d; }}"
        )
    if include_tabs:
        parts.append(
            f"QTabWidget::pane {{ border: 1px solid {palette['border']}; border-radius: 8px; top: -1px; }}"
            f"QTabBar::tab {{ background: {palette['control_bg']}; color: {palette['fg']}; border: 1px solid {palette['border']}; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}"
            f"QTabBar::tab:selected {{ background: {palette['selected_bg']}; color: {palette['selected_fg']}; }}"
        )
    parts.append(
        f"QTableWidget::item:selected {{ background: {palette['selected_bg']}; color: {palette['selected_fg']}; }}"
    )
    if include_progress:
        parts.append(
            f"QProgressBar {{ border: 1px solid {palette['border']}; border-radius: 6px; text-align: center; background: {palette['progress_bg']}; }}"
            f"QProgressBar::chunk {{ background: {palette['progress_chunk']}; border-radius: 5px; }}"
        )
    return theme, ''.join(parts)


def workflow_dialog_stylesheet(
    theme_name="Dark",
    *,
    control_selector="QPushButton, QComboBox, QLineEdit, QTableWidget",
    extra_styles="",
):
    theme = normalize_theme_name(theme_name)
    palette = _WORKFLOW_PALETTES[theme]
    style = (
        f"QWidget {{ background: {palette['bg']}; color: {palette['fg']}; font-size: 11pt; }} "
        f"{control_selector} {{ background: {palette['control_bg']}; color: {palette['fg']}; border: 1px solid {palette['border']}; border-radius: 5px; }} "
        "QPushButton { min-height: 30px; padding: 4px 10px; } "
        f"QPushButton:hover {{ background: {palette['hover']}; }} "
        "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
        f"QPushButton#workflowStepButton:checked {{ background: {palette['checked_bg']}; border: 2px solid {palette['checked_border']}; color: {palette['checked_fg']}; font-weight: {palette['checked_weight']}; }} "
        f"QPushButton#tableToggleButton {{ min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; padding: 0; border-radius: 12px; }} "
        f"QPushButton#tableToggleButton:hover {{ border: 1px solid {palette['checked_border']}; background: {palette['hover']}; }} "
        "QPushButton#tableToggleButton:checked { background: #22c55e; color: transparent; border: 1px solid #15803d; } "
        f"QPushButton#tableToggleButton:disabled {{ color: transparent; border: 1px solid {palette['border']}; }} "
        "QPushButton#tableToggleButton:checked:disabled { background: #86efac; color: transparent; border: 1px solid #22c55e; } "
        "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 30px; padding: 2px 4px; } "
        f"QHeaderView::section {{ background: {palette['header_bg']}; color: {palette['fg']}; border: 1px solid {palette['border']}; }} "
        f"QTableWidget::item:selected {{ background: {palette['table_selected_bg']}; color: {palette['table_selected_fg']}; }}"
    )
    return theme, style + str(extra_styles or '')


def switch_checkbox_theme_styles(theme_name="Dark", *, builder):
    theme = normalize_theme_name(theme_name)
    track, fill, border = _SWITCH_CHECKBOX_PALETTES[theme]
    return builder(track, fill, border)


__all__ = [
    'dialog_theme_stylesheet',
    'normalize_theme_name',
    'panel_dialog_stylesheet',
    'switch_checkbox_theme_styles',
    'workflow_dialog_stylesheet',
]
