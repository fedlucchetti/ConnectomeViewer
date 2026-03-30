#!/usr/bin/env python3
"""Helpers for matrix display controls and sample-selection state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DisplayControlsState:
    colormap_name: str
    auto_scale: bool
    min_text: str
    max_text: str
    scale_label: str
    min_enabled: bool
    max_enabled: bool


@dataclass(frozen=True)
class SampleControlsState:
    enabled: bool
    minimum: int
    maximum: int
    value: int
    special_value_text: str
    add_enabled: bool


def default_entry_display_settings(default_matrix_colormap: str) -> dict:
    return {
        "matrix_colormap": str(default_matrix_colormap or ""),
        "display_auto": True,
        "display_min_text": "",
        "display_max_text": "",
        "display_scale": "linear",
    }


def normalize_display_scale(choice) -> str:
    text = str(choice or "").strip().lower()
    if text.startswith("log"):
        return "log"
    return "linear"


def _normalize_colormap_name(
    current_name: str,
    *,
    default_matrix_colormap: str,
    available_colormap_names,
    fallback_colormap: str,
) -> str:
    names = list(available_colormap_names or [])
    cmap_name = str(current_name or "").strip()
    if cmap_name and (not names or cmap_name in names):
        return cmap_name
    if default_matrix_colormap in names:
        return default_matrix_colormap
    if fallback_colormap in names:
        return fallback_colormap
    if names:
        return str(names[0] or "")
    return str(default_matrix_colormap or fallback_colormap or "")


def ensure_entry_display_settings(
    entry,
    *,
    default_matrix_colormap: str,
    available_colormap_names,
    fallback_colormap: str,
):
    if entry is None:
        return None
    defaults = default_entry_display_settings(default_matrix_colormap)
    entry["matrix_colormap"] = _normalize_colormap_name(
        entry.get("matrix_colormap", defaults["matrix_colormap"]),
        default_matrix_colormap=default_matrix_colormap,
        available_colormap_names=available_colormap_names,
        fallback_colormap=fallback_colormap,
    )
    entry["display_auto"] = bool(entry.get("display_auto", defaults["display_auto"]))
    entry["display_min_text"] = str(entry.get("display_min_text", defaults["display_min_text"]) or "").strip()
    entry["display_max_text"] = str(entry.get("display_max_text", defaults["display_max_text"]) or "").strip()
    entry["display_scale"] = normalize_display_scale(entry.get("display_scale", defaults["display_scale"]))
    return entry


def build_display_controls_state(
    entry,
    *,
    default_matrix_colormap: str,
    available_colormap_names,
    fallback_colormap: str,
) -> DisplayControlsState | None:
    ensured = ensure_entry_display_settings(
        entry,
        default_matrix_colormap=default_matrix_colormap,
        available_colormap_names=available_colormap_names,
        fallback_colormap=fallback_colormap,
    )
    if ensured is None:
        return None
    auto_scale = bool(ensured.get("display_auto", True))
    scale = normalize_display_scale(ensured.get("display_scale", "linear"))
    return DisplayControlsState(
        colormap_name=str(ensured.get("matrix_colormap") or ""),
        auto_scale=auto_scale,
        min_text=str(ensured.get("display_min_text", "") or ""),
        max_text=str(ensured.get("display_max_text", "") or ""),
        scale_label="Log" if scale == "log" else "Linear",
        min_enabled=not auto_scale,
        max_enabled=not auto_scale,
    )


def store_display_controls(
    entry,
    *,
    colormap_name: str,
    auto_scale: bool,
    min_text: str,
    max_text: str,
    scale_choice,
    default_matrix_colormap: str,
    available_colormap_names,
    fallback_colormap: str,
):
    ensured = ensure_entry_display_settings(
        entry,
        default_matrix_colormap=default_matrix_colormap,
        available_colormap_names=available_colormap_names,
        fallback_colormap=fallback_colormap,
    )
    if ensured is None:
        return None
    ensured["matrix_colormap"] = _normalize_colormap_name(
        colormap_name,
        default_matrix_colormap=default_matrix_colormap,
        available_colormap_names=available_colormap_names,
        fallback_colormap=fallback_colormap,
    )
    ensured["display_auto"] = bool(auto_scale)
    ensured["display_min_text"] = str(min_text or "").strip()
    ensured["display_max_text"] = str(max_text or "").strip()
    ensured["display_scale"] = normalize_display_scale(scale_choice)
    return build_display_controls_state(
        ensured,
        default_matrix_colormap=default_matrix_colormap,
        available_colormap_names=available_colormap_names,
        fallback_colormap=fallback_colormap,
    )


def selected_colormap_name(
    *,
    entry=None,
    current_name: str = "",
    default_matrix_colormap: str,
    available_colormap_names,
    fallback_colormap: str,
) -> str:
    if entry is not None:
        ensured = ensure_entry_display_settings(
            entry,
            default_matrix_colormap=default_matrix_colormap,
            available_colormap_names=available_colormap_names,
            fallback_colormap=fallback_colormap,
        )
        return str(ensured.get("matrix_colormap") or fallback_colormap or "")
    return _normalize_colormap_name(
        current_name,
        default_matrix_colormap=default_matrix_colormap,
        available_colormap_names=available_colormap_names,
        fallback_colormap=fallback_colormap,
    )


def parse_display_limits(*, auto_scale: bool, min_text: str = "", max_text: str = ""):
    if auto_scale:
        return None, None, None
    min_value = None
    max_value = None
    min_text = str(min_text or "").strip()
    max_text = str(max_text or "").strip()
    if min_text:
        try:
            min_value = float(min_text)
        except ValueError:
            return None, None, "Display min must be numeric."
    if max_text:
        try:
            max_value = float(max_text)
        except ValueError:
            return None, None, "Display max must be numeric."
    if min_value is not None and max_value is not None and min_value >= max_value:
        return None, None, "Display min must be smaller than max."
    return min_value, max_value, None


def log_scale_error(matrix, vmin, vmax):
    if matrix is None:
        return "Log scale requires matrix values."
    values = np.asarray(matrix, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "Log scale requires finite values."
    max_val = float(np.max(finite))
    if max_val <= 0:
        return "Log scale requires positive values (max > 0)."
    min_val = float(np.min(finite))
    if min_val < 0:
        return "Log scale does not support negative values."
    if vmin is not None and vmin <= 0:
        return "Display min must be > 0 for log scale."
    if vmax is not None and vmax <= 0:
        return "Display max must be > 0 for log scale."
    return None


def update_sample_control_state(entry, *, axis, stack_len) -> SampleControlsState:
    special_value_text = "Average"
    if axis is None or stack_len is None:
        if entry is not None:
            entry["sample_index"] = None
            entry["stack_axis"] = None
            entry["stack_len"] = None
        return SampleControlsState(
            enabled=False,
            minimum=-1,
            maximum=0,
            value=-1,
            special_value_text=special_value_text,
            add_enabled=False,
        )

    normalized_axis = int(axis)
    normalized_len = max(int(stack_len), 0)
    sample_index = -1
    if entry is not None:
        entry.setdefault("sample_index", -1)
        raw_sample_index = entry.get("sample_index")
        if raw_sample_index is not None:
            try:
                sample_index = int(raw_sample_index)
            except Exception:
                sample_index = -1
    if sample_index >= normalized_len:
        sample_index = -1
    if entry is not None:
        entry["sample_index"] = sample_index
        entry["stack_axis"] = normalized_axis
        entry["stack_len"] = normalized_len
    return SampleControlsState(
        enabled=True,
        minimum=-1,
        maximum=max(normalized_len - 1, 0),
        value=sample_index,
        special_value_text=special_value_text,
        add_enabled=True,
    )
