#!/usr/bin/env python3
"""Helpers for preparing workspace entries for matrix plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlotEntryResolution:
    matrix: np.ndarray
    key: str | None
    source_path: Path | None
    stack_axis: int | None
    stack_len: int | None


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
    names = list(available_colormap_names or [])
    cmap_name = str(entry.get("matrix_colormap", defaults["matrix_colormap"]) or "").strip()
    if not cmap_name or (names and cmap_name not in names):
        if defaults["matrix_colormap"] in names:
            cmap_name = defaults["matrix_colormap"]
        elif fallback_colormap in names:
            cmap_name = fallback_colormap
        elif names:
            cmap_name = names[0]
        else:
            cmap_name = defaults["matrix_colormap"]
    entry["matrix_colormap"] = cmap_name
    entry["display_auto"] = bool(entry.get("display_auto", defaults["display_auto"]))
    entry["display_min_text"] = str(entry.get("display_min_text", defaults["display_min_text"]) or "").strip()
    entry["display_max_text"] = str(entry.get("display_max_text", defaults["display_max_text"]) or "").strip()
    entry["display_scale"] = normalize_display_scale(entry.get("display_scale", defaults["display_scale"]))
    return entry


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


def resolve_entry_plot(
    entry,
    *,
    ensure_entry_key,
    load_matrix_from_npz,
    stack_axis,
    average_to_square,
    select_stack_slice,
):
    source_path_raw = entry.get("source_path", entry.get("path"))
    source_path = Path(source_path_raw) if source_path_raw else None

    if entry.get("kind") == "derived":
        return PlotEntryResolution(
            matrix=np.asarray(entry.get("matrix")),
            key=entry.get("selected_key"),
            source_path=source_path,
            stack_axis=None,
            stack_len=None,
        )

    key = ensure_entry_key(entry)
    if not key:
        raise KeyError("No valid matrix key selected.")

    raw = load_matrix_from_npz(entry["path"], key, average=False)
    axis = None
    stack_len = None
    if raw.ndim == 3:
        axis = stack_axis(raw.shape)
        if axis is None:
            matrix = average_to_square(raw)
        else:
            stack_len = int(raw.shape[axis])
            sample_index = entry.get("sample_index", -1)
            if sample_index is None or sample_index < 0 or sample_index >= stack_len:
                matrix = average_to_square(raw)
            else:
                matrix = select_stack_slice(raw, axis, sample_index)
    else:
        matrix = raw
    return PlotEntryResolution(
        matrix=np.asarray(matrix),
        key=key,
        source_path=source_path,
        stack_axis=axis,
        stack_len=stack_len,
    )


def normalize_matrix_labels(matrix, labels, names, *, to_string_list):
    matrix = np.asarray(matrix)
    labels_list = to_string_list(labels)
    names_list = to_string_list(names)
    expected_len = matrix.shape[0] if matrix.ndim >= 1 else 0
    if labels_list is not None and len(labels_list) != expected_len:
        labels_list = None
    if names_list is not None and len(names_list) != expected_len:
        names_list = None
    return labels_list, names_list
