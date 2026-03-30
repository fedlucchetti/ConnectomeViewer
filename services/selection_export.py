#!/usr/bin/env python3
"""Helpers for selection state and matrix export packaging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .entry_helpers import entry_source_path, safe_name_fragment


@dataclass(frozen=True)
class KeyOptionsState:
    items: tuple[str, ...]
    selected_key: str | None
    enabled: bool


@dataclass(frozen=True)
class SelectionChangeState:
    key_options: KeyOptionsState
    source_path: Path | None
    title_text: str
    auto_title: bool


@dataclass(frozen=True)
class GridExportItem:
    matrix: object
    title: str
    colormap: object
    vmin: float | None
    vmax: float | None
    zscale: str


@dataclass(frozen=True)
class GridExportBundle:
    items: tuple[GridExportItem, ...]
    skipped: tuple[str, ...]
    error: str | None


def build_key_options_state(entry, *, valid_keys=None) -> KeyOptionsState:
    if entry is None:
        return KeyOptionsState(items=tuple(), selected_key=None, enabled=False)
    if entry.get("kind") == "file":
        items = tuple(str(key) for key in (valid_keys or []))
        selected_key = entry.get("selected_key")
        if selected_key and selected_key in items:
            normalized_key = str(selected_key)
        elif items:
            normalized_key = items[0]
            entry["selected_key"] = normalized_key
        else:
            normalized_key = None
            entry["selected_key"] = None
        return KeyOptionsState(items=items, selected_key=normalized_key, enabled=bool(items))

    selected_key = str(entry.get("selected_key") or "").strip() or None
    items = (selected_key,) if selected_key else tuple()
    return KeyOptionsState(items=items, selected_key=selected_key, enabled=False)


def build_selection_change_state(
    entry,
    *,
    valid_keys=None,
    stored_title: str = "",
    default_title: str = "",
) -> SelectionChangeState:
    key_options = build_key_options_state(entry, valid_keys=valid_keys)
    label = str((entry or {}).get("label", "Matrix") or "Matrix")
    auto_title = bool((entry or {}).get("auto_title", True))
    title_text = str(default_title or label) if auto_title else str(stored_title or label)
    return SelectionChangeState(
        key_options=key_options,
        source_path=entry_source_path(entry),
        title_text=title_text,
        auto_title=auto_title,
    )


def default_matrix_export_name(entry, *, selected_key: str | None) -> str:
    source_path = entry_source_path(entry)
    if source_path is not None:
        source_stem = source_path.stem
    else:
        source_stem = str((entry or {}).get("label", "matrix") or "matrix")
    key_part = safe_name_fragment(selected_key or "matrix")
    return f"{safe_name_fragment(source_stem)}_{key_part}_matrix_pop_avg.npz"


def normalize_grid_export_output_path(path_value, *, selected_filter: str) -> Path:
    output_path = Path(path_value).expanduser()
    if output_path.suffix.lower() in {".pdf", ".svg", ".png"}:
        return output_path
    selected_filter = str(selected_filter or "")
    if "SVG" in selected_filter:
        return output_path.with_suffix(".svg")
    if "PNG" in selected_filter:
        return output_path.with_suffix(".png")
    return output_path.with_suffix(".pdf")


def collect_grid_export_items(
    entry_ids,
    entries,
    titles,
    *,
    matrix_for_entry,
    display_limits_for_entry,
    display_scale_for_entry,
    log_scale_error,
    selected_colormap_for_entry,
) -> GridExportBundle:
    plot_items = []
    skipped = []
    for entry_id in entry_ids:
        entry = entries.get(entry_id)
        if entry is None:
            continue
        label = str(entry.get("label", entry_id) or entry_id)
        try:
            matrix, _selected_key = matrix_for_entry(entry)
        except Exception as exc:
            skipped.append(f"{label} ({exc})")
            continue
        vmin, vmax, scaling_error = display_limits_for_entry(entry)
        if scaling_error:
            return GridExportBundle(items=tuple(), skipped=tuple(skipped), error=f"{label}: {scaling_error}")
        zscale = display_scale_for_entry(entry)
        if zscale == "log":
            scale_error = log_scale_error(matrix, vmin, vmax)
            if scale_error:
                return GridExportBundle(items=tuple(), skipped=tuple(skipped), error=f"{label}: {scale_error}")
        plot_items.append(
            GridExportItem(
                matrix=matrix,
                title=str(titles.get(entry_id, entry.get("label", "Matrix")) or "Matrix"),
                colormap=selected_colormap_for_entry(entry),
                vmin=vmin,
                vmax=vmax,
                zscale=zscale,
            )
        )
    return GridExportBundle(items=tuple(plot_items), skipped=tuple(skipped), error=None)
