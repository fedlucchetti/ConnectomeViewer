#!/usr/bin/env python3
"""Pure helpers for workspace entry naming and export metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def safe_name_fragment(text: str) -> str:
    token = str(text or "").strip()
    if not token:
        return "matrix"
    cleaned = []
    for ch in token:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "matrix"


def _strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _parse_participant_id(value: str):
    text = str(value)
    if "-" in text:
        parts = text.split("-")
        if parts[0] in {"sub", "subject", "participant"}:
            return None, parts[-1]
        return parts[0], parts[-1]
    if "_" in text:
        parts = text.split("_")
        if parts[0] in {"sub", "subject", "participant"}:
            return None, parts[-1]
        return parts[0], parts[-1]
    return None, text


def _covars_series(info, name):
    if info is None:
        return None
    df = info.get("df") if isinstance(info, dict) else None
    if df is not None:
        return df[name].to_numpy() if name in df.columns else None
    data = info.get("data") if isinstance(info, dict) else None
    if data is None or getattr(data.dtype, "names", None) is None or name not in data.dtype.names:
        return None
    return data[name]


def normalize_covar_selection(columns, requested_name=None):
    normalized_columns = [str(column) for column in (columns or [])]
    requested = str(requested_name or "").strip()
    selected = requested if requested and requested in normalized_columns else None
    return normalized_columns, selected


def default_entry_title(entry, *, covars_info=None, group_value=None) -> str:
    base_label = entry.get("label", "Matrix")
    if entry.get("kind") != "file":
        return base_label
    sample_index = entry.get("sample_index")
    if sample_index is None or sample_index < 0:
        return base_label

    participant = _covars_series(covars_info, "participant_id")
    session = _covars_series(covars_info, "session_id")
    if participant is None or session is None:
        return base_label
    if sample_index >= len(participant) or sample_index >= len(session):
        return base_label

    participant_value = str(participant[sample_index])
    session_value = str(session[sample_index])
    group, sub = _parse_participant_id(participant_value)
    if group_value:
        group = str(group_value)
    sub = _strip_prefix(sub, "sub-")
    ses = _strip_prefix(session_value, "ses-")
    if group:
        return f"{group}-sub-{sub}_ses-{ses}"
    return f"sub-{sub}_ses-{ses}"


def entry_source_path(entry):
    if entry is None:
        return None
    source_path = entry.get("source_path", entry.get("path"))
    if not source_path:
        return None
    return Path(source_path)


def _has_values(values) -> bool:
    if values is None:
        return False
    try:
        return len(values) > 0
    except Exception:
        return True


def collect_export_metadata(
    entry,
    selected_key,
    *,
    labels=None,
    names=None,
    current_parcel_labels=None,
    current_parcel_names=None,
):
    metadata = {}

    if labels is not None:
        labels = np.asarray(labels)
    if names is not None:
        names = np.asarray(names, dtype=object)

    source_path = entry_source_path(entry)
    if source_path is not None and source_path.exists():
        try:
            with np.load(source_path, allow_pickle=True) as npz:
                for key in ("group", "modality", "metabolites"):
                    if key in npz:
                        metadata[key] = np.asarray(npz[key])
        except Exception:
            pass

    if labels is None and _has_values(current_parcel_labels):
        labels = np.asarray(current_parcel_labels)
    if names is None and _has_values(current_parcel_names):
        names = np.asarray(current_parcel_names, dtype=object)

    if labels is not None:
        metadata["parcel_labels_group"] = labels
    if names is not None:
        metadata["parcel_names_group"] = names
    if source_path is not None:
        metadata["source_file"] = np.asarray(str(source_path))
    if selected_key:
        metadata["source_key"] = np.asarray(str(selected_key))
    sample_index = entry.get("sample_index")
    if sample_index is not None:
        try:
            metadata["sample_index"] = np.asarray(int(sample_index))
        except Exception:
            pass
    return metadata
