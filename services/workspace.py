#!/usr/bin/env python3
"""Workspace entry storage helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WorkspaceStore:
    """Small state container for workspace entries and user-visible titles."""

    entries: dict = field(default_factory=dict)
    titles: dict = field(default_factory=dict)
    _derived_counter: int = 0

    @staticmethod
    def file_entry_id(path: Path) -> str:
        return f"file::{Path(path)}"

    def new_derived_id(self) -> str:
        self._derived_counter += 1
        return f"derived::{self._derived_counter}"

    def add_file_entry(self, path: Path, *, label: str | None = None) -> tuple[str, dict]:
        file_path = Path(path)
        entry_id = self.file_entry_id(file_path)
        entry = {
            "id": entry_id,
            "kind": "file",
            "path": file_path,
            "selected_key": None,
            "sample_index": None,
            "auto_title": True,
            "label": str(label or file_path.name),
        }
        self.entries[entry_id] = entry
        return entry_id, entry

    def add_derived_entry(
        self,
        matrix,
        *,
        label: str,
        source_path=None,
        selected_key=None,
        sample_index=None,
        auto_title: bool = True,
        extra_fields: dict | None = None,
    ) -> tuple[str, dict]:
        entry_id = self.new_derived_id()
        normalized_source = None
        if source_path:
            try:
                normalized_source = Path(source_path)
            except Exception:
                normalized_source = source_path
        entry = {
            "id": entry_id,
            "kind": "derived",
            "matrix": np.asarray(matrix),
            "source_path": normalized_source,
            "selected_key": selected_key,
            "sample_index": sample_index,
            "auto_title": bool(auto_title),
            "label": str(label or "derived"),
        }
        if extra_fields:
            entry.update(dict(extra_fields))
        self.entries[entry_id] = entry
        self.titles[entry_id] = entry["label"]
        return entry_id, entry

    def remove_entry(self, entry_id) -> None:
        self.entries.pop(entry_id, None)
        self.titles.pop(entry_id, None)

    def clear(self) -> None:
        self.entries.clear()
        self.titles.clear()
