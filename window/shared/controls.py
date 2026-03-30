#!/usr/bin/env python3
"""Shared small UI control factories."""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import QPushButton
except Exception:
    from PyQt5.QtWidgets import QPushButton


def make_toggle_button(text: str = "", checked: bool = False, object_name: str = "tableToggleButton") -> QPushButton:
    button = QPushButton(text)
    button.setObjectName(str(object_name))
    button.setCheckable(True)
    button.setChecked(bool(checked))
    button.setFixedSize(22, 22)
    return button
