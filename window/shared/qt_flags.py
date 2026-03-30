#!/usr/bin/env python3
"""Shared Qt compatibility helpers for item flags and enum aliases."""

from __future__ import annotations

_QT = None


def install_qt_compat_aliases(Qt, qt_lib) -> None:
    global _QT
    _QT = Qt
    if int(qt_lib) != 6:
        return
    try:
        Qt.Checked = Qt.CheckState.Checked
    except Exception:
        pass
    try:
        Qt.Unchecked = Qt.CheckState.Unchecked
    except Exception:
        pass


def _resolve_qt(Qt=None):
    if Qt is not None:
        return Qt
    if _QT is not None:
        return _QT
    try:
        from PyQt6.QtCore import Qt as QtCoreQt

        return QtCoreQt
    except Exception:
        from PyQt5.QtCore import Qt as QtCoreQt

        return QtCoreQt


def is_enabled_flag(Qt=None):
    qt = _resolve_qt(Qt)
    return getattr(qt, "ItemIsEnabled", getattr(qt.ItemFlag, "ItemIsEnabled"))


def is_selectable_flag(Qt=None):
    qt = _resolve_qt(Qt)
    return getattr(qt, "ItemIsSelectable", getattr(qt.ItemFlag, "ItemIsSelectable"))


def is_user_checkable_flag(Qt=None):
    qt = _resolve_qt(Qt)
    return getattr(qt, "ItemIsUserCheckable", getattr(qt.ItemFlag, "ItemIsUserCheckable"))


def is_editable_flag(Qt=None):
    qt = _resolve_qt(Qt)
    return getattr(qt, "ItemIsEditable", getattr(qt.ItemFlag, "ItemIsEditable"))
