#!/usr/bin/env python3
"""Shared covariate decoding and table helpers for dialogs."""

from __future__ import annotations

import numpy as np


def decode_scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:
            return str(value)
    return value


def display_text(value):
    value = decode_scalar(value)
    if value is None:
        return ""
    return str(value)


def covars_to_rows(covars_info):
    if covars_info is None:
        return [], []

    df = covars_info.get("df")
    if df is not None:
        columns = [str(col) for col in df.columns]
        rows = []
        for record in df.to_dict(orient="records"):
            rows.append({col: decode_scalar(record.get(col)) for col in columns})
        return columns, rows

    data = covars_info.get("data")
    if data is None:
        return [], []

    arr = np.asarray(data)
    if getattr(arr.dtype, "names", None):
        columns = [str(col) for col in arr.dtype.names]
        rows = []
        for rec in arr:
            rows.append({col: decode_scalar(rec[col]) for col in columns})
        return columns, rows

    if arr.ndim == 2:
        columns = [f"col_{idx}" for idx in range(arr.shape[1])]
        rows = []
        for row in arr:
            rows.append({columns[idx]: decode_scalar(row[idx]) for idx in range(arr.shape[1])})
        return columns, rows

    return [], []


def column_is_numeric(values):
    has_value = False
    for value in values:
        text = display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value
