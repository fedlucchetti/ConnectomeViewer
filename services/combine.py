#!/usr/bin/env python3
"""Pure helpers for matrix combination workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CombineAlignment:
    first_matrix: np.ndarray
    second_matrix: np.ndarray
    parcel_labels_group: np.ndarray | None
    parcel_names_group: list[str] | None
    summary_text: str
    used_label_intersection: bool


@dataclass
class CombineCorrelationStats:
    first_values: np.ndarray
    second_values: np.ndarray
    mode_text: str
    r_value: float
    p_value: float
    slope: float
    intercept: float


def combine_operation_label(operation: str) -> str:
    mapping = {
        "add": "Addition",
        "subtract": "Subtraction",
        "intersect": "Intersect",
        "correlation": "Correlation",
        "elementwise_product": "Elementwise Product",
        "matmul": "Matrix Multiplication",
    }
    return mapping.get(str(operation or "").strip().lower(), "Combine")


def combine_operation_symbol(operation: str) -> str:
    mapping = {
        "add": "+",
        "subtract": "-",
        "intersect": "intersect",
        "correlation": "corr",
        "elementwise_product": "*",
        "matmul": "@",
    }
    return mapping.get(str(operation or "").strip().lower(), "?")


def format_p_value(value) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    if number < 1e-3:
        return f"{number:.3e}"
    return f"{number:.4f}"


def align_matrices_by_intersection(
    first_matrix,
    second_matrix,
    *,
    first_labels_raw=None,
    second_labels_raw=None,
    first_names=None,
    second_names=None,
    coerce_label_indices=None,
) -> CombineAlignment:
    first_matrix = np.asarray(first_matrix, dtype=float)
    second_matrix = np.asarray(second_matrix, dtype=float)
    first_shape = tuple(first_matrix.shape)
    second_shape = tuple(second_matrix.shape)

    if (
        first_matrix.ndim != 2
        or second_matrix.ndim != 2
        or first_matrix.shape[0] != first_matrix.shape[1]
        or second_matrix.shape[0] != second_matrix.shape[1]
    ):
        return CombineAlignment(
            first_matrix=first_matrix,
            second_matrix=second_matrix,
            parcel_labels_group=None,
            parcel_names_group=None,
            summary_text="No parcel-label alignment applied.",
            used_label_intersection=False,
        )

    if coerce_label_indices is None:
        first_labels = np.asarray(first_labels_raw).reshape(-1) if first_labels_raw is not None else None
        second_labels = np.asarray(second_labels_raw).reshape(-1) if second_labels_raw is not None else None
    else:
        first_labels = (
            coerce_label_indices(first_labels_raw, first_matrix.shape[0])
            if first_labels_raw is not None
            else None
        )
        second_labels = (
            coerce_label_indices(second_labels_raw, second_matrix.shape[0])
            if second_labels_raw is not None
            else None
        )

    if first_labels is None or second_labels is None:
        if first_matrix.shape != second_matrix.shape:
            raise ValueError(
                "Matrices differ in size and cannot be aligned because parcel_labels_group is "
                "missing, invalid, or does not match the matrix size."
            )
        result_labels = first_labels_raw if first_labels_raw is not None else None
        result_names = first_names if first_names is not None else None
        return CombineAlignment(
            first_matrix=first_matrix,
            second_matrix=second_matrix,
            parcel_labels_group=result_labels,
            parcel_names_group=result_names,
            summary_text="Used original matrix order (no valid parcel-label intersection).",
            used_label_intersection=False,
        )

    if len(set(first_labels)) != len(first_labels) or len(set(second_labels)) != len(second_labels):
        raise ValueError("parcel_labels_group must be unique in both matrices for label-based alignment.")

    second_lookup = {int(label): idx for idx, label in enumerate(second_labels)}
    first_indices = []
    second_indices = []
    common_labels = []
    for idx, label in enumerate(first_labels):
        other_idx = second_lookup.get(int(label))
        if other_idx is None:
            continue
        first_indices.append(int(idx))
        second_indices.append(int(other_idx))
        common_labels.append(int(label))

    if not common_labels:
        raise ValueError("No overlapping parcel labels were found between the selected matrices.")

    aligned_first = np.asarray(first_matrix[np.ix_(first_indices, first_indices)], dtype=float)
    aligned_second = np.asarray(second_matrix[np.ix_(second_indices, second_indices)], dtype=float)

    result_names = None
    if first_names is not None and len(first_names) == len(first_labels):
        result_names = [first_names[idx] for idx in first_indices]
    elif second_names is not None and len(second_names) == len(second_labels):
        result_names = [second_names[idx] for idx in second_indices]

    summary_text = (
        f"Aligned by parcel-label intersection: {len(common_labels)} common labels "
        f"({first_shape[0]} vs {second_shape[0]} nodes)."
    )
    return CombineAlignment(
        first_matrix=aligned_first,
        second_matrix=aligned_second,
        parcel_labels_group=np.asarray(common_labels, dtype=int),
        parcel_names_group=result_names,
        summary_text=summary_text,
        used_label_intersection=True,
    )


def apply_matrix_operation(first_matrix, second_matrix, operation):
    operation_name = str(operation or "").strip().lower()
    first_matrix = np.asarray(first_matrix, dtype=float)
    second_matrix = np.asarray(second_matrix, dtype=float)

    if operation_name in {"add", "subtract", "elementwise_product", "correlation", "intersect"}:
        if first_matrix.shape != second_matrix.shape:
            raise ValueError(
                f"{combine_operation_label(operation)} requires matching shapes, "
                f"got {first_matrix.shape} and {second_matrix.shape}."
            )

    if operation_name == "add":
        return first_matrix + second_matrix
    if operation_name == "subtract":
        return first_matrix - second_matrix
    if operation_name == "intersect":
        return np.array(first_matrix, copy=True)
    if operation_name == "elementwise_product":
        return first_matrix * second_matrix
    if operation_name == "matmul":
        try:
            return np.matmul(first_matrix, second_matrix)
        except ValueError as exc:
            raise ValueError(
                f"Matrix multiplication requires compatible shapes, got "
                f"{first_matrix.shape} and {second_matrix.shape}."
            ) from exc
    raise ValueError(f"Unsupported operation: {operation}")


def correlation_vectors(first_matrix, second_matrix):
    first_matrix = np.asarray(first_matrix, dtype=float)
    second_matrix = np.asarray(second_matrix, dtype=float)
    if first_matrix.shape != second_matrix.shape:
        raise ValueError(
            f"Correlation requires matching shapes, got {first_matrix.shape} and {second_matrix.shape}."
        )

    if (
        first_matrix.ndim == 2
        and second_matrix.ndim == 2
        and first_matrix.shape[0] == first_matrix.shape[1]
        and second_matrix.shape[0] == second_matrix.shape[1]
    ):
        indices = np.triu_indices_from(first_matrix, k=1)
        first_values = np.asarray(first_matrix[indices], dtype=float)
        second_values = np.asarray(second_matrix[indices], dtype=float)
        mode_text = "upper triangle excluding diagonal"
    else:
        first_values = np.asarray(first_matrix, dtype=float).reshape(-1)
        second_values = np.asarray(second_matrix, dtype=float).reshape(-1)
        mode_text = "flattened values"

    finite_mask = np.isfinite(first_values) & np.isfinite(second_values)
    first_values = first_values[finite_mask]
    second_values = second_values[finite_mask]
    if first_values.size < 3:
        raise ValueError("Correlation needs at least three finite paired values.")
    if np.allclose(first_values, first_values[0]) or np.allclose(second_values, second_values[0]):
        raise ValueError("Correlation requires both matrices to have varying values.")
    return first_values, second_values, mode_text


def compute_correlation_stats(first_values, second_values, mode_text: str) -> CombineCorrelationStats:
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)
    try:
        from scipy.stats import linregress, pearsonr
    except Exception:
        linregress = None
        pearsonr = None

    if pearsonr is not None:
        pearson_result = pearsonr(first_values, second_values)
        r_value = getattr(pearson_result, "statistic", pearson_result[0])
        p_value = getattr(pearson_result, "pvalue", pearson_result[1])
    else:
        corr = np.corrcoef(first_values, second_values)
        r_value = float(corr[0, 1])
        p_value = np.nan

    if linregress is not None:
        regression = linregress(first_values, second_values)
        slope = float(regression.slope)
        intercept = float(regression.intercept)
    else:
        slope, intercept = np.polyfit(first_values, second_values, 1)
        slope = float(slope)
        intercept = float(intercept)

    return CombineCorrelationStats(
        first_values=first_values,
        second_values=second_values,
        mode_text=str(mode_text or ""),
        r_value=float(r_value),
        p_value=float(p_value),
        slope=slope,
        intercept=intercept,
    )
