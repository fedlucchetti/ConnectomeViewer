#!/usr/bin/env python3
"""Standalone 2D demo for path-segment penalties and free energy.

This mirrors the updated path-energy logic used in the classification
scatter free-energy view:

    alignment_i = step_unit_i dot ref_unit
    direction_penalty_i = 1 - alignment_i
    proximity_penalty_i = d_line_i / |ref|
    energy_i = direction_penalty_i + proximity_penalty_i
    total_path_energy = sum(energy_i)
    F = -(1 / lambda) * log(sum(exp(-lambda * total_path_energy)))

The demo draws:

1. One explicit example path with per-segment penalties highlighted
2. A coherent ensemble of similar paths
3. A diverse ensemble with detours and backward segments
4. The resulting free-energy curves as lambda varies
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def _point_distances_to_line(
    sample_points: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return perpendicular distances of points to a 2D line and the line length."""
    points = np.asarray(sample_points, dtype=float)
    start = np.asarray(line_start, dtype=float).reshape(-1)
    end = np.asarray(line_end, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or start.shape != (2,) or end.shape != (2,):
        raise ValueError("Line-distance inputs must be 2D coordinates.")

    line_vector = np.asarray(end - start, dtype=float)
    line_length = float(np.linalg.norm(line_vector))
    if line_length <= 0.0:
        raise ValueError("Reference line must have non-zero length.")
    line_unit = line_vector / line_length
    rel = points - start[np.newaxis, :]
    cross_vals = rel[:, 0] * line_unit[1] - rel[:, 1] * line_unit[0]
    return np.abs(np.asarray(cross_vals, dtype=float)), float(line_length)


def path_energy_breakdown(
    path_points: np.ndarray,
    ref_start: np.ndarray,
    ref_end: np.ndarray,
) -> dict:
    """Return per-segment energy terms and total path energy."""
    path_points = np.asarray(path_points, dtype=float)
    if path_points.ndim != 2 or path_points.shape[1] != 2 or path_points.shape[0] < 2:
        raise ValueError("`path_points` must have shape (N, 2) with N >= 2.")
    ref_start = np.asarray(ref_start, dtype=float).reshape(-1)
    ref_end = np.asarray(ref_end, dtype=float).reshape(-1)
    if ref_start.shape != (2,) or ref_end.shape != (2,):
        raise ValueError("Reference endpoints must be 2D vectors.")

    ref_vec = np.asarray(ref_end - ref_start, dtype=float)
    ref_norm = float(np.linalg.norm(ref_vec))
    if ref_norm <= 0.0:
        raise ValueError("Reference vector must have non-zero norm.")
    ref_unit = ref_vec / ref_norm

    steps = np.diff(path_points, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    valid = np.isfinite(step_norms) & (step_norms > 1e-12)
    if not np.all(valid):
        raise ValueError("Path contains a zero-length segment.")

    step_units = steps / step_norms[:, np.newaxis]
    alignments = np.clip(step_units @ ref_unit, -1.0, 1.0)
    direction_penalties = 1.0 - alignments

    midpoints = 0.5 * (path_points[:-1, :] + path_points[1:, :])
    line_distances, line_length = _point_distances_to_line(midpoints, ref_start, ref_end)
    proximity_penalties = line_distances / line_length
    segment_energies = direction_penalties + proximity_penalties
    total_energy = float(np.sum(segment_energies))
    return {
        "alignments": alignments,
        "direction_penalties": direction_penalties,
        "line_distances": line_distances,
        "proximity_penalties": proximity_penalties,
        "segment_energies": segment_energies,
        "total_energy": total_energy,
        "ref_unit": ref_unit,
        "ref_length": float(line_length),
        "midpoints": midpoints,
    }


def stable_free_energy(energies: np.ndarray, lam: float) -> float:
    """Compute F = -(1/lambda) log(sum(exp(-lambda * E))) stably."""
    values = np.asarray(energies, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    lam = max(1e-9, float(lam))
    scaled = -lam * values
    max_scaled = float(np.max(scaled))
    return float(-(1.0 / lam) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))))


def free_energy_curve(energies: np.ndarray, lambda_values: np.ndarray) -> np.ndarray:
    """Evaluate free energy over a lambda sweep."""
    lambda_values = np.asarray(lambda_values, dtype=float).reshape(-1)
    return np.asarray([stable_free_energy(energies, lam) for lam in lambda_values], dtype=float)


def build_ensembles():
    start_label = "R"
    end_label = "G"
    start = np.asarray((0.0, 0.0), dtype=float)
    end = np.asarray((5.0, 2.0), dtype=float)
    ref_vec = end - start
    ref_unit = ref_vec / np.linalg.norm(ref_vec)
    example_path = np.asarray(
        [(0.0, 0.0), (1.0, 0.3), (2.0, 1.0), (3.1, 0.8), (4.2, 1.6), (5.0, 2.0)],
        dtype=float,
    )

    # Each path has 6 points -> 5 segments.
    similar_paths = [
        np.asarray([(0.0, 0.0), (1.0, 0.3), (2.0, 0.8), (3.0, 1.1), (4.0, 1.6), (5.0, 2.0)], dtype=float),
        np.asarray([(0.0, 0.0), (0.9, 0.4), (1.9, 0.7), (3.0, 1.3), (4.1, 1.7), (5.0, 2.0)], dtype=float),
        np.asarray([(0.0, 0.0), (1.1, 0.2), (2.0, 0.9), (3.1, 1.0), (4.2, 1.8), (5.0, 2.0)], dtype=float),
        np.asarray([(0.0, 0.0), (1.0, 0.5), (2.2, 0.9), (3.0, 1.4), (4.0, 1.8), (5.0, 2.0)], dtype=float),
    ]

    diverse_paths = [
        np.asarray([(0.0, 0.0), (1.0, 0.8), (1.7, 0.2), (3.1, 1.0), (4.0, 2.1), (5.0, 2.0)], dtype=float),
        np.asarray([(0.0, 0.0), (0.8, -0.1), (1.6, 0.9), (2.8, 0.4), (4.1, 2.2), (5.0, 2.0)], dtype=float),
        np.asarray([(0.0, 0.0), (1.1, 0.4), (0.7, 0.7), (2.7, 1.0), (4.4, 1.8), (5.0, 2.0)], dtype=float),
        np.asarray([(0.0, 0.0), (1.3, 0.9), (2.5, 0.1), (2.2, 1.3), (4.0, 1.4), (5.0, 2.0)], dtype=float),
    ]

    return {
        "start_label": start_label,
        "end_label": end_label,
        "start": start,
        "end": end,
        "ref_vec": ref_vec,
        "ref_unit": ref_unit,
        "example_path": example_path,
        "ensembles": [
            {"name": "Similar paths", "paths": similar_paths, "color": "#2563eb"},
            {"name": "Diverse paths", "paths": diverse_paths, "color": "#dc2626"},
        ],
    }


def summarize_ensemble(name: str, paths: list[np.ndarray], ref_start: np.ndarray, ref_end: np.ndarray, lam: float) -> dict:
    """Compute per-path energies and ensemble free energy."""
    summaries = []
    total_energies = []
    for idx, path in enumerate(paths, start=1):
        breakdown = path_energy_breakdown(path, ref_start, ref_end)
        summaries.append(
            {
                "index": idx,
                "path": np.asarray(path, dtype=float),
                **breakdown,
            }
        )
        total_energies.append(float(breakdown["total_energy"]))
    total_energies = np.asarray(total_energies, dtype=float)
    return {
        "name": name,
        "paths": summaries,
        "total_energies": total_energies,
        "free_energy": stable_free_energy(total_energies, lam),
        "lambda": float(lam),
    }


def print_summary(payload: dict, lam: float) -> None:
    print("Reference line")
    print(f"  {payload['start_label']} -> {payload['end_label']}")
    print(f"  start = {payload['start']}")
    print(f"  end   = {payload['end']}")
    print(f"  ref_unit = {np.array2string(payload['ref_unit'], precision=4)}")
    print(f"  lambda = {lam:.4f}")
    print("")

    example = path_energy_breakdown(payload["example_path"], payload["start"], payload["end"])
    print("Example path penalty breakdown")
    for idx, (dot, e_dir, d_line, e_line, e_seg) in enumerate(
        zip(
            example["alignments"],
            example["direction_penalties"],
            example["line_distances"],
            example["proximity_penalties"],
            example["segment_energies"],
        ),
        start=1,
    ):
        print(
            f"  seg {idx}: dot={dot:.3f} | 1-dot={e_dir:.3f} | "
            f"d_line={d_line:.3f} | d_line/|ref|={e_line:.3f} | Eseg={e_seg:.3f}"
        )
    print(f"  Example total energy = {example['total_energy']:.6f}")
    print("")

    for ensemble in payload["ensembles"]:
        summary = summarize_ensemble(ensemble["name"], ensemble["paths"], payload["start"], payload["end"], lam)
        print(summary["name"])
        for path_summary in summary["paths"]:
            dots = ", ".join(f"{value:.3f}" for value in path_summary["alignments"])
            dir_terms = ", ".join(f"{value:.3f}" for value in path_summary["direction_penalties"])
            line_terms = ", ".join(f"{value:.3f}" for value in path_summary["proximity_penalties"])
            energies = ", ".join(f"{value:.3f}" for value in path_summary["segment_energies"])
            print(
                f"  path {path_summary['index']}: "
                f"dots=[{dots}] | dir=[{dir_terms}] | line=[{line_terms}] | "
                f"segE=[{energies}] | totalE={path_summary['total_energy']:.4f}"
            )
        print(f"  Free energy F = {summary['free_energy']:.6f}")
        print("")


def plot_penalty_breakdown(ax, path_summary: dict, start: np.ndarray, end: np.ndarray, ref_label: str) -> None:
    """Draw one path with segment-by-segment penalties annotated."""
    path = np.asarray(path_summary["path"], dtype=float)
    segments = np.stack((path[:-1, :], path[1:, :]), axis=1)
    energies = np.asarray(path_summary["segment_energies"], dtype=float)
    line_distances = np.asarray(path_summary["line_distances"], dtype=float)
    midpoints = np.asarray(path_summary["midpoints"], dtype=float)

    collection = LineCollection(
        segments,
        cmap="plasma",
        norm=plt.Normalize(float(np.min(energies)), float(np.max(energies)) if len(energies) > 1 else float(np.min(energies) + 1.0)),
        linewidths=3.4,
        alpha=0.95,
    )
    collection.set_array(energies)
    ax.add_collection(collection)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linestyle="--",
        linewidth=2.0,
        color="black",
        alpha=0.8,
        label=f"Reference {ref_label}",
    )
    ax.plot(path[:, 0], path[:, 1], color="#475569", linewidth=1.2, alpha=0.6)
    ax.scatter(path[:, 0], path[:, 1], s=36, color="#111827", zorder=4)
    ax.scatter([start[0], end[0]], [start[1], end[1]], s=95, c=["#ef4444", "#22c55e"], zorder=5)
    ax.text(start[0], start[1] - 0.18, "R", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(end[0], end[1] - 0.18, "G", ha="center", va="top", fontsize=11, fontweight="bold")

    ref_vector = np.asarray(end - start, dtype=float)
    ref_length_sq = float(np.dot(ref_vector, ref_vector))
    for idx, midpoint in enumerate(midpoints):
        if ref_length_sq > 0.0:
            projection = start + np.dot(midpoint - start, ref_vector) / ref_length_sq * ref_vector
            ax.plot(
                [midpoint[0], projection[0]],
                [midpoint[1], projection[1]],
                color="#94a3b8",
                linewidth=1.0,
                alpha=0.8,
            )
        ax.text(
            midpoint[0],
            midpoint[1] + 0.14,
            (
                f"s{idx + 1}\n"
                f"1-dot={path_summary['direction_penalties'][idx]:.2f}\n"
                f"line={path_summary['proximity_penalties'][idx]:.2f}"
            ),
            fontsize=8.0,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
        )

    cbar = plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Segment energy")
    ax.text(
        0.02,
        0.98,
        f"Example total energy = {path_summary['total_energy']:.4f}\n"
        f"Segments = direction penalty + line penalty",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#94a3b8"},
    )
    ax.set_title("Segment Penalty Breakdown")
    ax.set_xlabel("Gradient X")
    ax.set_ylabel("Gradient Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def plot_ensemble(ax, ensemble_summary: dict, start: np.ndarray, end: np.ndarray, ref_label: str, color: str) -> None:
    """Draw one ensemble on one subplot."""
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linestyle="--",
        linewidth=2.0,
        color="black",
        alpha=0.8,
        label=f"Reference {ref_label}",
    )

    energies = []
    for idx, path_summary in enumerate(ensemble_summary["paths"], start=1):
        path = np.asarray(path_summary["path"], dtype=float)
        total_energy = float(path_summary["total_energy"])
        energies.append(total_energy)
        alpha = 0.55 + 0.10 * (idx / max(1, len(ensemble_summary["paths"])))
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            alpha=min(alpha, 0.95),
            label=f"P{idx} E={total_energy:.2f}",
        )

        label_anchor = path[min(2, path.shape[0] - 2)]
        ax.text(
            label_anchor[0],
            label_anchor[1] + 0.10,
            f"P{idx}\nE={total_energy:.2f}",
            fontsize=8.5,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.82, "edgecolor": "#94a3b8"},
        )

    ax.scatter([start[0], end[0]], [start[1], end[1]], s=90, c=["#ef4444", "#22c55e"], zorder=5)
    ax.text(start[0], start[1] - 0.18, "R", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(end[0], end[1] - 0.18, "G", ha="center", va="top", fontsize=11, fontweight="bold")

    energy_list = ", ".join(f"{value:.2f}" for value in np.sort(np.asarray(energies, dtype=float)))
    ax.text(
        0.02,
        0.98,
        f"lambda = {ensemble_summary['lambda']:.2f}\n"
        f"F = {ensemble_summary['free_energy']:.4f}\n"
        f"Energies = [{energy_list}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#94a3b8"},
    )

    ax.set_title(ensemble_summary["name"])
    ax.set_xlabel("Gradient X")
    ax.set_ylabel("Gradient Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def plot_demo(payload: dict, lam: float, output: Path | None) -> None:
    start = payload["start"]
    end = payload["end"]
    ref_label = f"{payload['start_label']}{payload['end_label']}"
    example_summary = {
        "path": np.asarray(payload["example_path"], dtype=float),
        **path_energy_breakdown(payload["example_path"], start, end),
    }

    summaries = [
        summarize_ensemble(ensemble["name"], ensemble["paths"], start, end, lam)
        for ensemble in payload["ensembles"]
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24.0, 6.4), constrained_layout=True)
    fig.suptitle(
        "Path-Energy and Free-Energy Demo\n"
        "Left: one path with segment penalties, Middle: coherent family, Next: diverse family, Right: free energy vs lambda",
        fontsize=13,
    )

    plot_penalty_breakdown(axes[0], example_summary, start, end, ref_label)

    for ax, ensemble, summary in zip(axes[1:3], payload["ensembles"], summaries):
        plot_ensemble(ax, summary, start, end, ref_label, ensemble["color"])

    lambda_min = 0.05
    lambda_max = max(5.0, float(lam) * 2.5)
    lambda_values = np.linspace(lambda_min, lambda_max, 200, dtype=float)
    curve_ax = axes[3]
    for ensemble, summary in zip(payload["ensembles"], summaries):
        curve = free_energy_curve(summary["total_energies"], lambda_values)
        curve_ax.plot(
            lambda_values,
            curve,
            color=ensemble["color"],
            linewidth=2.2,
            label=f"{summary['name']} | F({lam:.2f})={stable_free_energy(summary['total_energies'], lam):.3f}",
        )
        current_free_energy = stable_free_energy(summary["total_energies"], lam)
        curve_ax.scatter([lam], [current_free_energy], color=ensemble["color"], s=42, zorder=4)

    curve_ax.axvline(float(lam), color="#111827", linestyle="--", linewidth=1.1, alpha=0.7)
    curve_ax.text(
        0.02,
        0.98,
        f"Current lambda = {lam:.2f}\nSweep: [{lambda_min:.2f}, {lambda_max:.2f}]",
        transform=curve_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#94a3b8"},
    )
    curve_ax.set_title("Free Energy Evolution")
    curve_ax.set_xlabel("lambda")
    curve_ax.set_ylabel("Free energy F")
    curve_ax.grid(True, alpha=0.25)
    curve_ax.legend(loc="best", fontsize=8)

    all_points = np.vstack(
        [payload["start"], payload["end"]]
        + [np.asarray(path, dtype=float) for ensemble in payload["ensembles"] for path in ensemble["paths"]]
    )
    x_pad = 0.45
    y_pad = 0.45
    x_limits = (float(np.min(all_points[:, 0]) - x_pad), float(np.max(all_points[:, 0]) + x_pad))
    y_limits = (float(np.min(all_points[:, 1]) - y_pad), float(np.max(all_points[:, 1]) + y_pad))
    for ax in axes[:2]:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    axes[2].set_xlim(*x_limits)
    axes[2].set_ylim(*y_limits)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D demo of segment penalties and free energy on similar vs diverse path ensembles."
    )
    parser.add_argument(
        "--lambda-value",
        dest="lambda_value",
        type=float,
        default=1.0,
        help="Lambda used in F = -(1/lambda) log(sum(exp(-lambda * E))). Default: 1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lam = max(1e-9, float(args.lambda_value))
    payload = build_ensembles()
    print_summary(payload, lam)
    plot_demo(payload, lam, args.output)


if __name__ == "__main__":
    main()
