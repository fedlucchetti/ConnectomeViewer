#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
SHOW_FIGURES_REQUESTED = "--show_figures" in sys.argv
if not SHOW_FIGURES_REQUESTED:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import plotting
from scipy import stats


########## LOCAL HELPERS ##########
VIEWER_ROOT = Path(__file__).resolve().parents[1]
TARGET_HELPER_RELATIVE = Path("experiments") / "nbs_metsim_rc_overlap" / "analyze_nbs_metsim_rc_overlap.py"


def _candidate_toolbox_roots() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(root: Path | None) -> None:
        if root is None:
            return
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            resolved = root.expanduser()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    add_candidate(VIEWER_ROOT / "mrsitoolbox")
    add_candidate(VIEWER_ROOT.parent / "mrsitoolbox")

    devanalyse = str(os.getenv("DEVANALYSEPATH") or "").strip()
    if devanalyse:
        dev_root = Path(devanalyse)
        add_candidate(dev_root / "mrsitoolbox")
        add_candidate(dev_root)

    try:
        import mrsitoolbox  # type: ignore
    except Exception:
        pass
    else:
        module_path = Path(getattr(mrsitoolbox, "__file__", "")).resolve()
        add_candidate(module_path.parent.parent)

    return candidates


def _resolve_toolbox_root_and_helper() -> tuple[Path, Path]:
    roots = _candidate_toolbox_roots()
    for toolbox_root in roots:
        helper_path = toolbox_root / TARGET_HELPER_RELATIVE
        if helper_path.exists():
            return toolbox_root, helper_path
    searched = "\n".join(str(root / TARGET_HELPER_RELATIVE) for root in roots)
    raise FileNotFoundError(
        "Could not find analyze_nbs_metsim_rc_overlap.py. Tried:\n"
        f"{searched}"
    )


MRSITOOLBOX_ROOT, OVERLAP_MODULE_PATH = _resolve_toolbox_root_and_helper()
if str(MRSITOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(MRSITOOLBOX_ROOT))

OVERLAP_SPEC = importlib.util.spec_from_file_location("nbs_metsim_rc_overlap_helpers", OVERLAP_MODULE_PATH)
if OVERLAP_SPEC is None or OVERLAP_SPEC.loader is None:
    raise ImportError(f"Could not load helper module from {OVERLAP_MODULE_PATH}")
OVERLAP_MODULE = importlib.util.module_from_spec(OVERLAP_SPEC)
sys.modules[OVERLAP_SPEC.name] = OVERLAP_MODULE
OVERLAP_SPEC.loader.exec_module(OVERLAP_MODULE)

MatrixBundle = OVERLAP_MODULE.MatrixBundle
load_matrix_bundle = OVERLAP_MODULE.load_matrix_bundle
extract_selected_nbs_mask = OVERLAP_MODULE.extract_selected_nbs_mask
build_nbs_edge_table = OVERLAP_MODULE.build_nbs_edge_table
parse_nuisance_from_filename = OVERLAP_MODULE.parse_nuisance_from_filename
parse_t_threshold_from_filename = OVERLAP_MODULE.parse_t_threshold_from_filename
default_output_subdir = OVERLAP_MODULE.default_output_subdir
infer_project_names = OVERLAP_MODULE.infer_project_names
discover_bids_root = OVERLAP_MODULE.discover_bids_root
resolve_structural_path = OVERLAP_MODULE.resolve_structural_path
align_nbs_and_structural_parcels = OVERLAP_MODULE.align_nbs_and_structural_parcels
align_bundle_to_pairs = OVERLAP_MODULE.align_bundle_to_pairs
print_section = OVERLAP_MODULE.print_section
info = OVERLAP_MODULE.info
_as_int = OVERLAP_MODULE._as_int
_as_str = OVERLAP_MODULE._as_str
_safe_name_array = OVERLAP_MODULE._safe_name_array
_covars_to_df = OVERLAP_MODULE._covars_to_df
_ensure_pair_columns = OVERLAP_MODULE._ensure_pair_columns
_pair_keys = OVERLAP_MODULE._pair_keys
_resolve_column = OVERLAP_MODULE._resolve_column
_symmetrize_matrix = OVERLAP_MODULE._symmetrize_matrix
_json_ready = OVERLAP_MODULE._json_ready
NetBasedAnalysis = OVERLAP_MODULE.NetBasedAnalysis
RICHCLUB_DENSITIES = OVERLAP_MODULE.RICHCLUB_DENSITIES

from tools.debug import Debug


########## STYLE ##########
FONTSIZE = 20
TICK_FONTSIZE = FONTSIZE - 2
LINEWIDTH = 2.5
SCATTER_POINT_SIZE = 55
VIOLIN_WIDTH = 0.75
BAR_WIDTH = 0.65
HUB_THRESHOLDS = [1, 5, 10, 20]
EDGE_CLASS_THRESHOLDS = [2, 5, 10, 20]
MAIN_HUB_THRESHOLD = 10
CONTROL_VALUE = 0
DEFAULT_RICHCLUB_DENSITY = float(RICHCLUB_DENSITIES[min(2, len(RICHCLUB_DENSITIES) - 1)]) if len(RICHCLUB_DENSITIES) else 0.10
RICHCLUB_NUM_RANDOM = 100
RICHCLUB_ALPHA = 0.05
RICHCLUB_N_JOBS = 16
RICHCLUB_K_TICK_STEP = 10

BRAIN_EDGE_CMAP = "Reds"
BRAIN_NODE_CMAP = "viridis"
BRAIN_NODE_COLOR = "#d95f02"
SCATTER_COLOR = "#2c7fb8"
SCATTER_LINE_COLOR = "#253494"
NULL_BAND_COLOR = "#bdbdbd"
NBS_GROUP_COLORS = {
    "NBS nodes": "#c44e52",
    "Non-NBS nodes": "#4c72b0",
}
HUB_THRESHOLD_COLORS = {
    1: "#111111",
    5: "#1b9e77",
    10: "#d95f02",
    20: "#7570b3",
}
EDGE_CLASS_COLORS = {
    "hub-hub": "#c44e52",
    "feeder": "#dd8452",
    "local": "#55a868",
}

debug = Debug()
sns.set_theme(style="whitegrid", context="talk")


########## GENERIC HELPERS ##########
def _save_matplotlib_figure(fig: plt.Figure, output_stem: Path) -> dict[str, str]:
    output_paths = {}
    for suffix in (".png", ".pdf"):
        outpath = output_stem.with_suffix(suffix)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        output_paths[suffix.lstrip(".")] = str(outpath)
    plt.close(fig)
    return output_paths


def _save_nilearn_display(save_callback, output_stem: Path) -> dict[str, str]:
    output_paths = {}
    for suffix in (".png", ".pdf"):
        outpath = output_stem.with_suffix(suffix)
        save_callback(str(outpath))
        output_paths[suffix.lstrip(".")] = str(outpath)
    return output_paths


def _log_figure_outputs(figure_outputs: dict[str, dict[str, str]]) -> None:
    for figure_name, output_map in figure_outputs.items():
        for fmt, path in output_map.items():
            info(f"Saved figure {figure_name} ({fmt}) to {path}")


def _compose_subplot_panel_figure(
    figure_outputs: dict[str, dict[str, str]],
    output_stem: Path,
    show: bool = False,
) -> dict[str, str]:
    panel_specs = [
        ("A", "metabolic_nbs_subnetwork", "Metabolic NBS subnetwork", (0, slice(0, 3))),
        ("B", "control_structural_strength", "Control structural node strength", (0, slice(3, 6))),
        ("C", "control_richclub_curve", "Control rich-club curve", (1, slice(0, 6))),
        ("D", "strength_violin", "Structural strength in NBS vs non-NBS nodes", (2, slice(0, 2))),
        ("E", "hub_enrichment", "Hub enrichment across thresholds", (2, slice(2, 4))),
        ("F", "edge_class_enrichment", "Structural class of NBS edges", (2, slice(4, 6))),
    ]

    def _draw_panel_figure(fig: plt.Figure) -> None:
        grid = fig.add_gridspec(3, 6, hspace=0.18, wspace=0.18)
        for panel_letter, key, title, (row_idx, col_slice) in panel_specs:
            ax = fig.add_subplot(grid[row_idx, col_slice])
            png_path = figure_outputs.get(key, {}).get("png")
            if png_path is None or not Path(png_path).exists():
                ax.axis("off")
                ax.text(0.5, 0.5, f"Missing panel {panel_letter}", ha="center", va="center", fontsize=FONTSIZE)
                continue
            image = plt.imread(png_path)
            ax.imshow(image)
            ax.set_title(f"{panel_letter}. {title}", fontsize=FONTSIZE, pad=10)
            ax.axis("off")

    fig = plt.figure(figsize=(24, 18))
    _draw_panel_figure(fig)
    output_paths = _save_matplotlib_figure(fig, output_stem)
    if show:
        fig_show = plt.figure(figsize=(24, 18))
        _draw_panel_figure(fig_show)
        plt.show()
        plt.close(fig_show)
    return output_paths


def _scale_values(values: np.ndarray, min_size: float, max_size: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, min_size, dtype=float)
    vmin = np.nanmin(values[finite])
    vmax = np.nanmax(values[finite])
    if np.isclose(vmin, vmax):
        return np.full(values.shape, 0.5 * (min_size + max_size), dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)


def _coerce_diag_series(df: pd.DataFrame) -> pd.Series:
    diag_col = _resolve_column(df, "Diag", "diag")
    if diag_col is None:
        raise ValueError("Could not find a diag column in the aligned covariates dataframe.")
    return pd.to_numeric(df[diag_col], errors="coerce")


def _compute_node_strength(matrix: np.ndarray) -> np.ndarray:
    matrix = _symmetrize_matrix(matrix)
    return np.asarray(matrix.sum(axis=1), dtype=float)


def _compute_nbs_node_load(nbs_mask: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(nbs_mask, dtype=bool).sum(axis=1), dtype=int)


def _sparse_integer_ticks(values: np.ndarray, step: int) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if values.size == 0:
        return np.array([], dtype=int)
    if step <= 1 or values.size <= 10:
        return values
    tick_values = np.arange(int(values.min()), int(values.max()) + 1, int(step), dtype=int)
    tick_values = np.unique(np.concatenate([tick_values, [int(values.min()), int(values.max())]]))
    return tick_values


def _top_percent_hub_mask(strength: np.ndarray, percent: float) -> np.ndarray:
    strength = np.asarray(strength, dtype=float)
    n_nodes = strength.size
    n_hubs = max(1, int(np.ceil(n_nodes * (float(percent) / 100.0))))
    order = np.argsort(-strength, kind="mergesort")
    hub_mask = np.zeros(n_nodes, dtype=bool)
    hub_mask[order[:n_hubs]] = True
    return hub_mask


def _empirical_pvalue(observed: float, null_values: np.ndarray, tail: str = "greater") -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0:
        return np.nan
    if tail == "greater":
        count = np.sum(null_values >= observed)
    elif tail == "less":
        count = np.sum(null_values <= observed)
    else:
        count = np.sum(np.abs(null_values) >= abs(observed))
    return float((count + 1) / (null_values.size + 1))


def _significance_marker(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _select_control_average_structural(
    aligned_struct: MatrixBundle,
    covars_df: pd.DataFrame,
    control_value: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    diag_series = _coerce_diag_series(covars_df)
    control_mask = diag_series == int(control_value)
    if not np.any(control_mask):
        raise ValueError(f"No control rows found with diag == {control_value}.")
    control_covars = covars_df.loc[control_mask].reset_index(drop=True)
    control_matrices = np.asarray(aligned_struct.matrices[control_mask.to_numpy()], dtype=float)
    if control_matrices.size == 0:
        raise ValueError("No structural matrices remained after selecting controls.")
    average_matrix = _symmetrize_matrix(np.nanmean(control_matrices, axis=0))
    return average_matrix, control_covars


def _spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    rho, p_standard = stats.spearmanr(x, y)
    null_rhos = np.empty(int(num_permutations), dtype=float)
    for idx in range(int(num_permutations)):
        null_rhos[idx] = stats.spearmanr(x, rng.permutation(y)).statistic
    p_empirical = _empirical_pvalue(float(rho), null_rhos, tail="two-sided")
    return float(rho), float(p_standard), float(p_empirical)


def _mannwhitney_with_effect(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> tuple[float, float, float]:
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    group_a = group_a[np.isfinite(group_a)]
    group_b = group_b[np.isfinite(group_b)]
    test = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    n1 = len(group_a)
    n2 = len(group_b)
    rank_biserial = (2.0 * test.statistic / (n1 * n2)) - 1.0 if n1 > 0 and n2 > 0 else np.nan
    return float(test.statistic), float(test.pvalue), float(rank_biserial)


def _compute_hub_enrichment(
    nbs_node_mask: np.ndarray,
    node_strength: np.ndarray,
    thresholds: list[int],
    num_permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_nodes = len(node_strength)
    nbs_nodes = np.where(nbs_node_mask)[0]
    n_nbs_nodes = len(nbs_nodes)
    all_nodes = np.arange(n_nodes, dtype=int)
    records = []

    for threshold in thresholds:
        hub_mask = _top_percent_hub_mask(node_strength, threshold)
        hub_indices = np.where(hub_mask)[0]
        observed_overlap = int(np.sum(nbs_node_mask[hub_indices]))
        null_overlaps = np.empty(int(num_permutations), dtype=float)
        for idx in range(int(num_permutations)):
            sampled = rng.choice(all_nodes, size=n_nbs_nodes, replace=False)
            null_overlaps[idx] = int(np.sum(hub_mask[sampled]))
        mean_null = float(np.mean(null_overlaps))
        lower_null = float(np.percentile(null_overlaps, 2.5))
        upper_null = float(np.percentile(null_overlaps, 97.5))
        enrichment_ratio = float(observed_overlap / mean_null) if mean_null > 0 else np.nan
        p_empirical = _empirical_pvalue(float(observed_overlap), null_overlaps, tail="greater")
        records.append(
            {
                "hub_threshold_percent": int(threshold),
                "n_hubs": int(len(hub_indices)),
                "hub_node_indices": json.dumps(_json_ready(hub_indices.tolist())),
                "observed_overlap": int(observed_overlap),
                "observed_overlap_fraction": float(observed_overlap / n_nbs_nodes) if n_nbs_nodes else np.nan,
                "mean_null_overlap": mean_null,
                "lower_null_overlap": lower_null,
                "upper_null_overlap": upper_null,
                "enrichment_ratio": enrichment_ratio,
                "lower_null_ratio": float(lower_null / mean_null) if mean_null > 0 else np.nan,
                "upper_null_ratio": float(upper_null / mean_null) if mean_null > 0 else np.nan,
                "p_empirical_greater": p_empirical,
            }
        )
    return pd.DataFrame.from_records(records)


def _classify_edge_classes(
    edge_indices: np.ndarray,
    hub_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    class_names = []
    counts = {"hub-hub": 0, "feeder": 0, "local": 0}
    for i_idx, j_idx in edge_indices.tolist():
        if hub_mask[i_idx] and hub_mask[j_idx]:
            class_name = "hub-hub"
        elif hub_mask[i_idx] or hub_mask[j_idx]:
            class_name = "feeder"
        else:
            class_name = "local"
        class_names.append({"node_i": int(i_idx), "node_j": int(j_idx), "edge_class": class_name})
        counts[class_name] += 1
    return pd.DataFrame.from_records(class_names), counts


def _compute_edge_class_enrichment(
    nbs_mask: np.ndarray,
    hub_mask: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_nodes = hub_mask.size
    edge_rows, edge_cols = np.where(np.triu(np.asarray(nbs_mask, dtype=bool), k=1))
    nbs_edges = np.column_stack([edge_rows, edge_cols]).astype(int)
    if nbs_edges.size == 0:
        raise ValueError("NBS mask contains no edges, cannot compute structural edge classes.")

    edge_class_df, observed_counts = _classify_edge_classes(nbs_edges, hub_mask)

    all_edge_rows, all_edge_cols = np.triu_indices(n_nodes, k=1)
    all_edges = np.column_stack([all_edge_rows, all_edge_cols]).astype(int)
    class_codes = np.zeros(all_edges.shape[0], dtype=int)
    both_hubs = hub_mask[all_edges[:, 0]] & hub_mask[all_edges[:, 1]]
    one_hub = hub_mask[all_edges[:, 0]] ^ hub_mask[all_edges[:, 1]]
    class_codes[both_hubs] = 0
    class_codes[one_hub] = 1
    class_codes[~(both_hubs | one_hub)] = 2
    class_order = ["hub-hub", "feeder", "local"]

    null_counts = np.empty((int(num_permutations), 3), dtype=float)
    for idx in range(int(num_permutations)):
        sampled_idx = rng.choice(all_edges.shape[0], size=nbs_edges.shape[0], replace=False)
        counts = np.bincount(class_codes[sampled_idx], minlength=3)
        null_counts[idx, :] = counts.astype(float)

    records = []
    for class_idx, class_name in enumerate(class_order):
        observed = int(observed_counts[class_name])
        mean_null = float(np.mean(null_counts[:, class_idx]))
        lower_null = float(np.percentile(null_counts[:, class_idx], 2.5))
        upper_null = float(np.percentile(null_counts[:, class_idx], 97.5))
        enrichment_ratio = float(observed / mean_null) if mean_null > 0 else np.nan
        p_empirical = _empirical_pvalue(float(observed), null_counts[:, class_idx], tail="greater")
        records.append(
            {
                "edge_class": class_name,
                "observed_count": observed,
                "mean_null_count": mean_null,
                "lower_null_count": lower_null,
                "upper_null_count": upper_null,
                "enrichment_ratio": enrichment_ratio,
                "lower_null_ratio": float(lower_null / mean_null) if mean_null > 0 else np.nan,
                "upper_null_ratio": float(upper_null / mean_null) if mean_null > 0 else np.nan,
                "p_empirical_greater": p_empirical,
            }
        )

    enrichment_df = pd.DataFrame.from_records(records)
    return edge_class_df, enrichment_df


def compute_control_richclub_curve(
    control_average_matrix: np.ndarray,
    density: float,
    num_random: int,
    alpha: float,
    n_jobs: int,
) -> pd.DataFrame:
    nba = NetBasedAnalysis()
    matrix = _symmetrize_matrix(control_average_matrix)
    density = float(density)
    adj = nba.binarize(
        matrix,
        threshold=density,
        mode="abs",
        threshold_mode="density",
        binarize=True,
    )
    adj = np.asarray(adj > 0, dtype=int)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    degrees, rc_coefficients, rand_params = nba.compute_richclub_stats(
        adj,
        num_random=int(num_random),
        alpha=float(alpha),
        null_model="random",
        edge_density=density,
        n_jobs=int(n_jobs),
    )
    degrees = np.asarray(degrees, dtype=int)
    rc_coefficients = np.asarray(rc_coefficients, dtype=float)
    median_rand_rc = np.asarray(rand_params.get("median", []), dtype=float)
    lower_rand_rc = np.asarray(rand_params.get("lower", []), dtype=float)
    upper_rand_rc = np.asarray(rand_params.get("upper", []), dtype=float)
    p_values = np.asarray(rand_params.get("pvalue", []), dtype=float)
    node_degree = np.asarray(nba.get_degree_per_node(adj), dtype=int)

    n_possible_edges = adj.shape[0] * max(adj.shape[0] - 1, 0) / 2.0
    actual_density = float(np.triu(adj, k=1).sum() / n_possible_edges) if n_possible_edges > 0 else np.nan
    records: list[dict[str, object]] = []
    for idx, k in enumerate(degrees.tolist()):
        hub_node_count = int(np.sum(node_degree >= int(k)))
        p_value = float(p_values[idx]) if idx < len(p_values) and np.isfinite(p_values[idx]) else np.nan
        records.append(
            {
                "k": int(k),
                "richclub_coefficient": float(rc_coefficients[idx]) if idx < len(rc_coefficients) else np.nan,
                "median_random_rc": float(median_rand_rc[idx]) if idx < len(median_rand_rc) and np.isfinite(median_rand_rc[idx]) else np.nan,
                "lower_random_rc": float(lower_rand_rc[idx]) if idx < len(lower_rand_rc) and np.isfinite(lower_rand_rc[idx]) else np.nan,
                "upper_random_rc": float(upper_rand_rc[idx]) if idx < len(upper_rand_rc) and np.isfinite(upper_rand_rc[idx]) else np.nan,
                "p_value": p_value,
                "is_significant_k": int(np.isfinite(p_value) and p_value <= float(alpha)),
                "hub_node_count": hub_node_count,
                "requested_density": density,
                "requested_density_percent": float(100.0 * density),
                "actual_density": actual_density,
                "actual_density_percent": float(100.0 * actual_density) if np.isfinite(actual_density) else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


########## PLOTTING ##########
def plot_metabolic_nbs_subnetwork(
    nbs_mask: np.ndarray,
    t_matrix: np.ndarray | None,
    centroids_world: np.ndarray,
    node_load: np.ndarray,
    output_stem: Path,
) -> dict[str, str]:
    active_nodes = np.where(node_load > 0)[0]
    if active_nodes.size == 0:
        raise ValueError("No NBS nodes were found for the metabolic NBS subnetwork plot.")

    edge_weights = np.asarray(nbs_mask, dtype=float)
    if t_matrix is not None and t_matrix.shape == nbs_mask.shape:
        edge_weights = np.abs(np.asarray(t_matrix, dtype=float)) * np.asarray(nbs_mask, dtype=float)

    edge_weights = edge_weights[np.ix_(active_nodes, active_nodes)]
    coords = np.asarray(centroids_world, dtype=float)[active_nodes]
    node_sizes = _scale_values(node_load[active_nodes], min_size=35.0, max_size=160.0)

    def _save(outpath: str) -> None:
        display = plotting.plot_connectome(
            adjacency_matrix=edge_weights,
            node_coords=coords,
            node_color=BRAIN_NODE_COLOR,
            node_size=node_sizes,
            edge_cmap=BRAIN_EDGE_CMAP,
            edge_vmin=float(np.nanmin(edge_weights[edge_weights > 0])) if np.any(edge_weights > 0) else 0.0,
            edge_vmax=float(np.nanmax(edge_weights)) if np.any(edge_weights > 0) else 1.0,
            display_mode="lyrz",
            black_bg=False,
            annotate=False,
            title="Metabolic NBS subnetwork",
            colorbar=True,
            edge_kwargs={"linewidth": LINEWIDTH},
            output_file=outpath,
        )
        if display is not None:
            try:
                display.close()
            except Exception:
                pass

    return _save_nilearn_display(_save, output_stem)


def plot_structural_strength_map(
    node_strength: np.ndarray,
    centroids_world: np.ndarray,
    hub_mask: np.ndarray,
    output_stem: Path,
) -> dict[str, str]:
    coords = np.asarray(centroids_world, dtype=float)
    sizes = _scale_values(node_strength, min_size=30.0, max_size=130.0)
    sizes = np.where(hub_mask, sizes * 1.2, sizes)

    def _save(outpath: str) -> None:
        display = plotting.plot_markers(
            node_values=node_strength,
            node_coords=coords,
            node_size=sizes,
            node_cmap=BRAIN_NODE_CMAP,
            alpha=0.95,
            display_mode="lyrz",
            title="Control-average structural node strength",
            annotate=False,
            black_bg=False,
            colorbar=True,
            node_kwargs={"edgecolors": "black", "linewidths": 0.35},
            output_file=outpath,
        )
        if display is not None:
            try:
                display.close()
            except Exception:
                pass

    return _save_nilearn_display(_save, output_stem)


def plot_control_richclub_curve(
    richclub_curve_df: pd.DataFrame,
    output_stem: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(18.0, 4.8))
    plot_df = richclub_curve_df.sort_values("k").reset_index(drop=True)
    x = plot_df["k"].to_numpy(dtype=float)
    y = plot_df["richclub_coefficient"].to_numpy(dtype=float)
    null_low = plot_df["lower_random_rc"].to_numpy(dtype=float)
    null_med = plot_df["median_random_rc"].to_numpy(dtype=float)
    null_high = plot_df["upper_random_rc"].to_numpy(dtype=float)
    significant_mask = plot_df["is_significant_k"].to_numpy(dtype=int) > 0

    ax.fill_between(
        x,
        null_low,
        null_high,
        color=NULL_BAND_COLOR,
        alpha=0.45,
        label="Null 95% interval",
    )
    ax.plot(
        x,
        null_med,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Null median",
    )
    ax.plot(
        x,
        y,
        color=SCATTER_LINE_COLOR,
        linewidth=LINEWIDTH,
        label="Control network",
    )
    if np.any(significant_mask):
        ax.scatter(
            x[significant_mask],
            y[significant_mask],
            color=SCATTER_LINE_COLOR,
            s=100,
            zorder=3,
            label="RC-significant k",
        )
    ax.set_xlabel("k", fontsize=FONTSIZE)
    ax.set_ylabel("Rich-club coefficient", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    tick_values = _sparse_integer_ticks(np.rint(x).astype(int), step=RICHCLUB_K_TICK_STEP)
    ax.set_xticks(tick_values)
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


def plot_strength_violin(
    df_strength_groups: pd.DataFrame,
    p_value: float,
    rank_biserial: float,
    output_stem: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.violinplot(
        data=df_strength_groups,
        x="group",
        y="control_structural_strength",
        hue="group",
        palette=NBS_GROUP_COLORS,
        inner=None,
        cut=0,
        linewidth=1.1,
        width=VIOLIN_WIDTH,
        dodge=False,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.boxplot(
        data=df_strength_groups,
        x="group",
        y="control_structural_strength",
        width=0.25,
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.5},
        ax=ax,
    )
    sns.stripplot(
        data=df_strength_groups,
        x="group",
        y="control_structural_strength",
        color="black",
        alpha=0.55,
        size=4,
        jitter=0.16,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Control-average structural node strength", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ymax = float(df_strength_groups["control_structural_strength"].max())
    ax.text(
        0.5,
        ymax * 1.02,
        f"Mann-Whitney P = {p_value:.3g}\nRank-biserial = {rank_biserial:.2f}",
        ha="center",
        va="bottom",
        fontsize=TICK_FONTSIZE,
    )
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


def plot_hub_enrichment(
    enrichment_df: pd.DataFrame,
    output_stem: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    x = enrichment_df["hub_threshold_percent"].to_numpy(dtype=float)
    y = enrichment_df["enrichment_ratio"].to_numpy(dtype=float)
    low = enrichment_df["lower_null_ratio"].to_numpy(dtype=float)
    high = enrichment_df["upper_null_ratio"].to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=NULL_BAND_COLOR, alpha=0.35, label="Null 95% interval")
    ax.plot(x, y, color="black", linewidth=LINEWIDTH, zorder=2)
    for _, row in enrichment_df.iterrows():
        threshold = int(row["hub_threshold_percent"])
        color = HUB_THRESHOLD_COLORS.get(threshold, "black")
        ax.scatter(
            row["hub_threshold_percent"],
            row["enrichment_ratio"],
            color=color,
            s=100,
            zorder=3,
        )
        marker = _significance_marker(float(row["p_empirical_greater"]))
        if marker:
            ax.text(
                row["hub_threshold_percent"],
                row["enrichment_ratio"] + 0.08,
                marker,
                ha="center",
                va="bottom",
                fontsize=FONTSIZE,
            )
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.4)
    ax.set_xlabel("Structural hub threshold (%)", fontsize=FONTSIZE)
    ax.set_ylabel("NBS-node enrichment ratio", fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


def plot_edge_class_enrichment(
    enrichment_df: pd.DataFrame,
    thresholds: list[int],
    output_stem: Path,
) -> dict[str, str]:
    class_order = ["hub-hub", "feeder", "local"]
    thresholds = [int(val) for val in thresholds]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    threshold_order = [threshold for threshold in thresholds if threshold in enrichment_df["hub_threshold_percent"].unique()]
    plot_df = enrichment_df.copy()
    plot_df["edge_class"] = pd.Categorical(plot_df["edge_class"], categories=class_order, ordered=True)
    plot_df["hub_threshold_percent"] = pd.Categorical(
        plot_df["hub_threshold_percent"],
        categories=threshold_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["edge_class", "hub_threshold_percent"]).reset_index(drop=True)
    max_y = float(
        np.nanmax(
            np.concatenate(
                [
                    plot_df["enrichment_ratio"].to_numpy(dtype=float),
                    plot_df["upper_null_ratio"].to_numpy(dtype=float),
                ]
            )
        )
    )
    threshold_palette = {threshold: HUB_THRESHOLD_COLORS.get(int(threshold), "black") for threshold in threshold_order}
    x_base = np.arange(len(class_order), dtype=float)
    n_thresholds = max(1, len(threshold_order))
    group_width = 0.8
    bar_width = group_width / n_thresholds

    for offset_idx, threshold in enumerate(threshold_order):
        subset = (
            plot_df[plot_df["hub_threshold_percent"] == int(threshold)]
            .set_index("edge_class")
            .reindex(class_order)
            .reset_index()
        )
        x_positions = x_base - 0.5 * group_width + (offset_idx + 0.5) * bar_width
        bars = ax.bar(
            x_positions,
            subset["enrichment_ratio"].to_numpy(dtype=float),
            width=bar_width * 0.92,
            color=threshold_palette[threshold],
            label=f"Top {int(threshold)}%",
            zorder=2,
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            x_center = bar.get_x() + 0.5 * bar.get_width()
            ax.vlines(
                x_center,
                row["lower_null_ratio"],
                row["upper_null_ratio"],
                color="black",
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )
            marker = _significance_marker(float(row["p_empirical_greater"]))
            if marker:
                ax.text(
                    x_center,
                    row["enrichment_ratio"] + 0.08,
                    marker,
                    ha="center",
                    va="bottom",
                    fontsize=FONTSIZE,
                )

    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.4)
    ax.set_xlabel("")
    ax.set_ylabel("Edge-class enrichment ratio", fontsize=FONTSIZE)
    ax.set_ylim(0, max(1.15, max_y * 1.18))
    ax.set_xticks(x_base)
    ax.set_xticklabels(class_order)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=12)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.legend(title="Hub threshold", fontsize=TICK_FONTSIZE, title_fontsize=TICK_FONTSIZE)
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


########## ARGUMENTS ##########
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the structural context of an NBS metabolic subnetwork using the control-average structural connectivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_nbs", required=True, help="Path to the NBS components_all NPZ.")
    parser.add_argument("--input_struct", default=None, help="Optional explicit path to the structural group NPZ.")
    parser.add_argument("--bids_dir", default=None, help="Optional BIDS root used to resolve the structural NPZ.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory. Defaults next to the NBS file.")
    parser.add_argument("--component", type=int, default=None, help="Optional NBS component index. Default uses the union of significant components.")
    parser.add_argument("--strict_parcels", action="store_true", help="Require the structural NPZ to contain every NBS parcel label.")
    parser.add_argument("--control_value", type=int, default=CONTROL_VALUE, help="Value of diag used to define controls.")
    parser.add_argument("--hub_thresholds", type=int, nargs="+", default=HUB_THRESHOLDS, help="Hub thresholds in percent for enrichment analyses.")
    parser.add_argument("--edge_class_thresholds", type=int, nargs="+", default=EDGE_CLASS_THRESHOLDS, help="Structural hub thresholds in percent used in panel F for edge-class enrichment.")
    parser.add_argument("--main_hub_threshold", type=int, default=MAIN_HUB_THRESHOLD, help="Main structural hub threshold in percent used for hub highlighting on the structural strength map.")
    parser.add_argument("--num_permutations", type=int, default=5000, help="Number of random permutations used for node and edge null models.")
    parser.add_argument("--richclub_density", type=float, default=DEFAULT_RICHCLUB_DENSITY, help="Density used to binarize the control structural graph for panel C.")
    parser.add_argument("--richclub_num_random", type=int, default=RICHCLUB_NUM_RANDOM, help="Number of random graphs used per density by compute_richclub_stats in panel C.")
    parser.add_argument("--richclub_alpha", type=float, default=RICHCLUB_ALPHA, help="Alpha used for the panel C null interval and significant-k flagging.")
    parser.add_argument("--richclub_n_jobs", type=int, default=RICHCLUB_N_JOBS, help="Worker processes passed to compute_richclub_stats in panel C.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation-based analyses.")
    parser.add_argument("--show_figures", action="store_true", help="Display the combined subplot figure with matplotlib.pyplot.show() after saving.")
    return parser


########## MAIN ##########
def main() -> None:
    ########## PARSE ARGUMENTS ##########
    args = build_argument_parser().parse_args()
    input_nbs = Path(args.input_nbs).expanduser().resolve()
    if not input_nbs.exists():
        raise FileNotFoundError(f"NBS input not found: {input_nbs}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_subdir(input_nbs, "nbs_control_structural_context")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    ########## LOAD NBS INPUT ##########
    print_section("LOAD NBS INPUT")
    nbs_data = np.load(input_nbs, allow_pickle=True)
    nbs_mask, nbs_selection_meta = extract_selected_nbs_mask(nbs_data, args.component)
    nbs_mask = np.asarray(nbs_mask, dtype=bool)
    np.fill_diagonal(nbs_mask, False)
    nbs_mask = np.triu(nbs_mask, k=1)
    nbs_mask = nbs_mask | nbs_mask.T

    parcel_labels = np.asarray(nbs_data["parcel_labels"])
    parcel_names = _safe_name_array(nbs_data["parcel_names"] if "parcel_names" in nbs_data.files else None, parcel_labels)
    centroids_world = np.asarray(nbs_data["centroids_world"], dtype=float)
    centroid_by_label = {
        int(label): centroids_world[idx]
        for idx, label in enumerate(parcel_labels.tolist())
    }
    subject_ids = np.asarray(nbs_data["subject_ids"]).astype(str)
    session_ids = np.asarray(nbs_data["session_ids"]).astype(str)
    covars_df = _covars_to_df(nbs_data["covars"] if "covars" in nbs_data.files else None)
    covars_df = _ensure_pair_columns(covars_df, subject_ids, session_ids)
    t_matrix = np.asarray(nbs_data["t_matrix"], dtype=float) if "t_matrix" in nbs_data.files else None
    target_pairs = _pair_keys(subject_ids, session_ids)

    nbs_summary = {
        "input_nbs": str(input_nbs),
        "group_from_metadata": _as_str(nbs_data["group"]) if "group" in nbs_data.files else "",
        "parc_scheme": _as_str(nbs_data["parc_scheme"]) if "parc_scheme" in nbs_data.files else "",
        "scale": _as_int(nbs_data["scale"], default=0),
        "regressor_name": _as_str(nbs_data["regressor_name"]) if "regressor_name" in nbs_data.files else "",
        "nuisance_terms": parse_nuisance_from_filename(input_nbs),
        "n_subject_session_pairs": len(target_pairs),
        "n_nodes": int(nbs_mask.shape[0]),
        "n_nbs_edges": int(np.triu(nbs_mask, k=1).sum()),
        "selection": nbs_selection_meta,
    }
    info(json.dumps(nbs_summary, indent=2))

    ########## RESOLVE STRUCTURAL INPUT ##########
    print_section("RESOLVE STRUCTURAL INPUT")
    stored_connectivity_path = _as_str(nbs_data["connectivity_path"]) if "connectivity_path" in nbs_data.files else ""
    bids_root = discover_bids_root(
        project_names=infer_project_names(input_nbs, nbs_data),
        bids_dir=args.bids_dir,
        connectivity_hint=stored_connectivity_path,
    )
    struct_path = resolve_structural_path(input_nbs, nbs_data, args.input_struct, bids_root)
    info(f"BIDS root: {bids_root if bids_root is not None else 'not found'}")
    info(f"Resolved structural path: {struct_path}")

    ########## LOAD STRUCTURAL INPUT ##########
    print_section("LOAD STRUCTURAL INPUT")
    structural_bundle = load_matrix_bundle(struct_path)
    info(
        f"Structural matrices: {structural_bundle.matrices.shape[0]} subjects, "
        f"{structural_bundle.matrices.shape[1]} x {structural_bundle.matrices.shape[2]} nodes"
    )

    ########## ALIGN STRUCTURAL INPUT TO NBS ##########
    print_section("ALIGN STRUCTURAL INPUT TO NBS")
    (
        structural_bundle,
        parcel_labels,
        parcel_names,
        nbs_mask,
        t_matrix,
        parcel_alignment_meta,
    ) = align_nbs_and_structural_parcels(
        structural_bundle=structural_bundle,
        nbs_labels=parcel_labels,
        nbs_names=parcel_names,
        nbs_mask=nbs_mask,
        t_matrix=t_matrix,
        strict=bool(args.strict_parcels),
    )
    aligned_struct, struct_indices, kept_target_indices, missing_pairs = align_bundle_to_pairs(
        structural_bundle,
        target_pairs,
    )
    covars_df = covars_df.iloc[kept_target_indices].reset_index(drop=True)
    centroids_world = np.asarray([centroid_by_label[int(label)] for label in parcel_labels.tolist()], dtype=float)
    info(json.dumps(parcel_alignment_meta, indent=2))
    if missing_pairs:
        info(f"Matched {len(kept_target_indices)} of {len(target_pairs)} NBS subject-session pair(s).")

    ########## BUILD CONTROL-AVERAGE STRUCTURAL CONNECTOME ##########
    print_section("BUILD CONTROL-AVERAGE STRUCTURAL CONNECTOME")
    control_average_matrix, control_covars = _select_control_average_structural(
        aligned_struct=aligned_struct,
        covars_df=covars_df,
        control_value=int(args.control_value),
    )
    node_strength = _compute_node_strength(control_average_matrix)
    nbs_node_load = _compute_nbs_node_load(nbs_mask)
    nbs_node_mask = nbs_node_load > 0
    main_hub_mask = _top_percent_hub_mask(node_strength, float(args.main_hub_threshold))
    info(f"Controls used in structural average: {control_covars.shape[0]}")
    info(f"NBS nodes: {int(nbs_node_mask.sum())}")

    ########## COMPUTE CONTROL RICH-CLUB CURVE ##########
    print_section("COMPUTE CONTROL RICH-CLUB CURVE")
    control_richclub_curve_df = compute_control_richclub_curve(
        control_average_matrix=control_average_matrix,
        density=float(args.richclub_density),
        num_random=int(args.richclub_num_random),
        alpha=float(args.richclub_alpha),
        n_jobs=int(args.richclub_n_jobs),
    )

    ########## COMPUTE NODE-LEVEL METRICS ##########
    print_section("COMPUTE NODE-LEVEL METRICS")
    rho, p_standard, p_empirical = _spearman_permutation_test(
        x=node_strength,
        y=nbs_node_load,
        num_permutations=int(args.num_permutations),
        rng=rng,
    )
    strength_nbs = node_strength[nbs_node_mask]
    strength_non_nbs = node_strength[~nbs_node_mask]
    _, violin_p, rank_biserial = _mannwhitney_with_effect(strength_nbs, strength_non_nbs)
    hub_enrichment_df = _compute_hub_enrichment(
        nbs_node_mask=nbs_node_mask,
        node_strength=node_strength,
        thresholds=[int(val) for val in args.hub_thresholds],
        num_permutations=int(args.num_permutations),
        rng=rng,
    )
    edge_class_thresholds = [int(val) for val in args.edge_class_thresholds]
    edge_class_tables = []
    edge_class_enrichment_tables = []
    edge_class_columns: dict[int, pd.Series] = {}
    for threshold in edge_class_thresholds:
        threshold_hub_mask = _top_percent_hub_mask(node_strength, threshold)
        edge_class_df_thr, edge_class_enrichment_df_thr = _compute_edge_class_enrichment(
            nbs_mask=nbs_mask,
            hub_mask=threshold_hub_mask,
            num_permutations=int(args.num_permutations),
            rng=rng,
        )
        edge_class_df_thr["hub_threshold_percent"] = int(threshold)
        edge_class_enrichment_df_thr["hub_threshold_percent"] = int(threshold)
        edge_class_tables.append(edge_class_df_thr)
        edge_class_enrichment_tables.append(edge_class_enrichment_df_thr)
        edge_class_columns[int(threshold)] = edge_class_df_thr["edge_class"].reset_index(drop=True)
    edge_class_df = pd.concat(edge_class_tables, ignore_index=True)
    edge_class_enrichment_df = pd.concat(edge_class_enrichment_tables, ignore_index=True)

    ########## BUILD OUTPUT TABLES ##########
    print_section("BUILD OUTPUT TABLES")
    node_metrics_df = pd.DataFrame(
        {
            "node_index": np.arange(len(parcel_labels), dtype=int),
            "parcel_label": parcel_labels,
            "parcel_name": parcel_names.astype(str),
            "x": centroids_world[:, 0],
            "y": centroids_world[:, 1],
            "z": centroids_world[:, 2],
            "nbs_node_load": nbs_node_load.astype(int),
            "is_nbs_node": nbs_node_mask.astype(int),
            "control_structural_strength": node_strength.astype(float),
            f"is_top_{int(args.main_hub_threshold)}pct_hub": main_hub_mask.astype(int),
        }
    )
    for threshold in [int(val) for val in args.hub_thresholds]:
        node_metrics_df[f"is_top_{threshold}pct_hub"] = _top_percent_hub_mask(node_strength, threshold).astype(int)

    strength_group_df = pd.DataFrame(
        {
            "group": np.where(nbs_node_mask, "NBS nodes", "Non-NBS nodes"),
            "control_structural_strength": node_strength.astype(float),
        }
    )
    nbs_edge_table = build_nbs_edge_table(nbs_mask, parcel_labels, parcel_names, t_matrix)
    if not edge_class_df.empty:
        nbs_edge_table = nbs_edge_table.reset_index(drop=True)
        for threshold in edge_class_thresholds:
            nbs_edge_table[f"edge_class_top_{threshold}pct"] = edge_class_columns[int(threshold)]

    ########## GENERATE FIGURES ##########
    print_section("GENERATE FIGURES")
    figure_outputs = {}
    figure_outputs["metabolic_nbs_subnetwork"] = plot_metabolic_nbs_subnetwork(
        nbs_mask=nbs_mask,
        t_matrix=t_matrix,
        centroids_world=centroids_world,
        node_load=nbs_node_load,
        output_stem=output_dir / "figure_1_metabolic_nbs_subnetwork",
    )
    figure_outputs["control_structural_strength"] = plot_structural_strength_map(
        node_strength=node_strength,
        centroids_world=centroids_world,
        hub_mask=main_hub_mask,
        output_stem=output_dir / "figure_2_control_structural_node_strength",
    )
    figure_outputs["control_richclub_curve"] = plot_control_richclub_curve(
        richclub_curve_df=control_richclub_curve_df,
        output_stem=output_dir / "figure_3_control_structural_richclub_by_k",
    )
    figure_outputs["strength_violin"] = plot_strength_violin(
        df_strength_groups=strength_group_df,
        p_value=violin_p,
        rank_biserial=rank_biserial,
        output_stem=output_dir / "figure_4_structural_strength_nbs_vs_non_nbs",
    )
    figure_outputs["hub_enrichment"] = plot_hub_enrichment(
        enrichment_df=hub_enrichment_df,
        output_stem=output_dir / "figure_5_hub_enrichment_across_thresholds",
    )
    figure_outputs["edge_class_enrichment"] = plot_edge_class_enrichment(
        enrichment_df=edge_class_enrichment_df,
        thresholds=edge_class_thresholds,
        output_stem=output_dir / "figure_6_structural_class_of_nbs_edges",
    )
    figure_outputs["combined_subplot_panel"] = _compose_subplot_panel_figure(
        figure_outputs=figure_outputs,
        output_stem=output_dir / "figure_all_structural_context_panels",
        show=bool(args.show_figures),
    )
    _log_figure_outputs(figure_outputs)
    if bool(args.show_figures):
        info("Requested --show_figures: displayed the combined subplot figure with matplotlib.")

    ########## SAVE RESULTS ##########
    print_section("SAVE RESULTS")
    np.savez_compressed(
        output_dir / "control_average_structural_connectivity.npz",
        matrix_pop_avg=control_average_matrix.astype(np.float32),
        parcel_labels_group=parcel_labels,
        parcel_names_group=parcel_names,
        control_subject_id_list=control_covars[_resolve_column(control_covars, "participant_id", "subject_id")].astype(str).to_numpy(),
        control_session_id_list=control_covars[_resolve_column(control_covars, "session_id", "session")].astype(str).to_numpy(),
        structural_input_path=str(struct_path),
        nbs_input_path=str(input_nbs),
    )
    node_metrics_path = output_dir / "node_metrics.tsv"
    node_metrics_df.to_csv(node_metrics_path, sep="\t", index=False)
    hub_enrichment_path = output_dir / "hub_enrichment.tsv"
    hub_enrichment_df.to_csv(hub_enrichment_path, sep="\t", index=False)
    strength_group_path = output_dir / "nbs_vs_non_nbs_structural_strength.tsv"
    strength_group_df.to_csv(strength_group_path, sep="\t", index=False)
    nbs_edge_path = output_dir / "nbs_edges_with_structural_classes.tsv"
    nbs_edge_table.to_csv(nbs_edge_path, sep="\t", index=False)
    edge_class_path = output_dir / "edge_class_enrichment.tsv"
    edge_class_enrichment_df.to_csv(edge_class_path, sep="\t", index=False)
    richclub_curve_path = output_dir / "control_richclub_curve.tsv"
    control_richclub_curve_df.to_csv(richclub_curve_path, sep="\t", index=False)
    control_covars_path = output_dir / "control_covars_used.tsv"
    control_covars.to_csv(control_covars_path, sep="\t", index=False)
    info(f"Saved node metrics to {node_metrics_path}")
    info(f"Saved hub enrichment to {hub_enrichment_path}")
    info(f"Saved NBS edge classes to {nbs_edge_path}")
    info(f"Saved control rich-club curve to {richclub_curve_path}")

    stats_summary = {
        "spearman_rho": float(rho),
        "spearman_p": float(p_standard),
        "spearman_permutation_p": float(p_empirical),
        "mannwhitney_p": float(violin_p),
        "rank_biserial": float(rank_biserial),
        "main_hub_threshold_percent": int(args.main_hub_threshold),
        "edge_class_thresholds_percent": edge_class_thresholds,
        "richclub_density": float(args.richclub_density),
        "richclub_num_random": int(args.richclub_num_random),
        "richclub_alpha": float(args.richclub_alpha),
        "n_controls": int(control_covars.shape[0]),
        "n_nbs_nodes": int(nbs_node_mask.sum()),
        "n_non_nbs_nodes": int((~nbs_node_mask).sum()),
        "n_nbs_edges": int(np.triu(nbs_mask, k=1).sum()),
    }
    summary = {
        "nbs_summary": nbs_summary,
        "parcel_alignment": parcel_alignment_meta,
        "pair_alignment": {
            "n_target_pairs": int(len(target_pairs)),
            "n_matched_pairs": int(len(kept_target_indices)),
            "n_missing_pairs": int(len(missing_pairs)),
            "missing_pairs": [f"{sub}-{ses}" for sub, ses in missing_pairs],
        },
        "structural_input_path": str(struct_path),
        "output_dir": str(output_dir),
        "control_value": int(args.control_value),
        "hub_thresholds": [int(val) for val in args.hub_thresholds],
        "edge_class_thresholds": edge_class_thresholds,
        "main_hub_threshold": int(args.main_hub_threshold),
        "num_permutations": int(args.num_permutations),
        "seed": int(args.seed),
        "stats_summary": stats_summary,
        "tables": {
            "node_metrics_tsv": str(node_metrics_path),
            "hub_enrichment_tsv": str(hub_enrichment_path),
            "strength_groups_tsv": str(strength_group_path),
            "nbs_edges_with_structural_classes_tsv": str(nbs_edge_path),
            "edge_class_enrichment_tsv": str(edge_class_path),
            "control_richclub_curve_tsv": str(richclub_curve_path),
            "control_covars_used_tsv": str(control_covars_path),
        },
        "figures": figure_outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
