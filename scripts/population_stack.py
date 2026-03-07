from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PopulationCleanupResult:
    matrix_subjects_list_clean: np.ndarray
    matrix_pop_avg_clean: np.ndarray
    metab_profiles_subjects_clean: np.ndarray | None
    parcel_labels_group: np.ndarray
    parcel_names_group: np.ndarray
    subject_id_arr_sel: np.ndarray
    session_id_arr_sel: np.ndarray
    discarded_subjects: list[str]
    discarded_sessions: list[str]
    n_voxel_counts_dict: dict | None = None
    raw_n_voxel_counts_dict: dict | None = None


def _log(debug, method: str, *args) -> None:
    if debug is None:
        return
    fn = getattr(debug, method, None)
    if callable(fn):
        fn(*args)


def _as_object_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=object)
    return np.asarray(values, dtype=object)


def _synchronize_population_axes(
    matrix_subjects_list,
    parcel_labels_group,
    parcel_names_group,
    metab_profiles_subjects,
    debug=None,
):
    matrix_subjects_list = np.asarray(matrix_subjects_list)
    if matrix_subjects_list.ndim != 3:
        raise ValueError(
            f"Expected subject matrix stack with shape [subjects, nodes, nodes], got {matrix_subjects_list.shape}."
        )
    if matrix_subjects_list.shape[1] != matrix_subjects_list.shape[2]:
        raise ValueError(f"Expected square subject matrices, got {matrix_subjects_list.shape}.")

    parcel_labels_group = np.asarray(parcel_labels_group).reshape(-1)
    parcel_names_group = np.asarray(parcel_names_group, dtype=object).reshape(-1)
    if metab_profiles_subjects is not None:
        metab_profiles_subjects = np.asarray(metab_profiles_subjects)
        if metab_profiles_subjects.ndim < 2:
            raise ValueError(
                f"Expected parcel profiles with shape [subjects, nodes, ...], got {metab_profiles_subjects.shape}."
            )

    axis_lengths = [
        int(matrix_subjects_list.shape[1]),
        int(parcel_labels_group.shape[0]),
        int(parcel_names_group.shape[0]),
    ]
    if metab_profiles_subjects is not None:
        axis_lengths.append(int(metab_profiles_subjects.shape[1]))

    target_len = min(axis_lengths)
    if target_len <= 0:
        raise ValueError("Population stack has no parcel nodes to clean.")

    if len(set(axis_lengths)) != 1:
        _log(
            debug,
            "warning",
            f"Population stack axis mismatch {axis_lengths}; truncating all parcel axes to {target_len}.",
        )
        matrix_subjects_list = matrix_subjects_list[:, :target_len, :target_len]
        parcel_labels_group = parcel_labels_group[:target_len]
        parcel_names_group = parcel_names_group[:target_len]
        if metab_profiles_subjects is not None:
            metab_profiles_subjects = metab_profiles_subjects[:, :target_len, ...]

    return matrix_subjects_list, parcel_labels_group, parcel_names_group, metab_profiles_subjects


def _apply_node_mask(
    matrix_subjects_list,
    parcel_labels_group,
    parcel_names_group,
    node_mask,
    metab_profiles_subjects=None,
):
    node_mask = np.asarray(node_mask, dtype=bool)
    parcel_labels_group = np.asarray(parcel_labels_group)[node_mask]
    parcel_names_group = np.asarray(parcel_names_group, dtype=object)[node_mask]
    matrix_subjects_list = np.asarray(matrix_subjects_list)[:, node_mask, :][:, :, node_mask]
    if metab_profiles_subjects is not None:
        metab_profiles_subjects = np.asarray(metab_profiles_subjects)[:, node_mask, ...]
    return matrix_subjects_list, parcel_labels_group, parcel_names_group, metab_profiles_subjects


def _remove_bnd_and_wm(
    matrix_subjects_list,
    parcel_labels_group,
    parcel_names_group,
    metab_profiles_subjects=None,
):
    parcel_names_group = np.asarray(parcel_names_group, dtype=object)
    keep_mask = np.array([str(name) != "BND" for name in parcel_names_group], dtype=bool)
    wm_start = None
    for idx, name in enumerate(parcel_names_group):
        if isinstance(name, str) and "wm-" in name:
            wm_start = idx
            break
    if wm_start is not None:
        keep_mask[wm_start:] = False
    return _apply_node_mask(
        matrix_subjects_list,
        parcel_labels_group,
        parcel_names_group,
        keep_mask,
        metab_profiles_subjects=metab_profiles_subjects,
    )


def _exclude_sparse_subjects(
    matrix_subjects_list,
    metab_profiles_subjects,
    subject_id_list,
    session_list,
    simm,
    debug=None,
    sigma=3,
):
    _log(debug, "title", "Exclude sparse within-subject-wise MeSiMs")
    matrix_list_sel, _include_indices, exclude_indices = simm.filter_sparse_matrices(
        matrix_subjects_list, sigma=sigma
    )
    matrix_list_sel = np.asarray(matrix_list_sel)
    exclude_indices = np.asarray(exclude_indices, dtype=int)

    subject_id_arr = _as_object_array(subject_id_list)
    session_id_arr = _as_object_array(session_list)
    if exclude_indices.size:
        metab_profiles_subjects_sel = np.delete(metab_profiles_subjects, exclude_indices, axis=0)
        subject_id_arr_sel = np.delete(subject_id_arr, exclude_indices, axis=0)
        session_id_arr_sel = np.delete(session_id_arr, exclude_indices, axis=0)
        discarded_subjects = [str(subject_id_arr[idx]) for idx in exclude_indices]
        discarded_sessions = [str(session_id_arr[idx]) for idx in exclude_indices]
    else:
        metab_profiles_subjects_sel = np.asarray(metab_profiles_subjects)
        subject_id_arr_sel = subject_id_arr
        session_id_arr_sel = session_id_arr
        discarded_subjects = []
        discarded_sessions = []

    if matrix_list_sel.ndim != 3 or matrix_list_sel.shape[0] == 0:
        raise ValueError("All subject matrices were excluded during sparse-subject cleanup.")

    _log(
        debug,
        "info",
        f"Excluded {len(exclude_indices)} sparse MeSiMs of shape, remaining {matrix_list_sel.shape[0]}",
    )
    for sub, ses in zip(discarded_subjects, discarded_sessions):
        _log(debug, "info", sub, ses, "was left out")

    return (
        matrix_list_sel,
        metab_profiles_subjects_sel,
        subject_id_arr_sel,
        session_id_arr_sel,
        discarded_subjects,
        discarded_sessions,
    )


def _filter_qmask_low_coverage(
    matrix_list_sel,
    metab_profiles_subjects_sel,
    parcel_labels_group,
    parcel_names_group,
    *,
    modality,
    parc=None,
    parcellation_img=None,
    qmask_pop_img=None,
    mrsi_cov=0,
    resample_to_img_fn=None,
    debug=None,
):
    n_voxel_counts_dict = None
    raw_n_voxel_counts_dict = None

    if modality != "mrsi":
        _log(debug, "warning", "Skipping qmask filtering for DWI modality")
        return (
            matrix_list_sel,
            metab_profiles_subjects_sel,
            parcel_labels_group,
            parcel_names_group,
            n_voxel_counts_dict,
            raw_n_voxel_counts_dict,
        )

    if parc is None or parcellation_img is None or qmask_pop_img is None:
        _log(debug, "warning", "Skipping qmask filtering because qmask inputs are incomplete.")
        return (
            matrix_list_sel,
            metab_profiles_subjects_sel,
            parcel_labels_group,
            parcel_names_group,
            n_voxel_counts_dict,
            raw_n_voxel_counts_dict,
        )

    if parcellation_img.shape != qmask_pop_img.shape:
        _log(
            debug,
            "warning",
            f"Parcel image shape {parcellation_img.shape} does not match Qmask shape {qmask_pop_img.shape}",
        )
        if resample_to_img_fn is None:
            raise ValueError("qmask filtering requires resample_to_img when parcel and qmask shapes differ.")
        _log(debug, "proc", "Resampling...")
        parcellation_img = resample_to_img_fn(
            source_img=parcellation_img,
            target_img=qmask_pop_img,
            interpolation="nearest",
        )

    n_voxel_counts_dict = parc.count_voxels_inside_parcel(
        qmask_pop_img.get_fdata(),
        parcellation_img.get_fdata().astype(int),
        parcel_labels_group,
    )
    ignore_parcel_idx = [
        parcel_idx for parcel_idx, coverage in n_voxel_counts_dict.items() if coverage < mrsi_cov
    ]
    ignore_rows = [
        np.where(parcel_labels_group == parcel_idx)[0][0]
        for parcel_idx in ignore_parcel_idx
        if len(np.where(parcel_labels_group == parcel_idx)[0]) != 0
    ]
    ignore_rows = np.sort(np.array(ignore_rows, dtype=int))

    if ignore_rows.size:
        keep_mask = np.ones(len(parcel_labels_group), dtype=bool)
        keep_mask[ignore_rows] = False
        matrix_list_sel, parcel_labels_group, parcel_names_group, metab_profiles_subjects_sel = _apply_node_mask(
            matrix_list_sel,
            parcel_labels_group,
            parcel_names_group,
            keep_mask,
            metab_profiles_subjects=metab_profiles_subjects_sel,
        )

    _log(
        debug,
        "info",
        f"Matrix shape {matrix_list_sel.shape[1:]} from {matrix_list_sel.shape[0]} subjects",
    )
    raw_n_voxel_counts_dict = parc.count_voxels_inside_parcel(
        qmask_pop_img.get_fdata(),
        parcellation_img.get_fdata().astype(int),
        parcel_labels_group,
        norm=False,
    )
    return (
        matrix_list_sel,
        metab_profiles_subjects_sel,
        parcel_labels_group,
        parcel_names_group,
        n_voxel_counts_dict,
        raw_n_voxel_counts_dict,
    )


def _remove_sparse_nodes(
    matrix_list_sel,
    metab_profiles_subjects_sel,
    parcel_labels_group,
    parcel_names_group,
    *,
    modality,
    debug=None,
):
    if modality != "mrsi":
        matrix_subjects_list_clean = np.asarray(matrix_list_sel)
        matrix_pop_avg_clean = np.asarray(matrix_subjects_list_clean.mean(axis=0))
        return (
            matrix_subjects_list_clean,
            matrix_pop_avg_clean,
            np.asarray(metab_profiles_subjects_sel),
            np.asarray(parcel_labels_group),
            np.asarray(parcel_names_group, dtype=object),
        )

    _log(debug, "title", "Remove sparse nodes")
    me_sim_pop_avg = np.asarray(matrix_list_sel.mean(axis=0))
    mask_parcel_indices = list(np.where(np.diag(me_sim_pop_avg) == 0)[0])
    nonzero_frac = np.mean(~np.isclose(matrix_list_sel, 0, atol=1e-10), axis=(0, 2))
    mask_parcel_indices.extend(np.where(nonzero_frac < 0.05)[0])
    mask_parcel_indices = np.unique(np.asarray(mask_parcel_indices, dtype=int))

    if mask_parcel_indices.size:
        labels_before = np.asarray(parcel_labels_group)
        names_before = np.asarray(parcel_names_group, dtype=object)
        keep_mask = np.ones(len(parcel_labels_group), dtype=bool)
        keep_mask[mask_parcel_indices] = False
        if not np.any(keep_mask):
            raise ValueError("All parcel nodes were removed during sparse-node cleanup.")
        matrix_subjects_list_clean, parcel_labels_group, parcel_names_group, metab_profiles_subjects_clean = (
            _apply_node_mask(
                matrix_list_sel,
                parcel_labels_group,
                parcel_names_group,
                keep_mask,
                metab_profiles_subjects=metab_profiles_subjects_sel,
            )
        )
        matrix_pop_avg_clean = np.asarray(matrix_subjects_list_clean.mean(axis=0))
        for idx in mask_parcel_indices:
            _log(debug, "info", "Removed sparse connectivty node", labels_before[idx], names_before[idx])
        _log(debug, "separator")
    else:
        matrix_subjects_list_clean = np.array(matrix_list_sel, copy=True)
        metab_profiles_subjects_clean = np.array(metab_profiles_subjects_sel, copy=True)
        matrix_pop_avg_clean = np.array(matrix_list_sel.mean(axis=0), copy=True)
        _log(debug, "info", "No sparse nodes detected, none removed")

    return (
        matrix_subjects_list_clean,
        matrix_pop_avg_clean,
        metab_profiles_subjects_clean,
        np.asarray(parcel_labels_group),
        np.asarray(parcel_names_group, dtype=object),
    )


def clean_population_stack(
    *,
    matrix_subjects_list,
    parcel_labels_group,
    parcel_names_group,
    metab_profiles_subjects,
    subject_id_list,
    session_list,
    modality,
    simm,
    debug=None,
    sigma=3,
    parc=None,
    parcellation_img=None,
    qmask_pop_img=None,
    mrsi_cov=0,
    resample_to_img_fn=None,
) -> PopulationCleanupResult:
    (
        matrix_subjects_list,
        parcel_labels_group,
        parcel_names_group,
        metab_profiles_subjects,
    ) = _synchronize_population_axes(
        matrix_subjects_list,
        parcel_labels_group,
        parcel_names_group,
        metab_profiles_subjects,
        debug=debug,
    )

    (
        matrix_subjects_list,
        parcel_labels_group,
        parcel_names_group,
        metab_profiles_subjects,
    ) = _remove_bnd_and_wm(
        matrix_subjects_list,
        parcel_labels_group,
        parcel_names_group,
        metab_profiles_subjects=metab_profiles_subjects,
    )
    if matrix_subjects_list.shape[1] == 0:
        raise ValueError("All parcel nodes were removed during BND/WM cleanup.")
    _log(
        debug,
        "info",
        f"Collected {matrix_subjects_list.shape[0]} matrices of shape {matrix_subjects_list.shape[1:]}",
    )

    (
        matrix_list_sel,
        metab_profiles_subjects_sel,
        subject_id_arr_sel,
        session_id_arr_sel,
        discarded_subjects,
        discarded_sessions,
    ) = _exclude_sparse_subjects(
        matrix_subjects_list,
        metab_profiles_subjects,
        subject_id_list,
        session_list,
        simm,
        debug=debug,
        sigma=sigma,
    )

    (
        matrix_list_sel,
        metab_profiles_subjects_sel,
        parcel_labels_group,
        parcel_names_group,
        n_voxel_counts_dict,
        raw_n_voxel_counts_dict,
    ) = _filter_qmask_low_coverage(
        matrix_list_sel,
        metab_profiles_subjects_sel,
        parcel_labels_group,
        parcel_names_group,
        modality=modality,
        parc=parc,
        parcellation_img=parcellation_img,
        qmask_pop_img=qmask_pop_img,
        mrsi_cov=mrsi_cov,
        resample_to_img_fn=resample_to_img_fn,
        debug=debug,
    )
    if len(parcel_labels_group) == 0:
        raise ValueError("All parcel nodes were removed during qmask cleanup.")

    (
        matrix_subjects_list_clean,
        matrix_pop_avg_clean,
        metab_profiles_subjects_clean,
        parcel_labels_group,
        parcel_names_group,
    ) = _remove_sparse_nodes(
        matrix_list_sel,
        metab_profiles_subjects_sel,
        parcel_labels_group,
        parcel_names_group,
        modality=modality,
        debug=debug,
    )

    return PopulationCleanupResult(
        matrix_subjects_list_clean=np.asarray(matrix_subjects_list_clean),
        matrix_pop_avg_clean=np.asarray(matrix_pop_avg_clean),
        metab_profiles_subjects_clean=(
            None if metab_profiles_subjects_clean is None else np.asarray(metab_profiles_subjects_clean)
        ),
        parcel_labels_group=np.asarray(parcel_labels_group),
        parcel_names_group=np.asarray(parcel_names_group, dtype=object),
        subject_id_arr_sel=np.asarray(subject_id_arr_sel, dtype=object),
        session_id_arr_sel=np.asarray(session_id_arr_sel, dtype=object),
        discarded_subjects=list(discarded_subjects),
        discarded_sessions=list(discarded_sessions),
        n_voxel_counts_dict=n_voxel_counts_dict,
        raw_n_voxel_counts_dict=raw_n_voxel_counts_dict,
    )


__all__ = ["PopulationCleanupResult", "clean_population_stack"]
