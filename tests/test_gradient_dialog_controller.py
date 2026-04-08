import ast
import inspect
from pathlib import Path
import textwrap
import unittest
from unittest import mock

import numpy as np

import services.gradient_dialog_controller as gradient_module
from services.gradient_dialog_controller import GradientDialogController


class _StatusBarStub:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(str(message))


class _HeaderStub(dict):
    def copy(self):
        return _HeaderStub(self)


class _TemplateImageStub:
    def __init__(self):
        self.affine = np.eye(4, dtype=float)
        self.header = _HeaderStub()


class _DataAccessStub:
    def entry_parcel_metadata(self, _entry, expected_len=None):
        labels = np.asarray([1, 2, 3], dtype=int)
        if expected_len is not None and int(expected_len) != labels.size:
            raise AssertionError("Unexpected expected_len")
        return labels, ["Parcel 1", "Parcel 2", "Parcel 3"]


class _ViewerStub:
    def __init__(self):
        self._status_bar = _StatusBarStub()
        self._entries = {
            "entry-1": {
                "id": "entry-1",
                "label": "matrix_a",
                "path": Path("/tmp/matrix_a.npz"),
            }
        }
        self._matrix = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=float,
        )
        self._data_access = _DataAccessStub()
        self._gradient_selected_entry_id = "entry-1"
        self._gradient_component_count = 2
        self._gradient_precomputed_bundle = None
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = False
        self._active_parcellation_path = Path("/tmp/template.nii.gz")
        self._active_parcellation_img = _TemplateImageStub()
        self._active_parcellation_data = np.asarray(
            [
                [[1, 0], [0, 2]],
                [[0, 3], [0, 0]],
            ],
            dtype=int,
        )
        self._gradients_busy = False
        self._gradients_progress_state = {"minimum": 0, "maximum": 1, "value": 0, "text": "Idle"}
        self._gradients_dialog = None
        self._last_gradients = None
        self._theme_name = "Dark"

    def statusBar(self):
        return self._status_bar

    def _default_results_dir(self):
        return Path("/tmp")

    def _current_entry(self):
        return None

    def _matrix_for_entry(self, _entry):
        return np.asarray(self._matrix, dtype=float), None


class _NetToolsStub:
    def __init__(self):
        self.project_calls = 0

    def dimreduce_matrix(self, matrix, *, output_dim, **_kwargs):
        return np.arange(matrix.shape[0], dtype=float) + float(output_dim)

    def project_to_3dspace(self, *_args, **_kwargs):
        self.project_calls += 1
        raise AssertionError("Gradient compute should not project to 3D space.")


class _QApplicationStub:
    @staticmethod
    def processEvents():
        return None

    @staticmethod
    def primaryScreen():
        return None


def _make_controller(viewer):
    return GradientDialogController(
        viewer,
        parcel_label_keys=("parcel_labels_group",),
        parcel_name_keys=("parcel_names_group",),
        to_string_list=lambda values: [str(value) for value in np.asarray(values).reshape(-1).tolist()],
        display_text=lambda value: str(value),
        load_covars_info=lambda _path: None,
        covars_to_rows=lambda _info: ([], []),
        normalize_subject_token=lambda value: str(value),
        normalize_session_token=lambda value: str(value),
        flatten_display_vector=lambda values: [str(value) for value in np.asarray(values).reshape(-1).tolist()],
        coerce_label_indices=lambda labels, expected_len: (
            np.asarray(labels, dtype=int).reshape(-1).tolist()
            if np.asarray(labels).size == int(expected_len)
            else None
        ),
    )


class GradientDialogControllerTests(unittest.TestCase):
    def test_compute_gradients_defers_projection_until_render(self):
        viewer = _ViewerStub()
        controller = _make_controller(viewer)
        nettools_stub = _NetToolsStub()

        with mock.patch.object(gradient_module, "nettools", nettools_stub), mock.patch.object(
            gradient_module, "QApplication", _QApplicationStub
        ):
            controller._compute_gradients()

        results = viewer._last_gradients
        self.assertIsNotNone(results)
        self.assertIsNone(results["projected_data"])
        self.assertEqual(results["gradients"].shape, (3, 2))
        self.assertEqual(nettools_stub.project_calls, 0)
        self.assertIn("Projection will run only when", viewer._status_bar.messages[-1])

    def test_bind_viewer_methods_uses_viewer_for_instance_method_self(self):
        viewer = _ViewerStub()
        controller = _make_controller(viewer)

        controller.bind_viewer_methods()

        self.assertIs(viewer._open_gradients_dialog.__self__, viewer)
        self.assertEqual(viewer._normalize_gradient_surface_mesh("bad-mesh"), "fsaverage4")

    def test_classification_scatter_call_sets_default_proximity_keywords(self):
        source = textwrap.dedent(inspect.getsource(GradientDialogController._classify_gradients_fsaverage))
        tree = ast.parse(source)
        scatter_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GradientScatterDialog"
        ]
        self.assertEqual(len(scatter_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in scatter_calls[0].keywords if keyword.arg}
        self.assertIn("show_proximity_circles", keywords)
        self.assertIn("initial_proximity_slider_value", keywords)
        self.assertIn("use_line_proximity_energy", keywords)
        self.assertIn("path_metric_coords", keywords)
        self.assertIs(keywords["show_proximity_circles"].value, False)
        self.assertEqual(keywords["initial_proximity_slider_value"].value, 1000)
        self.assertIs(keywords["use_line_proximity_energy"].value, False)

    def test_sync_gradients_dialog_state_reuses_precomputed_dialog_rows(self):
        source = textwrap.dedent(inspect.getsource(GradientDialogController._sync_gradients_dialog_state))
        self.assertIn("dialog_covars_rows", source)


if __name__ == "__main__":
    unittest.main()
