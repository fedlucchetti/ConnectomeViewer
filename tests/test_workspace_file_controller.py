from pathlib import Path
import tempfile
import unittest

import numpy as np

from services.workspace_file_controller import WorkspaceFileController


class _StatusBarStub:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(str(message))


class _WorkspaceStub:
    def __init__(self):
        self.last_entry = None

    def add_derived_entry(self, matrix, **kwargs):
        entry = {
            "id": "derived-1",
            "label": kwargs["label"],
            "matrix": np.asarray(matrix, dtype=float),
            **kwargs,
        }
        self.last_entry = entry
        return entry["id"], entry


class _ViewerStub:
    def __init__(self):
        self._status_bar = _StatusBarStub()
        self._workspace = _WorkspaceStub()
        self.added_items = []

    def statusBar(self):
        return self._status_bar

    def _add_workspace_list_item(self, entry, **kwargs):
        self.added_items.append((entry, kwargs))


class WorkspaceFileControllerTests(unittest.TestCase):
    def test_bind_viewer_methods_uses_viewer_for_instance_method_self(self):
        viewer = _ViewerStub()
        controller = WorkspaceFileController(viewer)

        controller.bind_viewer_methods()

        self.assertIs(viewer._add_files.__self__, viewer)
        self.assertIs(viewer._open_batch_import_dialog.__self__, viewer)

    def test_batch_connectivity_paths_filters_and_sorts_recursively(self):
        viewer = _ViewerStub()
        controller = WorkspaceFileController(viewer)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'z_dir').mkdir()
            (root / 'a_dir').mkdir()
            (root / 'z_dir' / 'connectivity_func_scale3.npz').write_text('x', encoding='utf-8')
            (root / 'a_dir' / 'connectivity_mrsi_scale2.npz').write_text('x', encoding='utf-8')
            (root / 'a_dir' / 'matrix_scale2.npz').write_text('x', encoding='utf-8')
            (root / 'connectivity_notes.txt').write_text('x', encoding='utf-8')

            paths = controller._batch_connectivity_paths(root)

        self.assertEqual(
            [path.as_posix().split(tmpdir + '/')[-1] for path in paths],
            [
                'a_dir/connectivity_mrsi_scale2.npz',
                'z_dir/connectivity_func_scale3.npz',
            ],
        )

    def test_import_selector_aggregate_creates_workspace_entry_with_filter_label(self):
        viewer = _ViewerStub()
        controller = WorkspaceFileController(viewer)

        ok = controller._import_selector_aggregate(
            {
                'matrix': np.eye(3, dtype=float),
                'source_path': '/tmp/source_connectivity.npz',
                'matrix_key': 'matrix_pop_avg',
                'method': 'zfisher',
                'selected_rows': [1, 3],
                'n_total_rows': 8,
                'filter_covar': 'group',
                'filter_values': ['hc', 'chr'],
            }
        )

        self.assertTrue(ok)
        self.assertEqual(
            viewer._workspace.last_entry['label'],
            'zfisher matrix_pop_avg [group=hc,chr] (source_connectivity.npz)',
        )
        self.assertEqual(viewer._workspace.last_entry['extra_fields']['aggregation_method'], 'zfisher')
        self.assertEqual(viewer._status_bar.messages[-1], 'Imported aggregated matrix (zfisher).')


if __name__ == '__main__':
    unittest.main()
