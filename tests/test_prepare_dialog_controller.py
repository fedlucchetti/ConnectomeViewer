from pathlib import Path
import tempfile
import unittest

import numpy as np

from services.prepare_dialog_controller import PrepareDialogController


class _ViewerStub:
    def __init__(self, entry, key="matrix_pop_avg", covars_info=None):
        self._entry = entry
        self._key = key
        self._covars_info_value = covars_info

    def _current_entry(self):
        return self._entry

    def _ensure_entry_key(self, _entry):
        return self._key

    def _covars_info_cached(self, _path):
        return self._covars_info_value


class PrepareDialogControllerTests(unittest.TestCase):
    def test_current_stack_source_uses_cached_stack_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.npz"
            source_path.write_text("x", encoding="utf-8")
            entry = {
                "kind": "file",
                "path": source_path,
                "stack_axis": 2,
                "stack_len": 5,
            }
            controller = PrepareDialogController(
                _ViewerStub(entry),
                covars_columns=lambda _info: [],
                load_matrix_from_npz=lambda *_args, **_kwargs: None,
                stack_axis=lambda _shape: None,
            )
            source = controller.current_stack_source()
            self.assertEqual(
                source,
                {"path": source_path, "key": "matrix_pop_avg", "stack_len": 5},
            )

    def test_current_stack_source_resolves_stack_length_from_raw_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.npz"
            source_path.write_text("x", encoding="utf-8")
            entry = {
                "kind": "file",
                "path": source_path,
            }
            controller = PrepareDialogController(
                _ViewerStub(entry),
                covars_columns=lambda _info: [],
                load_matrix_from_npz=lambda *_args, **_kwargs: np.zeros((4, 4, 7), dtype=float),
                stack_axis=lambda shape: 2 if shape == (4, 4, 7) else None,
            )
            source = controller.current_stack_source()
            self.assertEqual(
                source,
                {"path": source_path, "key": "matrix_pop_avg", "stack_len": 7},
            )


if __name__ == "__main__":
    unittest.main()
