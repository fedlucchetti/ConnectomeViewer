import unittest

from services.workspace_matrix_controller import WorkspaceMatrixController


class _ComboBoxStub:
    def __init__(self):
        self.items = []
        self.current_text = ""
        self.enabled = False

    def blockSignals(self, _value):
        return None

    def clear(self):
        self.items = []
        self.current_text = ""

    def addItem(self, value):
        self.items.append(str(value))

    def findText(self, value):
        try:
            return self.items.index(str(value))
        except ValueError:
            return -1

    def setCurrentText(self, value):
        self.current_text = str(value)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def currentText(self):
        return self.current_text

    def count(self):
        return len(self.items)

    def setCurrentIndex(self, index):
        if 0 <= index < len(self.items):
            self.current_text = self.items[index]


class _LineEditStub:
    def __init__(self, text=""):
        self._text = text
        self.enabled = True

    def text(self):
        return self._text

    def setText(self, value):
        self._text = str(value)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def blockSignals(self, _value):
        return None


class _CheckBoxStub:
    def __init__(self, checked=False):
        self._checked = bool(checked)

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._checked = bool(value)

    def blockSignals(self, _value):
        return None


class _ViewerStub:
    def __init__(self):
        self.key_combo = _ComboBoxStub()
        self.cmap_combo = _ComboBoxStub()
        self.cmap_combo.items = ["plasma", "magma"]
        self.cmap_combo.current_text = "plasma"
        self.covar_combo = _ComboBoxStub()
        self.display_auto_check = _CheckBoxStub(True)
        self.display_min_edit = _LineEditStub("")
        self.display_max_edit = _LineEditStub("")
        self.display_scale_combo = _ComboBoxStub()
        self.display_scale_combo.items = ["Linear", "Log"]
        self.display_scale_combo.current_text = "Linear"
        self.title_edit = _LineEditStub("")
        self.sample_spin = None
        self.sample_add_button = None
        self.titles = {}
        self._default_matrix_colormap = "plasma"
        self._custom_cmaps = set()
        self._colorbar = None

    def _available_colormap_names(self):
        return ["plasma", "magma"]

    def _get_valid_keys_cached(self, _path):
        return ["alpha", "beta"]


class WorkspaceMatrixControllerTests(unittest.TestCase):
    def test_refresh_key_options_selects_first_available_key(self):
        viewer = _ViewerStub()
        controller = WorkspaceMatrixController(
            viewer,
            fallback_colormap="plasma",
            covars_columns=lambda _info: [],
        )
        entry = {"kind": "file", "path": "dummy.npz", "selected_key": None}
        controller.refresh_key_options(entry)
        self.assertEqual(entry["selected_key"], "alpha")
        self.assertEqual(viewer.key_combo.items, ["alpha", "beta"])
        self.assertEqual(viewer.key_combo.current_text, "alpha")
        self.assertTrue(viewer.key_combo.enabled)

    def test_current_display_limits_reads_manual_widget_values(self):
        viewer = _ViewerStub()
        viewer.display_auto_check.setChecked(False)
        viewer.display_min_edit.setText("1.5")
        viewer.display_max_edit.setText("3.5")
        controller = WorkspaceMatrixController(
            viewer,
            fallback_colormap="plasma",
            covars_columns=lambda _info: [],
        )
        self.assertEqual(
            controller.current_display_limits(),
            (1.5, 3.5, None),
        )


if __name__ == "__main__":
    unittest.main()
