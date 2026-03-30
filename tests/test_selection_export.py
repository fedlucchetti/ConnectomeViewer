from pathlib import Path
import unittest

import numpy as np

from services.selection_export import (
    build_key_options_state,
    build_selection_change_state,
    collect_grid_export_items,
    default_matrix_export_name,
    normalize_grid_export_output_path,
)


class SelectionExportTests(unittest.TestCase):
    def test_build_key_options_state_sets_first_available_key(self):
        entry = {"kind": "file", "selected_key": None}
        state = build_key_options_state(entry, valid_keys=["alpha", "beta"])
        self.assertEqual(state.items, ("alpha", "beta"))
        self.assertEqual(state.selected_key, "alpha")
        self.assertTrue(state.enabled)
        self.assertEqual(entry["selected_key"], "alpha")

    def test_build_selection_change_state_uses_manual_title_when_present(self):
        entry = {
            "kind": "derived",
            "selected_key": "matrix_pop_avg",
            "auto_title": False,
            "label": "Derived",
            "source_path": Path("/tmp/example.npz"),
        }
        state = build_selection_change_state(
            entry,
            stored_title="Custom Title",
            default_title="Default Title",
        )
        self.assertEqual(state.key_options.items, ("matrix_pop_avg",))
        self.assertFalse(state.key_options.enabled)
        self.assertEqual(state.title_text, "Custom Title")
        self.assertEqual(state.source_path, Path("/tmp/example.npz"))

    def test_default_matrix_export_name_sanitizes_source_and_key(self):
        entry = {"source_path": Path("/tmp/my matrix.npz")}
        name = default_matrix_export_name(entry, selected_key="group/value")
        self.assertEqual(name, "my_matrix_group_value_matrix_pop_avg.npz")

    def test_normalize_grid_export_output_path_uses_selected_filter(self):
        output_path = normalize_grid_export_output_path(
            "~/exports/connectome_grid",
            selected_filter="SVG (*.svg)",
        )
        self.assertEqual(output_path.suffix, ".svg")

    def test_collect_grid_export_items_packages_results_and_skips_failures(self):
        entries = {
            "good": {"label": "Good"},
            "bad": {"label": "Bad"},
        }
        bundle = collect_grid_export_items(
            ["good", "bad"],
            entries,
            {"good": "Good Title"},
            matrix_for_entry=lambda entry: (
                (np.asarray([[1.0, 2.0], [2.0, 1.0]]), None)
                if entry["label"] == "Good"
                else (_raise(ValueError("missing key")))
            ),
            display_limits_for_entry=lambda _entry: (None, None, None),
            display_scale_for_entry=lambda _entry: "linear",
            log_scale_error=lambda _matrix, _vmin, _vmax: None,
            selected_colormap_for_entry=lambda _entry: "plasma",
        )
        self.assertIsNone(bundle.error)
        self.assertEqual(len(bundle.items), 1)
        self.assertEqual(bundle.items[0].title, "Good Title")
        self.assertEqual(bundle.items[0].colormap, "plasma")
        self.assertEqual(len(bundle.skipped), 1)
        self.assertIn("Bad", bundle.skipped[0])

    def test_collect_grid_export_items_surfaces_scale_error(self):
        entries = {"good": {"label": "Good"}}
        bundle = collect_grid_export_items(
            ["good"],
            entries,
            {},
            matrix_for_entry=lambda _entry: (np.asarray([[1.0, 2.0], [2.0, 1.0]]), None),
            display_limits_for_entry=lambda _entry: (None, None, "Display min must be numeric."),
            display_scale_for_entry=lambda _entry: "linear",
            log_scale_error=lambda _matrix, _vmin, _vmax: None,
            selected_colormap_for_entry=lambda _entry: "plasma",
        )
        self.assertEqual(bundle.error, "Good: Display min must be numeric.")


def _raise(exc):
    raise exc


if __name__ == "__main__":
    unittest.main()
