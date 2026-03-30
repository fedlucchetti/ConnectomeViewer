import unittest

import numpy as np

from services.display_state import (
    build_display_controls_state,
    log_scale_error,
    parse_display_limits,
    selected_colormap_name,
    store_display_controls,
    update_sample_control_state,
)


class DisplayStateTests(unittest.TestCase):
    def test_build_and_store_display_controls_normalize_entry_state(self):
        entry = {"matrix_colormap": "invalid", "display_scale": "LOG"}
        state = build_display_controls_state(
            entry,
            default_matrix_colormap="plasma",
            available_colormap_names=["plasma", "magma"],
            fallback_colormap="plasma",
        )
        self.assertEqual(state.colormap_name, "plasma")
        self.assertEqual(state.scale_label, "Log")
        self.assertTrue(state.auto_scale)
        self.assertFalse(state.min_enabled)
        self.assertFalse(state.max_enabled)

        stored = store_display_controls(
            entry,
            colormap_name="magma",
            auto_scale=False,
            min_text=" 1.5 ",
            max_text=" 2.5 ",
            scale_choice="linear",
            default_matrix_colormap="plasma",
            available_colormap_names=["plasma", "magma"],
            fallback_colormap="plasma",
        )
        self.assertEqual(entry["matrix_colormap"], "magma")
        self.assertFalse(entry["display_auto"])
        self.assertEqual(entry["display_min_text"], "1.5")
        self.assertEqual(entry["display_max_text"], "2.5")
        self.assertEqual(entry["display_scale"], "linear")
        self.assertTrue(stored.min_enabled)
        self.assertTrue(stored.max_enabled)

    def test_selected_colormap_name_falls_back_to_available_values(self):
        name = selected_colormap_name(
            current_name="missing",
            default_matrix_colormap="plasma",
            available_colormap_names=["cividis"],
            fallback_colormap="viridis",
        )
        self.assertEqual(name, "cividis")

    def test_parse_display_limits_and_log_scale_error(self):
        vmin, vmax, error = parse_display_limits(auto_scale=False, min_text="1", max_text="3")
        self.assertEqual((vmin, vmax, error), (1.0, 3.0, None))

        _vmin, _vmax, error = parse_display_limits(auto_scale=False, min_text="4", max_text="2")
        self.assertEqual(error, "Display min must be smaller than max.")

        self.assertEqual(
            log_scale_error(np.asarray([[1.0, 2.0], [3.0, 4.0]]), 0.0, 4.0),
            "Display min must be > 0 for log scale.",
        )
        self.assertIsNone(
            log_scale_error(np.asarray([[1.0, 2.0], [3.0, 4.0]]), 1.0, 4.0)
        )

    def test_update_sample_control_state_clears_and_normalizes_entry(self):
        entry = {"sample_index": 9, "stack_axis": 2, "stack_len": 9}
        state = update_sample_control_state(entry, axis=1, stack_len=4)
        self.assertTrue(state.enabled)
        self.assertEqual(state.maximum, 3)
        self.assertEqual(state.value, -1)
        self.assertEqual(entry["sample_index"], -1)
        self.assertEqual(entry["stack_axis"], 1)
        self.assertEqual(entry["stack_len"], 4)

        state = update_sample_control_state(entry, axis=None, stack_len=None)
        self.assertFalse(state.enabled)
        self.assertEqual(entry["sample_index"], None)
        self.assertEqual(entry["stack_axis"], None)
        self.assertEqual(entry["stack_len"], None)


if __name__ == "__main__":
    unittest.main()
