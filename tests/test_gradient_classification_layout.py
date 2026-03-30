import unittest
from unittest import mock

from window.plot_msmode import GradientClassificationDialog


class GradientClassificationLayoutTests(unittest.TestCase):
    def test_surface_views_layout_splits_both_hemispheres_into_two_rows(self):
        dialog = GradientClassificationDialog.__new__(GradientClassificationDialog)
        dialog._fsaverage_mesh = "fsaverage4"
        dialog._hemisphere_mode = "both"

        fake_assets = {
            "mesh_left": object(),
            "mesh_right": object(),
            "sulc_left": object(),
            "sulc_right": object(),
        }
        with mock.patch(
            "window.plot_msmode.GradientSurfaceDialog._get_surface_assets",
            return_value=fake_assets,
        ):
            rows = dialog._surface_views_layout()

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(len(rows[1]), 2)
        self.assertEqual(rows[0][0][4], "LH Lateral")
        self.assertEqual(rows[0][1][4], "LH Medial")
        self.assertEqual(rows[1][0][4], "RH Medial")
        self.assertEqual(rows[1][1][4], "RH Lateral")


if __name__ == "__main__":
    unittest.main()
