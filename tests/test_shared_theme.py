import unittest

from window.shared.theme import workflow_dialog_stylesheet


class SharedThemeTests(unittest.TestCase):
    def test_workflow_theme_styles_table_toggle_buttons(self):
        _theme, style = workflow_dialog_stylesheet("Dark")

        self.assertIn("QPushButton#tableToggleButton:checked", style)
        self.assertIn("background: #22c55e", style)
        self.assertIn("QPushButton#tableToggleButton:disabled", style)


if __name__ == "__main__":
    unittest.main()
