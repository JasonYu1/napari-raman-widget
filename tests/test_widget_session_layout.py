"""Regressions for session-only inputs and workflow section placement."""

import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class WidgetSessionLayoutTests(unittest.TestCase):
    def _source(self, filename: str) -> str:
        return (PACKAGE / filename).read_text(encoding="utf-8")

    def test_dark_noise_starts_empty_and_has_session_clear_action(self):
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = self._source(filename)
                row_start = source.index("self.dark_noise_path = QLineEdit()")
                row_end = source.index(
                    "loading_layout.addLayout(dark_noise_layout)", row_start
                )
                row_source = source[row_start:row_end]

                self.assertNotIn("self.dark_noise_path.setText", row_source)
                self.assertIn("None (raw spectra only)", row_source)
                self.assertIn("self.clear_dark_noise_btn", row_source)
                self.assertIn(
                    "clicked.connect(self.clear_dark_noise)", row_source
                )
                self.assertIn("def clear_dark_noise(self):", source)
                self.assertIn("self.dark_noise_path.clear()", source)

    def test_hardware_does_not_load_or_persist_saved_dark_noise(self):
        source = self._source("hardware_widget.py")
        defaults_start = source.index("def _load_user_defaults(self):")
        defaults_end = source.index("# -------- file pickers", defaults_start)
        defaults_source = source[defaults_start:defaults_end]

        self.assertNotIn(
            '"dark_noise_file": self.dark_noise_path', defaults_source
        )
        self.assertIn('supported.add("dark_noise_file")', defaults_source)
        self.assertNotIn("_save_dark_noise_as_default", source)
        self.assertNotIn("update_hardware_defaults", source)
        self.assertIn("selected for this session", source)

    def test_pixel_to_stage_is_nested_under_stage_grid(self):
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = self._source(filename)
                self.assertIn(
                    "grid_layout.addWidget(px2stage_box)", source
                )
                self.assertIn('(\"Analysis\", [dataset_box])', source)
                self.assertNotIn(
                    "analysis_sections = [dataset_box, px2stage_box]", source
                )
                self.assertIn(
                    "Fit & save model in Generate stage grid", source
                )
        self.assertIn(
            "self.px2stage_box.hide()",
            self._source("demo_widget.py"),
        )

    def test_assistant_has_its_own_optional_workflow_tab(self):
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = self._source(filename)
                self.assertIn("if self.chat_panel is not None:", source)
                self.assertIn(
                    'workflow_groups.append(("Assistant", [self.chat_panel]))',
                    source,
                )
                self.assertIn('(\"Analysis\", [dataset_box])', source)
                self.assertNotIn(
                    "analysis_sections.append(self.chat_panel)", source
                )


if __name__ == "__main__":
    unittest.main()
