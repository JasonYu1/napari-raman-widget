"""Import-isolation checks for the selection package."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SelectionImportTests(unittest.TestCase):
    def test_public_exports_do_not_eagerly_load_segmentation_runtime(
        self,
    ) -> None:
        script = """
import sys

import napari_raman_widget.selection as selection
from napari_raman_widget.selection import automatic, grid, layers, manual

exports = (
    selection.automated_point_selections,
    selection.grid_point_selections,
    selection.create_point_sources,
    selection.center_manual_selections,
    selection.manual_point_selections,
)
implementations = (
    automatic.automated_point_selections,
    grid.grid_point_selections,
    layers.create_point_sources,
    manual.center_manual_selections,
    manual.manual_point_selections,
)
assert all(public is implementation for public, implementation in zip(
    exports,
    implementations,
))
unexpected = {"raman_mda_engine", "cellpose", "torch"}.intersection(
    sys.modules
)
assert not unexpected, f"eagerly imported: {sorted(unexpected)}"
"""
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
