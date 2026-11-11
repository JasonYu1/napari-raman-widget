import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectDependencyTests(unittest.TestCase):
    def test_racer_io_and_zarr_stack_are_pinned_compatibly(self) -> None:
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn(
            '"racer-raman[io] @ '
            "git+https://github.com/JasonYu1/racer-raman.git@"
            'a640a53362880fb3010bafa7d735c8a9851830ec"',
            project,
        )
        self.assertIn('"pandas>=2"', project)
        self.assertIn('"zarr>=2.15,<3"', project)

    def test_raman_control_uses_the_requested_revision(self) -> None:
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn(
            '"raman-control @ '
            "git+https://github.com/JasonYu1/raman-control.git@"
            '5cc3e46f2faa1075806b2f16b1bd80deba5c4b2c"',
            project,
        )


if __name__ == "__main__":
    unittest.main()
