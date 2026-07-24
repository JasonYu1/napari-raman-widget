import unittest

from napari_raman_widget.position_specs import (
    parse_position_spec,
    resolve_position_specs,
)


class PositionSpecTests(unittest.TestCase):
    def test_blank_keeps_existing_selection(self):
        self.assertIsNone(parse_position_spec("", 8))
        self.assertIsNone(parse_position_spec("None", 8))

    def test_single_integer_selects_every_nth_position(self):
        self.assertEqual(parse_position_spec("3", 8), [0, 3, 6])
        self.assertEqual(parse_position_spec("1", 4), [0, 1, 2, 3])

    def test_comma_selects_exact_positions(self):
        self.assertEqual(parse_position_spec("0, 2, 5", 8), [0, 2, 5])
        self.assertEqual(parse_position_spec("5,", 8), [5])

    def test_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            parse_position_spec("0", 8, "Autofocus positions")

    def test_explicit_positions_must_be_in_range(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            parse_position_spec("0,8", 8, "Imaging positions")

    def test_autofocus_interval_does_not_reduce_blank_imaging(self):
        autofocus, imaging = resolve_position_specs(
            "10", "", 25, range(25)
        )
        self.assertEqual(autofocus, [0, 10, 20])
        self.assertEqual(imaging, list(range(25)))

    def test_imaging_override_remains_independent(self):
        autofocus, imaging = resolve_position_specs(
            "10", "1,4,7", 25, range(25)
        )
        self.assertEqual(autofocus, [0, 10, 20])
        self.assertEqual(imaging, [1, 4, 7])


if __name__ == "__main__":
    unittest.main()
