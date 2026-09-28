import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from napari_raman_widget.engine_compat import (
    make_pillar_shape_filter,
    pillar_suppression_kwargs,
)


class EngineCompatibilityTests(unittest.TestCase):
    def test_disabled_selection_suppression_does_not_import_engine_feature(self):
        with patch(
            "napari_raman_widget.engine_compat.import_module"
        ) as import_module:
            self.assertIsNone(make_pillar_shape_filter(False))

        import_module.assert_not_called()

    def test_enabled_selection_suppression_constructs_engine_feature(self):
        instance = object()
        filter_type = Mock(return_value=instance)
        aiming = SimpleNamespace(PillarShapeFilter=filter_type)

        with patch(
            "napari_raman_widget.engine_compat.import_module",
            return_value=aiming,
        ) as import_module:
            self.assertIs(make_pillar_shape_filter(True), instance)

        import_module.assert_called_once_with("raman_mda_engine.aiming")
        filter_type.assert_called_once_with()

    def test_old_engine_reports_missing_selection_suppression(self):
        with patch(
            "napari_raman_widget.engine_compat.import_module",
            return_value=SimpleNamespace(),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "does not support pillar suppression",
            ):
                make_pillar_shape_filter(True)

    def test_supported_engine_receives_true_and_false(self):
        class NewEngine:
            def __init__(self, *, suppress_pillars=False):
                pass

        self.assertEqual(
            pillar_suppression_kwargs(NewEngine, True),
            {"suppress_pillars": True},
        )
        self.assertEqual(
            pillar_suppression_kwargs(NewEngine, False),
            {"suppress_pillars": False},
        )

    def test_old_engine_is_compatible_while_option_is_off(self):
        class OldEngine:
            def __init__(self, *, segment_and_track=False):
                pass

        self.assertEqual(pillar_suppression_kwargs(OldEngine, False), {})
        with self.assertRaisesRegex(
            RuntimeError,
            "does not support pillar suppression",
        ):
            pillar_suppression_kwargs(OldEngine, True)

    def test_engine_accepting_extra_keywords_is_supported(self):
        class FlexibleEngine:
            def __init__(self, **kwargs):
                pass

        self.assertEqual(
            pillar_suppression_kwargs(FlexibleEngine, True),
            {"suppress_pillars": True},
        )


if __name__ == "__main__":
    unittest.main()
