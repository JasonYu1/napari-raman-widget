import unittest

from napari_raman_widget.qt_messages import (
    _is_invisible_combo_mouse_grab_warning,
)


class QtMessageFilterTests(unittest.TestCase):
    def test_filters_only_invisible_combo_mouse_grab_warning(self):
        warning = (
            "QWindowsWindow::setMouseGrabEnabled: Not setting mouse grab for "
            "invisible window QWidgetWindow/"
            "'QComboBoxPrivateContainerClassWindow'"
        )
        self.assertTrue(_is_invisible_combo_mouse_grab_warning(warning))

        self.assertFalse(
            _is_invisible_combo_mouse_grab_warning(
                "QWindowsWindow::setMouseGrabEnabled: a different warning"
            )
        )
        self.assertFalse(
            _is_invisible_combo_mouse_grab_warning(
                "QWindowsWindow::setMouseGrabEnabled: Not setting mouse grab "
                "for invisible window QWidgetWindow/'SomeOtherWindow'"
            )
        )


if __name__ == "__main__":
    unittest.main()
