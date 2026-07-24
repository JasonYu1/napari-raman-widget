import unittest
from unittest.mock import patch

from napari_raman_widget import qt_messages
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

    def test_install_is_a_noop_without_a_qt_binding(self):
        original_installed = qt_messages._installed
        original_handler = qt_messages._previous_handler
        try:
            qt_messages._installed = False
            qt_messages._previous_handler = None
            with patch.dict("sys.modules", {"qtpy.QtCore": None}):
                qt_messages.install_qt_message_filter()
            self.assertFalse(qt_messages._installed)
            self.assertIsNone(qt_messages._previous_handler)
        finally:
            qt_messages._installed = original_installed
            qt_messages._previous_handler = original_handler


if __name__ == "__main__":
    unittest.main()
