# run_napari.py
"""Entry point: launches napari with the Raman widget docked on the right."""
import napari

from napari_raman_widget import HardwareWidget


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.axes.visible = False  # napari-micromanager axes crash workaround
    widget = HardwareWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Raman", area="right")
    try:
        napari.run()
    finally:
        # aboutToQuit normally performs this first. The finally block covers
        # alternate event-loop exits and remains safe because cleanup is
        # idempotent.
        widget.shutdown_hardware()
