"""Launch napari with the hardware-free Raman demonstration widget."""

import napari

from napari_raman_widget import DemoWidget


if __name__ == "__main__":
    viewer = napari.Viewer(title="napari Raman demonstration")
    viewer.axes.visible = False
    widget = DemoWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Raman Demo", area="right")
    napari.run()
