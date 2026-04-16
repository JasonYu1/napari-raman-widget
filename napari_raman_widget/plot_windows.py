"""Pop-up matplotlib windows used to display calibration, spectra, and scans."""
import numpy as np
from qtpy.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget


class CalibrationPlotWindow(QMainWindow):
    """Pop-up showing max projection of calibration images with point overlay."""

    def __init__(self, ds, title="Calibration result"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 650)

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        central = QWidget()
        layout = QVBoxLayout(central)
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        ax = self.fig.add_subplot(111)

        imgs = ds["imgs"].max(axis=0)
        ax.imshow(np.asarray(imgs))

        X = ds.dims["X"]
        Y = ds.dims["Y"]

        pix_BF = np.asarray(ds["rel_BF_pos"])
        ax.scatter(pix_BF[:, 0], pix_BF[:, 1], color="r", s=20)

        ax.set_title(title)
        self.fig.tight_layout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)


class SpectrumWindow(QMainWindow):
    """Pop-up plot window with a toggle between mean and all-traces views."""

    def __init__(self, spec, title="Spectrum"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 550)
        self.spec = np.asarray(spec)
        self._show_mean = True

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        central = QWidget()
        layout = QVBoxLayout(central)

        self.toggle_btn = QPushButton("Show all traces")
        self.toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self.toggle_btn)

        self.fig = Figure(figsize=(7, 4.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setCentralWidget(central)
        self._redraw()

    def _toggle(self):
        self._show_mean = not self._show_mean
        self.toggle_btn.setText(
            "Show all traces" if self._show_mean else "Show mean"
        )
        self._redraw()

    def _redraw(self):
        from cns_control.utils import filter_mean
        import matplotlib.cm as cm

        self.ax.clear()
        if self._show_mean:
            self.ax.plot(filter_mean(self.spec))
        else:
            n = self.spec.shape[0]
            colors = cm.viridis(np.linspace(0, 1, n))
            for i in range(n):
                self.ax.plot(self.spec[i], color=colors[i], linewidth=0.8)
        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title(self.windowTitle())
        self.fig.tight_layout()
        self.canvas.draw_idle()


class ReferenceSpectraWindow(QMainWindow):
    """Pop-up showing reference spectra colored by z, with a colorbar."""

    def __init__(self, all_raman, zs, title="Reference spectra"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 600)

        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.cm as cm
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        from matplotlib.colors import Normalize
        from cns_control.utils import filter_mean

        central = QWidget()
        layout = QVBoxLayout(central)
        self.fig = Figure(figsize=(8, 5.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        ax = self.fig.add_subplot(111)

        zs = np.asarray(zs)
        n = len(zs)
        norm = Normalize(vmin=float(zs.min()), vmax=float(zs.max()))
        cmap = cm.viridis

        for i in range(n):
            color = cmap(norm(zs[i]))
            ax.plot(filter_mean(all_raman[i]), color=color, linewidth=0.9)

        ax.set_xlabel("Pixels")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(title)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax)
        cbar.set_label("z (um)")

        self.fig.tight_layout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)


class GridScanPlotWindow(QMainWindow):
    """Pop-up showing BF, end_BF, any extra channels, and the mean spectrum."""

    FIXED = ["BF", "end_BF"]

    def __init__(self, ds, title="Grid scan result"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 700)

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        skip = {"laser_pos", "grid_pos", "specs"}
        image_vars = [
            name for name in ds.data_vars
            if name not in skip and ds[name].ndim == 2
        ]
        ordered = [c for c in self.FIXED if c in image_vars]
        ordered += [c for c in image_vars if c not in self.FIXED]

        ncols = max(len(ordered), 1)
        self.fig = Figure(figsize=(3 * ncols + 1, 7))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        gs = self.fig.add_gridspec(2, ncols, height_ratios=[2, 1], hspace=0.4)
        first_ax = None
        for i, name in enumerate(ordered):
            ax = self.fig.add_subplot(
                gs[0, i], sharex=first_ax, sharey=first_ax,
            )
            if first_ax is None:
                first_ax = ax
            cmap = "gray" if name in self.FIXED else None
            ax.imshow(np.asarray(ds[name].values), cmap=cmap)
            ax.set_title(name)

        ax_spec = self.fig.add_subplot(gs[1, :])
        specs = np.asarray(ds["specs"].values)
        ax_spec.plot(specs.mean(axis=0))
        ax_spec.set_xlabel("Pixels")
        ax_spec.set_ylabel("Mean intensity (a.u.)")
        ax_spec.set_title(f"Mean spectrum ({specs.shape[0]} points)")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        self.fig.tight_layout(h_pad=2.0)