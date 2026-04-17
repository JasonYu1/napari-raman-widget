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

class DatasetViewerWindow(QMainWindow):
    """Interactive viewer: BF image with laser scatter + spectrum, Qt sliders for t/p/z."""

    def __init__(self, df, da, title="Dataset viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 650)
        self.df = df
        self.da = da
        self.bf = da.sel(c=0).values  # (t, p, z, y, x)
        self._pt_selected = 0

        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.cm as mcm
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        from qtpy.QtWidgets import QSlider, QHBoxLayout, QVBoxLayout, QLabel

        self._mcm = mcm

        central = QWidget()
        main_layout = QVBoxLayout(central)

        # --- Sliders ---
        self.t_vals = da.coords["t"].values
        self.p_vals = da.coords["p"].values
        self.z_vals = da.coords["z"].values

        slider_layout = QHBoxLayout()
        self.t_slider, self.t_label = self._make_slider(
            "t", 0, len(self.t_vals) - 1
        )
        self.p_slider, self.p_label = self._make_slider(
            "p", 0, len(self.p_vals) - 1
        )
        self.z_slider, self.z_label = self._make_slider(
            "z", 0, len(self.z_vals) - 1
        )
        for label, slider in [
            (self.t_label, self.t_slider),
            (self.p_label, self.p_slider),
            (self.z_label, self.z_slider),
        ]:
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider, 1)
        main_layout.addLayout(slider_layout)

        # --- Figure ---
        self.fig = Figure(figsize=(12, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_img = self.fig.add_subplot(121)
        self.ax_spec = self.fig.add_subplot(122)

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        # --- Initial draw ---
        t0 = int(self.t_vals[0])
        p0 = int(self.p_vals[0])
        z0 = int(self.z_vals[0])

        self.im = self.ax_img.imshow(
            self.bf[0, 0, 0], cmap="gray", aspect="equal"
        )
        self.ax_img.set_title(f"t={t0}, p={p0}, z={z0}")

        # Scatter
        try:
            sub = df.loc[t0, p0, z0]
            n = len(sub)
            offsets = np.c_[sub["Y"].to_numpy(), sub["X"].to_numpy()]
        except KeyError:
            n = 0
            offsets = np.empty((0, 2))

        self.scat = self.ax_img.scatter(
            offsets[:, 0] if n else [],
            offsets[:, 1] if n else [],
            c=np.arange(n) if n else [],
            cmap="viridis",
            vmin=0,
            vmax=max(n - 1, 1),
            picker=5,
        )

        # Spectrum line
        try:
            spec0 = df.loc[t0, p0, z0, 0].values[:-3]
        except KeyError:
            spec0 = np.zeros(100)
        colors = self._pt_colors(max(n, 1))
        (self.spec_line,) = self.ax_spec.plot(
            spec0, color=colors[0] if n else "C0"
        )
        self.ax_spec.set_xlabel("pixel")
        self.ax_spec.set_ylabel("intensity (a.u.)")
        self.ax_spec.set_title(f"pt={self._pt_selected}")

        self.fig.tight_layout()

        # --- Connect signals ---
        self.t_slider.valueChanged.connect(self._on_slider)
        self.p_slider.valueChanged.connect(self._on_slider)
        self.z_slider.valueChanged.connect(self._on_slider)
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def _make_slider(self, name, lo, hi):
        from qtpy.QtWidgets import QSlider, QLabel
        from qtpy.QtCore import Qt
        label = QLabel(f"{name}=0")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        slider.setValue(0)
        slider.valueChanged.connect(
            lambda v, n=name, lbl=label: lbl.setText(f"{n}={v}")
        )
        return slider, label

    def _pt_colors(self, n):
        if n <= 1:
            return self._mcm.viridis(np.array([0.5]))
        return self._mcm.viridis(np.linspace(0, 1, n))

    def _current_tpz(self):
        ti = self.t_slider.value()
        pi = self.p_slider.value()
        zi = self.z_slider.value()
        t = int(self.t_vals[ti])
        p = int(self.p_vals[pi])
        z = int(self.z_vals[zi])
        return t, p, z, ti, pi, zi

    def _on_slider(self, _=None):
        t, p, z, ti, pi, zi = self._current_tpz()

        # Update image
        self.im.set_data(self.bf[ti, pi, zi])
        self.im.set_clim(
            self.bf[ti, pi, zi].min(), self.bf[ti, pi, zi].max()
        )
        self.ax_img.set_title(f"t={t}, p={p}, z={z}")

        # Update scatter
        try:
            sub = self.df.loc[t, p, z]
            n = len(sub)
            offsets = np.c_[sub["Y"].to_numpy(), sub["X"].to_numpy()]
        except KeyError:
            n = 0
            offsets = np.empty((0, 2))

        self.scat.set_offsets(offsets)
        self.scat.set_array(np.arange(n))
        self.scat.set_clim(0, max(n - 1, 1))

        self._pt_selected = min(self._pt_selected, max(n - 1, 0))
        self._update_spectrum()
        self.canvas.draw_idle()

    def _on_pick(self, event):
        if event.artist is not self.scat:
            return
        self._pt_selected = int(event.ind[0])
        self._update_spectrum()
        self.canvas.draw_idle()

    def _update_spectrum(self):
        t, p, z, _, _, _ = self._current_tpz()
        pt = self._pt_selected
        try:
            y = self.df.loc[t, p, z, pt].values[:-3]
            n = len(self.df.loc[t, p, z])
        except KeyError:
            return
        self.spec_line.set_ydata(y)
        if len(y) != len(self.spec_line.get_xdata()):
            self.spec_line.set_xdata(np.arange(len(y)))
        self.spec_line.set_color(self._pt_colors(n)[min(pt, n - 1)])
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_spec.set_title(f"pt={pt}")