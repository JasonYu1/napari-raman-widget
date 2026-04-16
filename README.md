# napari-raman-widget

A napari dock widget for controlling the CNS Raman microscopy rig.

## What it does

Provides a single collapsible sidebar panel inside napari with sections for:

- Loading Micro-Manager config and transformer model
- Collecting Raman spectra at clicked points
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra with autofocus
- Running spatial Raman mapping (grid scan) over a shape
- Automated cell selection inside a mask
- Running a Raman MDA with fluorescence channels and Z stacks

All outputs (reference `.npy` files, `grid_scan_*.zarr`, recalibrated models,
the MDA writer directory) are written relative to the current working
directory - or an output folder you set in the Loading section, which is
switched to on connect.

## Install

```bash
pip install -e .
```

This assumes `pymmcore-plus`, `raman-control`, `raman-mda-engine`, and
`cns-control` are already installed in the same environment (they are not
on PyPI and must be installed from your internal/local sources).

## Run

From inside the repo, with the `micro-control-2` conda env active:

```bash
python run_napari.py
```

### One-click launcher (Windows)

A `launch_napari.bat` script is included for convenience. Double-click it to:

1. Activate the `micro-control-2` conda environment
2. Change into the repo directory
3. Launch napari with the widget

You can also pin it to the taskbar or Start menu:

1. Right-click `launch_napari.bat` -> Create shortcut.
2. Right-click the shortcut -> Properties, and prepend `cmd /c ` to the
   Target field so it becomes `cmd /c "<full path>\launch_napari.bat"`.
3. Optionally click Change Icon to give it a recognizable icon.
4. Right-click the shortcut -> Pin to taskbar (or drag it to the desktop).

If your conda install isn't at the default
`C:\Users\<you>\miniforge3\Scripts\activate.bat`, edit the `call` line in
`launch_napari.bat` to point at your actual `activate.bat`. You can find it
with `where conda` in any terminal.

## Structure

- `run_napari.py` - entry point; just launches napari with the widget.
- `launch_napari.bat` - Windows one-click launcher (activates env + runs script).
- `napari_raman_widget/widget.py` - the main `HardwareWidget` class.
- `napari_raman_widget/plot_windows.py` - matplotlib pop-up windows.
- `napari_raman_widget/log_window.py` - streaming stdout log window.
- `napari_raman_widget/ui_helpers.py` - small Qt helpers.

Hardware-library imports (`pymmcore_plus`, `cns_control`, etc.) are done
lazily inside methods, so the package imports cleanly on machines without
the rig.