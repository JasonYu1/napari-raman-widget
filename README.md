# napari-raman-widget

[![CI](https://github.com/JasonYu1/napari-raman-widget/actions/workflows/ci.yml/badge.svg)](https://github.com/JasonYu1/napari-raman-widget/actions/workflows/ci.yml)

A napari dock widget for controlling the Raman microscopy rig.

![napari Raman widget showing the simulated microscope sample and Raman control panel](napari-raman-widget-screenshot.PNG)

*The widget running in demonstration mode with the simulated microscope.*

## What it does

Provides a sidebar with Setup, Selection, Acquire, and Analysis tabs, plus an
optional Assistant tab. Its collapsible sections cover:

- Loading Micro-Manager config and transformer model
- Collecting Raman spectra at clicked points
- Removing cosmic rays from FVB spectra and 2D detector images with
  [RACER](#cosmic-ray-removal-racer)
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra across an axial Z range
- Running spatial Raman mapping only inside a closed region of interest (ROI)
- Automated cell selection inside a mask
- Running a Raman MDA with fluorescence channels and Z stacks

The interface also includes:

- **Inline help** - every field has a hover tooltip explaining what it does,
  and a **Help** button at the top of the panel opens the full PDF user manual.
- **AI assistant** - a built-in chat box that maps plain-English commands to
  the panel's existing actions (see [AI assistant](#ai-assistant-chat-panel)).
- **Dockable plots** - spectra, detector images, scans, calibration views,
  datasets, and logs share the **Raman Plots** workspace, normally a native
  Napari tab alongside the Raman controls in the same dock area.
- **Persistent status** - the current acquisition status stays visible below
  the controls while you change tabs or scroll.

Relative outputs (reference `.zarr` datasets, `grid_scan_*.zarr`, recalibrated
models, and the MDA writer directory) resolve from the current **Active
folder**, shown in **Setup > Loading**. Absolute output paths stay unchanged.

## Video demonstrations

Watch the complete procedure series on the
[napari-raman-widget YouTube channel](https://www.youtube.com/@napari-raman-widget).
The videos cover software loading, hardware connection, point-based spectrum
collection, laser aiming calibration, axial background scanning, spatial
mapping, stage-grid generation, automated cell selection, Raman MDA, dataset
generation, and pixel-to-stage calibration.

<a id="supporting-your-hardware"></a>

<details>
<summary>Supporting your hardware</summary>

> **For most users, adapting the platform to different microscope hardware
> requires changes only in
> [`raman-control`](https://github.com/JasonYu1/raman-control).** Typically, you only need to
> update its **collector** to communicate with your Raman spectrometer or
> detector and its **DAQ implementation** to communicate with your DAQ board
> and connected devices. The napari widget and
> [`raman-mda-engine`](https://github.com/JasonYu1/raman-mda-engine) use the
> existing `raman-control` interface and should not require hardware-specific
> changes.

</details>

## Install

For normal use, install the widget directly from GitHub:

```bash
pip install git+https://github.com/JasonYu1/napari-raman-widget.git
```

This installs the declared dependencies automatically, including the
project-specific [`raman-control`](https://github.com/JasonYu1/raman-control)
and [`raman-mda-engine`](https://github.com/JasonYu1/raman-mda-engine) packages,
plus the pinned
[`racer-raman`](https://github.com/JasonYu1/racer-raman) cosmic-ray removal
implementation, from their public GitHub repositories.
The RACER dependency is installed as `racer-raman[io]` and imported as
`racer_raman`.
RACER's storage support uses Zarr 2, so Xarray is constrained to versions before
2026.4, which still support the Zarr 2 Python library.

<details>
<summary>Development installation</summary>

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/JasonYu1/napari-raman-widget.git
cd napari-raman-widget
pip install -e .
```

</details>

<details>
<summary>Optional AI assistant dependency</summary>

The AI assistant is optional. To enable it, also install the Anthropic SDK
and set an API key (see [AI assistant](#ai-assistant-chat-panel)):

```bash
pip install anthropic
```

</details>

## Run

From inside the repo, with your conda environment active:

```bash
python run_napari.py
```

For the hardware-free simulated microscope, use the dedicated demo launcher:

```bash
python launch_demo_napari.py
```

The demo launcher connects automatically and does not require a Micro-Manager
configuration or coordinate-transform model.
On Windows, you can also double-click `launch_demo_napari.bat`.

## Usage guide

Expand a topic for instructions, settings, and workflow details.

The control sidebar groups loading and calibration under **Setup**, stage-grid
and cell tools under **Selection**, measurement workflows under **Acquire**,
and dataset generation under **Analysis**. Pixel-to-stage calibration is inside
**Selection → Generate stage grid**.
The optional AI chat has its own **Assistant** tab.

<a id="changing-the-output-folder"></a>

<details>
<summary>Changing the output folder</summary>

In **Setup > Loading**, type or browse to an output folder, then click
**Apply folder**. The entered path is a draft until applied; **Active folder**
shows where future relative outputs will go. Applying creates the folder if
needed and validates it without reconnecting hardware. Hardware **Connect**
also applies the selected folder automatically. The same Apply control is
available in demo mode.

Stop acquisitions, camera/Raman live mode, and MDA before applying: a running
operation in any registered Raman or demo widget blocks the change. This
changes the **process-wide working directory**, so it also affects relative
paths used by other plugins in the same Napari process. Existing files are
not moved, and absolute output paths are unchanged. Existing relative input
file selections are converted to absolute paths when the files exist, so they
continue to refer to the same files after a folder change.

</details>

<a id="arranging-plots-and-controls"></a>

<details>
<summary>Arranging plots and controls</summary>

Plots and logs initially open in a native **Raman Plots** tab alongside their
Raman controls in the same Napari dock area. Demo results similarly use
**Raman Demo Plots** alongside the demo controls. New results automatically
bring the plots workspace and its newest result to the front. If the controls
are floating or standalone, the workspace may open separately instead of
joining them as a tab.

Inside the workspace, each result has its own tab. Drag these tabs to reorder
them, and hover over a tab to see its complete acquisition title. The **Plots**
button at the top of the Raman controls brings the workspace back if hidden.

The workspace uses Napari's native title-bar controls, just like the Raman
controls panel: **close**, **hide** (the minimize-style icon), and **float**.
Drag the floating window's title bar to an edge of Napari to dock it again;
double-click the title bar to toggle floating. Floating windows use the
platform's window decorations, so their icons can differ from docked panels.
Move or float the plots independently of the controls. New results and
hiding/reopening the workspace preserve your chosen location and floating
state for the current session. Hiding or closing the workspace keeps all results
available through **Plots**; the close button on an individual tab closes
that result. **Plots** also restores a minimized floating workspace.
Closing a live spectrum tab requests a stop after the current exposure;
hiding or floating the workspace lets acquisition continue.

Each spectral result has compact display controls and an expandable
**Processing** section for RACER cosmic-ray removal, baseline subtraction,
and smoothing. These settings change the display only; acquired and saved
data stay unchanged. Dataset navigation separates time, position, and Z from
the spectral processing controls. Repeated FVB spectra and detector-image
collections also provide a repeat slider: **Mean** preserves the aggregate
view, while positions `1…N` display an individual acquired spectrum or frame.
For detector images, the selected repeat applies to both the image and its
row-sum spectrum view.

Plot backgrounds are transparent by default so they blend into Napari's theme.
Check **White background** on a plot for an opaque white canvas; uncheck it
to restore transparency. Axes and labels adjust for readability.

The compact display row groups **Fix Y scale** (where available), **Show
wavenumber**, and **White background**. Spectra always start in pixels.
Wavenumber calibration is optional and is never loaded automatically from
saved defaults: select a calibration JSON in Setup, then opt in with **Show
wavenumber** on a new plot. The checkbox is disabled without a calibration.

Laser aiming calibration has a progress bar and stage readout in its log.
In the calibration result, click a calibration point to inspect its stored
spectrum alongside the image; this does not acquire new data.

</details>

<a id="cosmic-ray-removal-racer"></a>

<details>
<summary>Cosmic-ray removal (RACER)</summary>

**Remove cosmic rays (RACER)** is off by default and remains disabled until a
matching session calibration is loaded from **Loading → Load RACER
calibration…**. Choose a closed-shutter Zarr stack and a separate illuminated
open-reference Zarr stack when desired. For FVB/RACER-1D, cancelling the
second file chooser uses the closed-shutter variance alone. RACER-2D requires
the illuminated reference. When both stores are used, collect them with the
same exposure, detector readout, binning, gain, region of interest, and detector
geometry as the measurements.
RACER-1D is available for full-vertical-binning spectra and RACER-2D for
detector images; single-track correction is not enabled because that readout is
not currently validated by RACER. The loaded calibration is held in memory for
this session and can be cleared without changing the existing dark-noise
selection.

<details>
<summary>Collecting calibration data</summary>

To collect and use a calibration end to end:

1. Select **FVB** or **Image**, set the exposure and repeat count, physically
   close the shutter or block excitation, and click **Collect dark noise**. The
   hardware workflow closes its configured excitation `Fluoshutter`, but that
   does not prove that the detector sees a fully dark optical path; physically
   verify that excitation and ambient light are blocked. Demo mode has no
   physical shutter.
2. FVB collection writes two files with the same generated stem in the current
   working directory: a `.npy` file selected for the existing mean-bias
   subtraction and a `.zarr` closed-shutter stack for RACER. Image collection
   writes the RACER `.zarr` stack and opens an image result, but does not replace
   the selected spectral-bias `.npy` file. Single-track remains `.npy` bias-only
   because RACER does not validate that readout mode.
3. Optionally for FVB, reopen the shutter, place a stable representative open
   reference, keep exposure, readout, binning, gain, and region of interest
   unchanged, and use a finite **Collect spectra at selected point** acquisition
   with **Save as** filled in. This creates the separate open-reference `.zarr`
   stack. This step is required for Image/RACER-2D.
4. In **Loading**, click **Load RACER calibration…**, choose the generated
   closed-shutter Zarr first, then either choose the illuminated reference or
   cancel the second chooser to use dark-only variance for FVB. Acquire the
   matching sample data, open **Processing**, and check **Remove cosmic rays
   (RACER)**.

RACER requires at least three calibration frames, but that is an algorithmic
minimum rather than a useful acquisition target. As a practical starting point,
the upstream measured-data examples use 1,000 closed-shutter frames for both
readouts, 1,000 open-reference spectra for FVB, and 100 open-reference frames
for Image. Use enough frames to sample real particle events and detector noise;
calibration quality is more important than fitting from the minimum count.
Both generated stores include the detector shape/readout signature, ADU units,
and the RACER default detector assumptions (`0.7 e-/ADU`, saturation at
`65535 ADU`) so mismatched pairs can be rejected. These gain values are marked
as assumptions in the metadata; verify them for the connected detector before
using the correction for quantitative work.

</details>

<details>
<summary>Loading and remembering reference files</summary>

The **Closed shutter** and **Open reference** path fields are remembered when
you browse for a Zarr directory or finish editing a path. On reopening either
widget, the paths are restored and the calibration stays unloaded. Click
**Load RACER calibration…** to load those files without selecting them again.
Use each field's browse button to change files; an empty open-reference field
selects dark-only variance for FVB. **Clear** unloads the active calibration
while keeping the remembered paths. Missing or incompatible files are reported
only when you explicitly load them.

</details>

<details>
<summary>Advanced calibration settings</summary>

Use **Loading → Advanced calibration settings…** to change RACER's fit
parameters before loading calibration files. Separate tabs cover FVB dark
calibration, FVB reference variance, and Image calibration, including the
**Trace half-width (rows)**. Values start at the installed RACER defaults;
**Save settings** remembers your values in local app preferences and restores
them when either widget opens again. **Restore Defaults → Save settings**
remembers the reset. Changes take effect on the next calibration load, so reload
an active calibration after changing them. Exposure, detector gain, and
saturation continue to come from the stores.

</details>

<details>
<summary>Reference variance and correction behavior</summary>

For FVB/RACER-1D, the illuminated open-reference spectra are **not** used as a
2-D optical row profile. They measure the per-column illuminated master and
variance at the reference's signal level, which helps avoid mistaking ordinary
fluctuations for particle events. This is not a fitted variance-versus-intensity
curve: the reference should be stable and representative of the count levels in
the samples being corrected. RACER-2D additionally uses its open-reference
images to fit the optical row profile. If no open reference is selected for
RACER-1D, the widget fits RACER with the closed-shutter stack as both inputs so
the fixed variance comes only from the dark acquisition. This avoids
transferring a reference sample's peak-specific variance, but it also omits
signal-dependent shot noise and can over-flag bright sample features. The
Spectrum view therefore enables dark-only FVB correction only when at least two
same-point repeats are available, and restores narrow features that agree with
the repeat consensus while retaining one-off particle corrections. A single
FVB spectrum still requires a genuine illuminated FVB reference because one
narrow peak cannot be distinguished safely from one cosmic event.
For detector images, RACER's optical-trace correction is complemented by a
conservative dark-calibrated cleanup for compact off-trace events; broad
off-trace structures are left unchanged.
The generic Dataset viewer remains disabled for RACER because imported datasets
do not currently carry the exposure and detector-signature evidence needed for
safe matching.

RACER corrects each raw trace or detector frame before repeat averaging. The
display pipeline then applies optional dark-noise bias subtraction, baseline
subtraction, and smoothing. A calibration is never inferred from the data
being corrected, and incompatible exposure or detector geometry leaves the
checkbox disabled. The Processing section reports how many detector values
were actually changed and how many repeat-consistent FVB values were preserved.
See
[racer-raman](https://github.com/JasonYu1/racer-raman) for calibration guidance.

</details>

The pinned RACER dependency includes an
[MIT license](https://github.com/JasonYu1/racer-raman/blob/a640a53362880fb3010bafa7d735c8a9851830ec/LICENSE).

</details>

<a id="dark-noise-bias-subtraction"></a>

<details>
<summary>Dark-noise bias subtraction</summary>

Dark noise starts as **None** whenever a widget opens, including when an older
defaults file contains a dark-noise path. Select or collect a dark-noise file
to use it for the current session, or click **Clear** to return to None.
This optional mean-bias subtraction is separate from RACER: a dark-noise
`.npy` file by itself is not a RACER calibration and is never treated as one.

</details>

<a id="spatial-mapping-inside-an-roi"></a>

<details>
<summary>Spatial mapping inside an ROI</summary>

Spatial mapping always samples **inside the actual closed ROI**, not its
bounding box. Use a rectangle, ellipse, or polygon in the active Shapes layer;
concave polygons and rotated or sheared shapes are supported. Select only one
ROI. If no shape is selected, the most recently drawn shape is used; selecting
several shapes is rejected. Open lines and paths are not scan regions.

Alternatively, select a **2D Labels layer** and click **Preview grid scan**.
All non-zero label IDs are included together, regardless of the selected label
or selected-label-only display setting. Zero-valued background, holes, and gaps
between objects are excluded. This creates a regular sampling grid inside the
labels, not one centroid per object. The original Labels layer is never changed.

Labels must have the same pixel dimensions and pixel-to-world transform as the
camera image. Shared image/label scaling or rotation is supported; mismatched
or 3D label grids are rejected with an alignment message. The target point count
is across **all labels together per Z plane**, not per object. Coarse spacing can
miss small objects; the preview reports how many label IDs have grid points so
you can increase the target or reduce spacing when needed.

After confirming Start, a read-only **Spatial map points** layer displays the
exact points used for acquisition, with a `label_id` feature for each point.
Cancelling the preview does not create a layer. Saved scans include `label_id`
alongside `grid_pos` and record the source label IDs and all-nonzero selection
rule; keep the original Labels layer to retain the complete segmentation mask.

Choose one of two **Sampling** modes:

- **Total points:** enter **Target points per Z plane** (default 400).
  The widget automatically calculates one uniform X/Y grid spacing to obtain
  approximately that number of points inside the ROI. This is a target, not an
  exact count or a grid side: clipping the grid to the shape changes how many
  points fit. Review the target, actual count, and calculated spacing in the
  preview before starting.
- **Pixel spacing:** enter equal X/Y spacing in camera pixels (default 10 px).
  A square lattice is clipped to the ROI, so its shape and spacing determine
  the actual number of points. The preview reports that count before starting.

Both modes keep all sampling points on a **uniform square lattice** with the
same X and Y spacing. Total-points mode does not add scattered points or move
individual points to force an exact count.

Each mode requires at least two points per Z plane for the acquisition/DAQ
path and supports at most 250,000. Increase the sampling density if a spacing
choice leaves fewer than two points inside the ROI. The old `N × N` grid-side
input is no longer used. Both modes work the same way in the hardware and demo
widgets.

Spacing is measured in **camera-image pixels**, after converting Shapes-layer
coordinates through the napari layer transforms into the reference image.
Keep an unambiguous visible 2D Image layer whose dimensions match the camera.
If multiple matching images are visible, they must have the same transform;
hide unrelated images if their transforms differ. A visible image with no
matching camera-sized reference produces an error rather than a guessed
conversion. Without any visible image, only an untransformed ROI already in
raw camera-pixel coordinates is accepted. The preview identifies the pixel
reference and rejects sample points outside the camera image.

The preview overlays the planned points and ROI outline. Saved scan datasets
retain the actual point coordinates in `grid_pos`, the ROI shape type and
image-pixel vertices in `roi`, and the pixel-reference/source-layer names.
Metadata also records `sampling_mode`, actual `points_per_z`, the target
`requested_points_per_z` when applicable, and `point_spacing_px` (including
automatically calculated spacing), plus planned/completed spectrum counts.
This preserves the selected region and sampling settings with the data.

</details>

<a id="raman-band-and-ratio-maps"></a>

<details>
<summary>Raman band and ratio maps</summary>

Open a spatial scan result and expand **Raman map**. Select **Band A area**
and set its lower and upper spectral bounds, or choose **Band A / B ratio**
and also set Band B's bounds. Click **Apply map** to color the measured points
over the brightfield image, with a colorbar. **Points only** restores the
standard point display and is the default for new results.

Use **Colormap** and **Reverse colors** to choose the map's colors. Changes
update an existing overlay and colorbar immediately, without clicking
**Apply map** again or changing values or color limits. Your color choice
persists across Z planes and map modes within that result tab; invalid points
remain gray.

Bounds use spectral pixels by default. To use cm⁻¹, load an optional
wavenumber calibration and enable **Show wavenumber**. Band areas integrate
the raw saved spectra using the trapezoidal rule; units are intensity × pixel
or intensity × cm⁻¹, while ratios are dimensionless. **Processing** settings
still affect only the displayed spectrum, not these map values.
Changing bounds or axis units clears the old map; click **Apply map** again.
Unit changes preserve the selected detector-bin range. Orange and blue bands
on the spectrum indicate A and B respectively.

Click a point to inspect its spectrum, and use the Z controls to review other
planes. The color scale automatically adjusts to the current Z plane.
Maps color only measured points: they do not interpolate across gaps,
holes, or unsampled regions. Invalid values, including non-finite samples or
a ratio denominator at or below 10⁻¹², appear gray and are counted in the map
status. Mapping does not acquire data or change the saved spectra, and a band
map alone does not identify a chemical species.

</details>

<a id="raman-mda-settings-and-pre-run-review"></a>

<details>
<summary>Raman MDA settings and pre-run review</summary>

In **Acquire > Run Raman MDA**, **Advanced** starts collapsed and holds
autofocus search ranges/points, autofocus and imaging positions,
segment-and-track options, and the refocus/segmentation cadence. Collapsing
it preserves settings. Output, Raman glass offset, exposure, loops, interval,
Z settings, extra channels, and Run/Stop remain outside Advanced.

**Run Raman MDA** opens a pre-run review before creating output or starting
hardware. It shows prepared stage positions, current cell/target counts,
Raman pattern collections, time points, exposure, Z planes, and the resolved
output folder. **Start acquisition** is disabled when setup is missing,
settings are invalid, or another acquisition/live mode is running.
**Back to settings** is the default and keeps your settings without starting.
The duration is a lower bound from current cell Raman exposures and scheduled
intervals, not a predicted completion time: imaging, autofocus, tracking,
motion, readout, and saving add time. Non-empty output folders and intervals
shorter than the Raman exposure workload are flagged. The confirmed review
is also copied to the MDA log.

</details>

<a id="reviewing-and-stopping-acquisitions"></a>

<details>
<summary>Reviewing and stopping acquisitions</summary>

Before a **grid scan** or **axial background scan**, a review dialog shows the
selected ROI or point, sampling settings, target versus actual points per Z plane, Z
positions, repeats, total spectrum count, and exact output path. For spatial
mapping, total spectra equals the actual ROI point count multiplied by the
number of Z planes. The preview uses copied settings and does not
acquire data. Choose **Start scan** to proceed or **Cancel** to return; Cancel
is the default keyboard action. Large grids show a sampled display while the
summary still counts every planned acquisition point.

Starting a spatial grid scan automatically turns off camera live imaging
before changing channels/exposure or taking the first image. Demo live mode
and its image timer stop too. Cancelling the preview leaves live mode unchanged;
live imaging stays off after the scan, and a live-stop failure prevents the
scan from starting.

The duration estimate is an **exposure-only lower bound**, not a completion
prediction. It excludes stage movement, settling, readout, autofocus, processing,
and saving. Grid Z offsets are measured from the initial Z minus the configured
Z offset. An axial background scan measures the configured evenly spaced Z
positions around the initial Z, with the chosen repeats at each position; it
does not run an additional fine scan or automatically choose a focus.

**Dark-noise collections, finite selected-point collections, laser aiming
calibration, grid scans, and axial background scans** run as background jobs in
both the hardware and demo widgets. A task strip below the controls shows
actual completed work, the current stage, elapsed time, **Stop**, and **Log**.
The corresponding log also shows progress and a Stop control.
Progress advances when a measured batch or processing stage finishes; it does
not simulate detector progress during an exposure. Conflicting widget controls
are unavailable until the job finishes.

**Stop is cooperative, not an emergency stop.** It waits for the current
hardware step or acquisition batch to finish before cleanup; a long exposure
or repeated-spectrum batch can take time. Keep the application open while it
shows that a stop is pending. Hiding a log does not stop acquisition.
An interrupted dark-noise or selected-point collection is discarded rather
than saved or plotted as if all requested repeats completed.

When a requested stop completes normally, grid and axial scans save any
completed measurements to the previewed output path. These datasets carry
`acquisition_status="stopped"`; an uninterrupted scan uses `"complete"`.
Axial data contains only measured Z planes and records `planned_z_planes`.
Partial grid data stores only acquired samples, with `sample_grid_index` and
`sample_z_index` locating them in the planned grid/Z coordinates, and records
`completed_spectra` and `planned_spectra`. No unmeasured spectra are filled in.
If nothing was measured, no dataset is saved. A calibration cancelled before
saving is not saved or installed as a finished calibration. Hardware or storage
failures can prevent saving, so inspect the final status and log.

Logs hide routine successful CCD progress tuples by default. Check **Show
camera diagnostics** to reveal the retained diagnostic lines; warnings and
errors remain visible. This only filters the widget log: original terminal
output is unchanged.

Selection/refinement workflows and Raman MDA are not part of this task strip;
MDA and live spectrum collection retain their existing stop controls.

</details>

## Configuration and help

<a id="one-click-launcher-windows"></a>

<details>
<summary>One-click launcher (Windows)</summary>

A `launch_napari.bat` script is included for convenience. Double-click it to:

1. Activate your conda environment
2. Change into the repo directory
3. Launch napari with the widget

Before using it, edit `launch_napari.bat` to match your setup:

- The `call ... activate.bat <env-name>` line: replace `<env-name>` with
  your own conda environment name.
- The `cd /d <repo-path>` line: replace with the path to your local clone
  of this repo.

You can also pin the launcher to the taskbar or Start menu:

1. Right-click `launch_napari.bat` -> Create shortcut.
2. Right-click the shortcut -> Properties, and prepend `cmd /c ` to the
   Target field so it becomes `cmd /c "<full path>\launch_napari.bat"`.
3. Optionally click Change Icon to give it a recognizable icon.
4. Right-click the shortcut -> Pin to taskbar (or drag it to the desktop).

</details>

<a id="hardware-defaults-file"></a>

<details>
<summary>Hardware defaults file</summary>

To prefill machine-specific configuration, model, wavelength, and selection
fields whenever the hardware widget opens:

1. Copy `napari_raman_defaults.example.json` to
   `napari_raman_defaults.json` in the repository root.
2. Edit the copied file with your paths and preferred values.
3. Restart napari.

The local defaults filename is ignored by Git. Relative paths inside it are
resolved relative to the defaults file. To keep the file elsewhere, set the
`NAPARI_RAMAN_DEFAULTS` environment variable to its full path before launch.

</details>

<a id="slack-alerts-for-raman-mda-failures"></a>

<details>
<summary>Slack alerts for Raman MDA failures</summary>

Slack alerts are optional and are disabled when no webhook is configured.
Set an incoming-webhook URL in the environment before launching napari:

```powershell
$env:NAPARI_RAMAN_SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
python run_napari.py
```

For a persistent Windows setting, use `setx` and open a new terminal before
launching napari:

```powershell
setx NAPARI_RAMAN_SLACK_WEBHOOK_URL "https://hooks.slack.com/services/..."
```

Do not put the webhook URL in Git or in the shared defaults JSON. The hardware
widget sends a channel alert with the traceback only when Raman MDA setup or
its background acquisition thread raises an exception. Ordinary Python
warnings remain visible in the log, continue using the normal warning rules,
and do not trigger Slack.

Other synchronous operations can use the same helper directly:

```python
from napari_raman_widget.slack_notifications import (
    send_slack_message,
    set_webhook_url,
    slack_notify,
)

@slack_notify
def my_operation():
    ...
```

`set_webhook_url(...)` provides a process-local override when environment
configuration is not convenient, and `send_slack_message(...)` sends a manual
message. Webhook/network failures are logged but never replace the original
microscope exception.

</details>

<a id="inline-help-and-manual"></a>

<details>
<summary>Inline help and manual</summary>

Two forms of built-in documentation ship with the panel:

- **Hover tooltips.** Every editable field and button carries a short
  description shown on mouse-over. The text lives in one central dictionary
  in `napari_raman_widget/field_help.py` (keyed by widget attribute name) and
  is attached in one call, `apply_tooltips(self)`, at the end of the widget's
  constructor. To reword a tooltip, edit that dictionary only - no other file
  changes are needed. The wording is kept consistent with the PDF manual.
- **User manual.** A **Help** link at the top-right of the panel opens
  `napari_raman_widget/resources/napari-raman-widget-manual.pdf`, a detailed
  guide to every section, the layer each workflow consumes, what must be
  prepared first, and what output is produced. The manual's LaTeX source is
  kept alongside it so it can be regenerated.

</details>

## AI assistant (chat panel)

The optional **Assistant** tab controls existing widget actions through chat.

<details>
<summary>Setup</summary>

1. `pip install anthropic`
2. Set an Anthropic API key in the environment **before** launching napari,
   e.g. on Windows: `setx ANTHROPIC_API_KEY "sk-ant-..."` (open a new
   terminal afterwards), or per-session `$env:ANTHROPIC_API_KEY = "sk-ant-..."`.
3. Set `MODEL` at the top of `chat_panel.py` to a model your account can
   access.

The assistant is created automatically when `HardwareWidget` opens; no widget
source changes are required.

</details>

<details>
<summary>Console, commands, and capabilities</summary>

The **Assistant** tab is a full-height, terminal-style console inside the
widget. Type directly at the `> ` prompt below the conversation—there is no
separate typing box or Send button. **Enter** sends, **Shift+Enter** inserts
a new line, and **Up/Down** recalls submitted requests. You can select and
copy earlier output, but cannot accidentally edit it. Pasting multiple lines
does not submit them; review the text and press Enter when ready.

This is an AI command interface, **not a system shell or Python terminal**.
`napari_raman_widget/chat_panel.py` turns plain-English requests into the
panel's existing GUI actions. It never
touches hardware directly - it drives the same methods the buttons call (and
a few napari / Micro-Manager operations), so it reuses every existing range
check and validation. Anything that moves the stage, laser, or shutters pops
a confirmation dialog first; read-only queries run automatically.

**What it can do:**

- Run registered acquisition actions (connect, calibrate, collect spectra, run selection,
  run the Raman MDA, generate a dataset, ...).
- Read state: connection, status, wavelength, grating, image size, selection
  readiness, registered widget settings, active workflow tab, and open plots.
- Change settings without starting a run, or supply all settings consumed by
  an action in the same request (including selection `n_x`, aiming pattern,
  Cellpose options, scan/MDA channels, autofocus, and tracking controls).
- Create napari Points/Shapes layers.
- Set camera exposure and channel; snap; start/stop live.
- Move the stage by a relative offset (clamped by a safety limit), and read
  the current stage position.
- Open napari-micromanager sub-docks (stage controller, MDA, ...).
- Build a `useq` MDA sequence (channels, z-stack, timelapse, positions) and
  load it into the MDA widget, add the current stage position to it, then
  start it.
- List open plots with stable IDs; show, hide, dock, or float the workspace.
- Change supported plot controls: white/transparent background, Y-scale lock,
  pixel/wavenumber axis, display-only RACER cosmic-ray removal, smoothing and
  baseline subtraction, and plot-specific view/navigation settings.
  Unsupported settings are rejected.
- Load or clear optional wavenumber calibration, optionally attaching a loaded
  model to one explicitly selected existing plot. Loading never automatically
  opts a pixel plot into wavenumbers. Clear dark noise, collect new dark spectra
  (with confirmation), or stop live Raman spectra after the current exposure.
- Inspect a calibration result's recorded spectrum by point number, and query
  the calibration log's actual progress/stage and bounded recent messages.
- Start, inspect, or cancel the spectral-axis calibration wizard. Picking
  peaks, entering known Raman shifts, and finishing/saving remain manual.

Try: "List my plots", "Make the current spectrum background white and lock
its Y scale", "Show the recorded spectrum at calibration point 3", or
"What does the calibration log say?" For wavenumbers, supply your calibration
JSON path and say which existing plot to apply it to, then request wavenumber
display. Ask "What can you do?" for the current registered capabilities.

The Assistant gets tool descriptions and explicit state snapshots; it does not
see your screen, read arbitrary source files, inherit developer conversations,
or automatically observe manual UI changes. Progress queries are not background
monitoring. Calibration, grid scans, and axial background scans report their
running state through the task strip and log after launch; a launch response
does not mean acquisition has completed. Individual
spectrum samples are returned only when explicitly requested, with a bounded
output size; ordinary result queries return compact numeric summaries.

</details>

<details>
<summary>Saved history and local profiles</summary>

**Memory and local profiles:** Chat is private/in-memory by default and is lost
when the widget closes. Check **Save history** above the console to opt in to
local persistence; this saves the current conversation into a new profile.
The choice is remembered, and the active saved profile resumes on the next
launch. Use **History > New profile…** to name a separate conversation, and
the profile selector to switch between them. Switching replaces the AI's
conversation context, transcript, and command recall; it does not replay
commands or restore hardware/plot settings. Profile changes and clearing are
disabled while a request is running.

Uncheck **Save history** (or select **Private session**) to start a fresh,
unsaved conversation. Existing saved profiles are kept. Enabling saving from
a private session always creates a new profile, so it cannot silently merge
one customer's chat into another's. **History > Clear current history…**
clears the active conversation, draft, and command recall, and deletes its
saved chat after confirmation; other profiles are untouched. If saving is
still enabled, future messages will be saved again.

To remove a profile entirely, select it in the profile dropdown, choose
**History > Delete profile…**, and confirm its name. This removes the profile
from the list and deletes its local saved chat, then clears the current chat,
draft, and command recall and starts a fresh **Private session**. Other profiles
are untouched. Deletion cannot be undone in the widget and cannot remove copies
held by the API provider, backups, or other already-open Assistant sessions.
The delete option is disabled during a request and when no saved profile is selected.
If deletion is interrupted, a small non-chat deletion marker prevents that profile
from loading or saving again, including after a restart. The Assistant shows a
warning and lets you retry **Delete profile** to finish removing it.

History files contain chat text and tool inputs/results in **unencrypted local
JSON**, not secure customer accounts. On Windows they are under
`%LOCALAPPDATA%/napari-raman-widget/assistant`; macOS uses
`~/Library/Application Support/napari-raman-widget/assistant`, and Linux uses
`$XDG_DATA_HOME/napari-raman-widget/assistant` (or
`~/.local/share/napari-raman-widget/assistant`). Anyone with access to that OS
account may read them. No API key is deliberately added to history, but text
you type and tool output may contain sensitive information. Do not enter
secrets or sensitive customer data. Resumed context is sent to the configured
Anthropic API with your next request. Clearing local history cannot delete
provider-side records or backups.

Saved history retains at most 30 completed request/tool-loop groups
and 500 visible events, with an additional file-size limit. It is not unlimited
memory. Chats are saved after replies finish; an interrupted request may not
be saved. Invalid or unwritable files produce a visible warning rather than
silently overwriting saved history.

</details>

<details>
<summary>Developer integration</summary>

Action schemas and capability descriptions come from `ACTIONS` in
`chat_panel.py`, including the shared `assistant_*_tools.py` registries.
Add an adapter and registry entry when adding a capability. Coverage tests
include main-widget fields, shared calibration controls, and plot controls;
new UI features do not become Assistant tools automatically.
Pass `ChatPanel(self, confirm=False)` to run recognized commands without the
confirmation dialog (not recommended on live hardware).

</details>

## Structure

<details>
<summary>Source files and responsibilities</summary>

- `run_napari.py` - entry point; just launches napari with the widget.
- `launch_napari.bat` - Windows one-click launcher (activates env + runs script).
- `napari_raman_widget/hardware_widget.py` - standalone real-hardware widget;
  it contains no simulator imports or demo-mode branches.
- `napari_raman_widget/core_guard.py` - runtime retry protection installed on
  the shared `CMMCorePlus` instance before napari-micromanager is loaded.
- `napari_raman_widget/slack_notifications.py` - optional webhook alerts for
  real exceptions raised during background Raman MDA runs.
- `napari_raman_widget/hardware_defaults.py` - discovers and parses the
  machine-local hardware defaults file.
- `napari_raman_widget/demo_widget.py` - standalone simulated widget used by
  `launch_demo_napari.py`; it does not inherit from `HardwareWidget`.
- `napari_raman_widget/widget.py` - compatibility import for existing code that
  still imports `HardwareWidget` from the old module path.
- `napari_raman_widget/demo/` - simulated world, camera/DAQ backend, collector,
  Cellpose preprocessing, and coordinate transformer.
- `napari_raman_widget/field_help.py` - central hover-tooltip text and the
  `apply_tooltips()` helper.
- `napari_raman_widget/chat_panel.py` - built-in LLM-backed assistant that
  drives the panel's actions.
- `napari_raman_widget/plot_windows.py` - dock-ready Matplotlib plot panels.
- `napari_raman_widget/cosmic_ray_removal.py` - validated RACER calibration
  loading and display-only correction adapters.
- `napari_raman_widget/racer_ui.py` - session calibration controls shared by
  the hardware and demonstration widgets.
- `napari_raman_widget/plot_workspace.py` - shared, tabbed Napari plot dock.
- `napari_raman_widget/log_window.py` - streaming stdout log window.
- `napari_raman_widget/scan_preview.py` - immutable scan plans and confirmation
  previews without hardware operations.
- `napari_raman_widget/scan_workflows.py` - shared previewed grid/axial workflows
  and background calibration launch.
- `napari_raman_widget/acquisition_jobs.py` - serialized calibration/scan jobs,
  progress, and cooperative stop handling.
- `napari_raman_widget/ui_helpers.py` - small Qt helpers.
- `napari_raman_widget/resources/napari-raman-widget-manual.pdf` - the user
  manual opened by the Help link (LaTeX source kept alongside).

Hardware control, calibration, point selection, acquisition workflows, and
spectral processing are included under `napari_raman_widget`. The optional
assistant reports that it is unavailable when the Anthropic SDK or API key is
missing.

</details>

## License

`napari-raman-widget` is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
