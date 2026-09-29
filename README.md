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
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra with autofocus
- Running spatial Raman mapping (grid scan) over a shape
- Automated cell selection inside a mask
- Running a Raman MDA with fluorescence channels and Z stacks

The interface also includes:

- **Inline help** - every field has a hover tooltip explaining what it does,
  and a **Help** button at the top of the panel opens the full PDF user manual.
- **AI assistant** - a built-in chat box that maps plain-English commands to
  the panel's existing actions (see [AI assistant](#ai-assistant-chat-panel)).
- **Dockable plots** - spectra, detector images, scans, calibration views,
  datasets, and logs share the **Raman Plots** workspace, initially floating.
- **Persistent status** - the current acquisition status stays visible below
  the controls while you change tabs or scroll.

All outputs (reference `.npy` files, `grid_scan_*.zarr`, recalibrated models,
the MDA writer directory) are written relative to the current working
directory - or an output folder you set in the Loading section, which is
switched to on connect.

## Video demonstrations

Watch the complete procedure series on the
[napari-raman-widget YouTube channel](https://www.youtube.com/@napari-raman-widget).
The videos cover software loading, hardware connection, point-based spectrum
collection, laser aiming calibration, axial background scanning, spatial
mapping, stage-grid generation, automated cell selection, Raman MDA, dataset
generation, and pixel-to-stage calibration.

## Supporting your hardware

> **For most users, adapting the platform to different microscope hardware
> requires changes only in
> [`raman-control`](https://github.com/JasonYu1/raman-control).** Typically, you only need to
> update its **collector** to communicate with your Raman spectrometer or
> detector and its **DAQ implementation** to communicate with your DAQ board
> and connected devices. The napari widget and
> [`raman-mda-engine`](https://github.com/JasonYu1/raman-mda-engine) use the
> existing `raman-control` interface and should not require hardware-specific
> changes.

## Install

For normal use, install the widget directly from GitHub:

```bash
pip install git+https://github.com/JasonYu1/napari-raman-widget.git
```

This installs the declared dependencies automatically, including the
project-specific [`raman-control`](https://github.com/JasonYu1/raman-control)
and [`raman-mda-engine`](https://github.com/JasonYu1/raman-mda-engine) packages
from their public GitHub repositories.

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/JasonYu1/napari-raman-widget.git
cd napari-raman-widget
pip install -e .
```

The AI assistant is optional. To enable it, also install the Anthropic SDK
and set an API key (see [AI assistant](#ai-assistant-chat-panel)):

```bash
pip install anthropic
```

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

### Arranging plots and controls

Results open as tabs in the **Raman Plots** workspace, which starts as a
floating window. Drag tabs to reorder
them, and hover over a tab to see its complete acquisition title. The **Plots**
button at the top of the Raman controls brings the workspace back if hidden.

Click **Dock back** to dock it inside Napari, or **Float** to detach it again.
New results and hiding/reopening the workspace preserve your docking choice
for the current session. You can also drag the dock title bar
to an edge of the main window to change its dock position. **Hide** keeps all
results available; the close button on an individual tab closes that result.
Closing a live spectrum tab requests a stop after the current exposure;
hiding or floating the workspace lets acquisition continue.

Each spectrum has compact display controls and an expandable **Processing**
section for baseline subtraction and smoothing. These settings change the
display only; acquired data stays unchanged. Dataset navigation separates
time, position, and Z from the spectral processing controls.

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

The control sidebar groups loading and calibration under **Setup**, stage-grid
and cell tools under **Selection**, measurement workflows under **Acquire**,
and dataset generation under **Analysis**. Pixel-to-stage calibration is inside
**Selection → Generate stage grid**.
The optional AI chat has its own **Assistant** tab.

Dark noise starts as **None** whenever a widget opens, including when an older
defaults file contains a dark-noise path. Select or collect a dark-noise file
to use it for the current session, or click **Clear** to return to None.

### One-click launcher (Windows)

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

### Hardware defaults file

To prefill machine-specific configuration, model, wavelength, and selection
fields whenever the hardware widget opens:

1. Copy `napari_raman_defaults.example.json` to
   `napari_raman_defaults.json` in the repository root.
2. Edit the copied file with your paths and preferred values.
3. Restart napari.

The local defaults filename is ignored by Git. Relative paths inside it are
resolved relative to the defaults file. To keep the file elsewhere, set the
`NAPARI_RAMAN_DEFAULTS` environment variable to its full path before launch.

### Slack alerts for Raman MDA failures

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

## Inline help and manual

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

## AI assistant (chat panel)

The **Assistant** tab is a full-height, terminal-style console inside the
widget. Type directly at the `> ` prompt below the conversation—there is no
separate typing box or Send button. **Enter** sends, **Shift+Enter** inserts
a new line, and **Up/Down** recalls submitted requests. You can select and
copy earlier output, but cannot accidentally edit it. Pasting multiple lines
does not submit them; review the text and press Enter when ready.

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
  pixel/wavenumber axis, display-only smoothing and baseline subtraction,
  and plot-specific view/navigation settings. Unsupported settings are rejected.
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
monitoring. A synchronous calibration launched through chat can keep the chat
busy until it returns; the on-screen log still shows its progress. Individual
spectrum samples are returned only when explicitly requested, with a bounded
output size; ordinary result queries return compact numeric summaries.

**Setup:**

1. `pip install anthropic`
2. Set an Anthropic API key in the environment **before** launching napari,
   e.g. on Windows: `setx ANTHROPIC_API_KEY "sk-ant-..."` (open a new
   terminal afterwards), or per-session `$env:ANTHROPIC_API_KEY = "sk-ant-..."`.
3. Set `MODEL` at the top of `chat_panel.py` to a model your account can
   access.

The assistant is created automatically when `HardwareWidget` opens; no widget
source changes are required.

Action schemas and capability descriptions come from `ACTIONS` in
`chat_panel.py`, including the shared `assistant_*_tools.py` registries.
Add an adapter and registry entry when adding a capability. Coverage tests
include main-widget fields, shared calibration controls, and plot controls;
new UI features do not become Assistant tools automatically.
Pass `ChatPanel(self, confirm=False)` to run recognized commands without the
confirmation dialog (not recommended on live hardware).

## Structure

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
- `napari_raman_widget/plot_workspace.py` - shared, tabbed Napari plot dock.
- `napari_raman_widget/log_window.py` - streaming stdout log window.
- `napari_raman_widget/ui_helpers.py` - small Qt helpers.
- `napari_raman_widget/resources/napari-raman-widget-manual.pdf` - the user
  manual opened by the Help link (LaTeX source kept alongside).

Hardware control, calibration, point selection, acquisition workflows, and
spectral processing are included under `napari_raman_widget`. The optional
assistant reports that it is unavailable when the Anthropic SDK or API key is
missing.

## License

`napari-raman-widget` is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
