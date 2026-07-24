@echo off
call C:\Users\Rayleigh\miniforge3\Scripts\activate.bat micro-control-2
cd /d "%~dp0"
python launch_demo_napari.py
pause
