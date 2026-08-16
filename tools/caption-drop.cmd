@echo off
rem Drag one or more videos onto this file to caption them.
rem
rem Windows only supports dropping files onto batch scripts, not onto .ps1 --
rem so this is a shim. -ExecutionPolicy Bypass applies to this invocation only
rem and does not change the machine policy; it is what lets an explicitly
rem invoked script run on a default Restricted system.
rem
rem Run with no files (double-click) to choose the output folder and preset.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0caption-drop.ps1" %*
endlocal
