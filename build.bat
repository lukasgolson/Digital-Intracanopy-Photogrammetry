@echo off
REM This script bundles the application into a single executable file using ExePy.


REM copy license.md and readme.md into the scripts folder
copy license.md scripts
copy readme.md scripts

ExePy-Creator.exe