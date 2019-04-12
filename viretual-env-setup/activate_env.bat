@echo off

if exist ".\VirtualEnv\Scripts\activate.bat" (
 @echo on
 echo Using VirtualEnv Virtual environment
 start .\VirtualEnv\Scripts\activate.bat
) ELSE (
 @echo on
 echo Please run setup.bat first to install virtual env
)

