@echo off

if not exist ".\VirtualEnv\" (
mkdir .\VirtualEnv\
@echo on
echo virtualenv-start
pip install virtualenv==16.1.0
call virtualenv VirtualEnv
call .\VirtualEnv\Scripts\activate.bat
pip install -r .\requirements.txt
echo Finished Env setup
@echo off
)
if exist ".\VirtualEnv\" (
 @echo on
 echo Using VirtualEnv Virtual environment
 call .\VirtualEnv\Scripts\activate.bat
)
