@echo off
goto :main

:(){}
python3 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
exit

:main
if "%OS%"=="Windows_NT" (
	py.exe -m venv venv 2>NUL
	call venv/Scripts/activate.bat
	pip install -r requirements.txt
)