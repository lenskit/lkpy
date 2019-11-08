"%PYTHON%" setup.py build_helper
"%PYTHON%" setup.py install
if errorlevel 1 exit 1
