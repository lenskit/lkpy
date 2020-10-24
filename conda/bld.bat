"%PYTHON%" setup.py build_helper || goto :fail
"%PYTHON%" -m pip install --no-deps . || goto :fail

goto :EOF

:fail
echo "Failed with code %errorlevel%"
exit 1
