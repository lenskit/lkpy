"%PYTHON%" setup.py build_helper || goto :fail
"%PYTHON%" setup.py install || goto :fail

goto :EOF

:fail
echo "Failed with code %errorlevel%"
exit 1
