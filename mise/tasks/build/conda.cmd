@echo off
goto :start
#MISE description="Build conda packages"
#MISE depends=["build:dist -sd"]
:start
bash %MISE_TASK_DIR%/conda.sh %*
exit %ERRORLEVEL%
