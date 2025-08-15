@echo off
mise run build:dist -- -sd
if %ERRORLEVEL% NEQ 0 exit %ERRORLEVEL%
bash %MISE_TASK_DIR%/conda.sh %*
exit %ERRORLEVEL%
