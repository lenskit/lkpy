@echo off
GOTO start
#MISE description="Build distribution"
#USAGE flag "-s --sdist" help="build source distribution only"
:start
bash %MISE_TASK_DIR%/dist.sh %*
exit %ERRORLEVEL%
