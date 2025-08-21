@echo off
GOTO start
#MISE description="Build distribution"
#USAGE flag "-s --sdist" help="build source dist only"
#USAGE flag "-c --clean" help="clean staged sources before building"
#USAGE flag "-d --dynamic-version" help="create dynamically-versioned sdist"
:start
bash %MISE_TASK_DIR%/dist.sh %*
exit %ERRORLEVEL%
