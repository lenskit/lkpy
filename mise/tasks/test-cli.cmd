@echo off
goto :start
#MISE description="Run CLI tests"
:start

usage bash tests/cli/run.sh %*
exit %ERRORLEVEL%
