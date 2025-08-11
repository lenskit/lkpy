@echo off
echo executing task %1
echo %PATH%
if %~x1 == ".py" (
    python %*
    exit %%ERRORLEVEL%%
)

echo No runner found for ^~x1
exit 1
