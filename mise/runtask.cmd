@echo off

if %~x1 == .py (
    uv run python %*
    exit %%ERRORLEVEL%%
)

echo No runner found for %~x1
exit 1
