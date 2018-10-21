import sys
import os
import distutils.util as du
from invoke import task
from invoke.exceptions import Failure
from invoke.runners import Result
import shutil
from pathlib import Path
import importlib.machinery


@task
def test(c, cover=False, verbose=True, slow=True, eval=True, match=None, mark=None, debug=False,
         forked=False, fail_fast=False):
    "Run tests"
    import pytest
    args = ['tests']
    if cover:
        args.append('--cov=lenskit')
    if verbose:
        args.append('--verbose')
    if fail_fast:
        args.append('-x')
    if not slow:
        args.append('-m')
        args.append('not slow')
    elif not eval:
        args.append('-m')
        args.append('not eval')
    if match:
        args.append('-k')
        args.append(match)
    if mark:
        args.append('-m')
        args.append(mark)
    if debug:
        args.append('--log-cli-level=DEBUG')
    if forked:
        args.append('--forked')
    rc = pytest.main(args)
    if rc:
        raise Failure(Result(exited=rc), 'tests failed')


@task
def docs(c):
    "Build documentation"
    c.run('sphinx-build -M html doc build/doc')


@task
def clean(c):
    print('remving build')
    shutil.rmtree('build', ignore_errors=True)
    print('remving dist')
    shutil.rmtree('dist', ignore_errors=True)
    print('remving .eggs')
    shutil.rmtree('.eggs', ignore_errors=True)
    print('remving lenskit.egg-info')
    shutil.rmtree('lenskit.egg-info', ignore_errors=True)


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()
