import sys
import distutils.util as du
from invoke import task
from invoke.exceptions import Failure
from invoke.runners import Result
import shutil
from pathlib import Path


@task
def build(c):
    c.run('{} setup.py build'.format(sys.executable))
    ldir = Path('build/lib.%s-%d.%d' % (du.get_platform(), *sys.version_info[:2]))
    for pyd in ldir.glob('lenskit/*/*.pyd'):
        path = pyd.relative_to(ldir)
        if not path.exists() or pyd.stat().st_mtime > path.stat().st_mtime:
            print('copying', pyd, '->', path)
            shutil.copy2(pyd, path)


@task
def test(c, cover=False, verbose=True, slow=True, match=None, mark=None):
    "Run tests"
    import pytest
    args = ['tests']
    if cover:
        args.append('--cov=lenskit')
    if verbose:
        args.append('--verbose')
    if not slow:
        args.append('-m')
        args.append('not slow')
    if match:
        args.append('-k')
        args.append(match)
    if mark:
        args.append('-m')
        args.append(mark)
    rc = pytest.main(args)
    if rc:
        raise Failure(Result(exited=rc), 'tests failed')


@task
def docs(c):
    "Build documentation"
    c.run('sphinx-build -M html doc build/doc')


@task
def clean(c):
    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree('.eggs', ignore_errors=True)
    shutil.rmtree('lenskit.egg-info', ignore_errors=True)


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()
