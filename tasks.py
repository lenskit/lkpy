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
def build(c, cover=False):
    try:
        if cover:
            print('enabling coverage & profiling in Cython build')
            os.environ['COVERAGE'] = '1'

        print('running Python build')

        c.run('{} setup.py build'.format(sys.executable))
    finally:
        if 'COVERAGE' in os.environ:
            del os.environ['COVERAGE']

    ldir = Path('build/lib.%s-%d.%d' % (du.get_platform(), *sys.version_info[:2]))
    files = set()
    for ext in importlib.machinery.EXTENSION_SUFFIXES:
        files |= set(ldir.glob('lenskit/*/*' + ext))
    files |= set(ldir.glob('lenskit/*/*.pdb'))
    for pyd in files:
        path = pyd.relative_to(ldir)
        if not path.exists() or pyd.stat().st_mtime > path.stat().st_mtime:
            print('copying', pyd, '->', path)
            shutil.copy2(str(pyd), str(path))
        else:
            print(path, 'is up to date')


@task
def test(c, cover=False, verbose=True, slow=True, match=None, mark=None, debug=False):
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
    if debug:
        args.append('--log-level=DEBUG')
        args.append('--log-cli-level=DEBUG')
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
    print('remving .eggs')
    shutil.rmtree('.eggs', ignore_errors=True)
    print('remving lenskit.egg-info')
    shutil.rmtree('lenskit.egg-info', ignore_errors=True)
    ldir = Path('.')
    files = set()
    for ext in importlib.machinery.EXTENSION_SUFFIXES:
        files |= set(ldir.glob('lenskit/*/*' + ext))
    files |= set(ldir.glob('lenskit/*/*.pdb'))
    files |= set(ldir.glob('lenskit/*/*.c'))
    for f in files:
        print('removing', f)
        f.unlink()


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()
