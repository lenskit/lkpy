from invoke import task
import shutil
import os.path


@task
def test(c, cover=False, verbose=True):
    "Run tests"
    import pytest
    args = ['tests']
    if cover:
        args.append('--cov=lenskit')
    if verbose:
        args.append('--verbose')
    rc = pytest.main(args)
    if rc:
        raise RuntimeError('tests failed with code ' + rc)


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
