from invoke import task


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


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()
