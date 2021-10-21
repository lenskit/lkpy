"""
Support tasks shared across LensKit packages.
"""

import sys
from invoke import task
from . import env

__ALL__ = [
    'dev_lock',
    'conda_platform'
]


@task(iterable=['extras'])
def dev_lock(c, platform=None, extras=None, version=None, blas=None, env_file=False):
    "Create a development lockfile"
    plat = env.conda_platform()

    if platform == 'all':
        plat_opt = ''
    elif platform:
        plat_opt = f'-p {platform}'
    else:
        plat_opt = f'-p {plat}'

    cmd = f'conda-lock lock --mamba {plat_opt} -f pyproject.toml'
    if env_file:
        cmd += ' -k env'

    if version:
        cmd += f' -f lkbuild/python-{version}-spec.yml'
    if blas:
        cmd += f' -f lkbuild/{blas}-spec.yml'
    for e in extras:
        cmd += f' -e {e}'

    print('running', cmd, file=sys.stderr)
    c.run(cmd)


@task
def conda_platform(c, gh_output=False):
    plat = env.conda_platform()
    if gh_output:
        print('::set-output name=conda-platform::' + plat)
    else:
        print(plat)
