from setuptools import setup

setup(
    requires = ['pandas'],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest', 'dask']
)