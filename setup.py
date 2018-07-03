import sys
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

extensions = cythonize('lenskit/**/*.pyx')
for ext in extensions:
    ext.extra_compile_flags = ['/openmp' if sys.platform == 'win32' else '-fopenmp']
    ext.extra_link_flags = ['/openmp' if sys.platform == 'win32' else '-fopenmp']

setup(
    name="lenskit",
    version="0.0.1",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    url="https://lenskit.github.io/lkpy",
    packages=find_packages(),
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),

    install_requires=[
        'pandas',
        'numpy'
    ],
    setup_requires=[
        'pytest-runner',
        'pytest-cov',
        'sphinx',
        'Cython'
    ],
    tests_require=[
        'pytest >= 3.5.1',
        'pytest-arraydiff',
        'dask',
        'toolz',
        'cloudpickle'
    ]
)
