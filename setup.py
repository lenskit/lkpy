import sys
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

openmp_flag = '/openmp' if sys.platform == 'win32' else '-fopenmp'

extensions = [
    Extension('*', ['lenskit/**/*.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=[openmp_flag],
              extra_link_args=[openmp_flag])
]

setup(
    name="lenskit",
    version="0.0.1",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    url="https://lenskit.github.io/lkpy",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
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
