import sys
import os
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

if sys.platform == 'win32':
    # MSVC - we do not yet support mingw or clang
    openmp_cflags = ['/openmp']
    openmp_lflags = []
else:
    # assume we are using GCC or compatible
    openmp_cflags = ['-fopenmp']
    openmp_lflags = ['-fopenmp']

debug_flags = []
cython_opts = {}
if 'COVERAGE' in os.environ:
    debug_flags += ['-DCYTHON_TRACE=1', '-DCYTHON_TRACE_NOGIL=1']
    cython_opts['linetrace'] = True

extensions = [
    Extension('*', ['lenskit/**/*.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=[] + debug_flags + openmp_cflags,
              extra_link_args=[] + openmp_lflags)
]

setup(
    name="lenskit",
    version="0.0.1",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    url="https://lenskit.github.io/lkpy",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'warn.undeclared': True,
        'warn.unused': True,
        'warn.maybe_uninitialized': True,
        **cython_opts
    }),
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
        'pytest-arraydiff'
    ]
)
