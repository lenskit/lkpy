import sys
import os
from setuptools import setup, find_packages
from distutils.extension import Extension
import warnings

# get the numpy path
try:
    import numpy
    numpy_path = [numpy.get_include()]
except ImportError:
    warnings.warn('NumPy not available, compilation may fail')
    numpy_path = []

try:
    from Cython.Build import cythonize as _cythonize
except ImportError:
    warnings.warn('cython not available, cannot build until reload')
    def _cythonize(mods, *args, **kwargs):
        return mods

# configure OpenMP
use_openmp = os.environ.get('USE_OPENMP', 'yes').lower() == 'yes'
if sys.platform == 'darwin' and 'USE_OPENMP' not in os.environ:
    use_openmp = False

if use_openmp:
    if sys.platform == 'win32':
        # MSVC - we do not yet support mingw or clang
        openmp_cflags = ['/openmp']
        openmp_lflags = []
    else:
        # assume we are using GCC or compatible
        openmp_cflags = ['-fopenmp']
        openmp_lflags = ['-fopenmp']
else:
    openmp_cflags = []
    openmp_lflags = []

# Configure build flags
debug_cflags = []
cython_opts = {}
if 'COVERAGE' in os.environ:
    debug_cflags += ['-DCYTHON_TRACE=1', '-DCYTHON_TRACE_NOGIL=1']
    cython_opts['linetrace'] = True


def extmod(name, openmp=False, cflags=[], ldflags=[]):
    "Create an extension object for one of our extension modules."
    comp_args = cflags + debug_cflags
    link_args = list(ldflags)
    if openmp:
        comp_args += openmp_cflags
        link_args += openmp_lflags
    parts = name.split('.')
    parts[-1] += '.pyx'
    path = os.path.join(*parts)
    return Extension(name, [path], include_dirs=numpy_path,
                     extra_compile_args=comp_args,
                     extra_link_args=link_args)


def cythonize(mods):
    "Pre-Cythonize modules if feasible"

    # we have Cython! pre-cythonize so we can set options
    return _cythonize(mods, compiler_directives={
        'warn.undeclared': True,
        'warn.unused': True,
        'warn.maybe_uninitialized': True,
        **cython_opts,
    }, compile_time_env={'OPENMP': use_openmp})


with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
    name="lenskit",
    version="0.1.0",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    long_description=readme,
    url="https://lkpy.lenskit.org",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

    setup_requires=[
        'pytest-runner',
        'cython',
        'numpy',
        'scipy'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'tables >= 3.2.0'
    ],
    tests_require=[
        'pytest >= 3.5.1',
        'pytest-arraydiff'
    ],

    packages=find_packages(),
    ext_modules=cythonize([
        extmod('lenskit.algorithms._item_knn', openmp=True),
        extmod('lenskit.algorithms._funksvd')
    ])
)
