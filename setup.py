import sys
import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension

# monkey-patch build_ext
__build_ext_finalize = build_ext.finalize_options
def _finalize_options(self):
    from Cython.Build import cythonize
    import numpy

    for mod in self.distribution.ext_modules:
        mod.include_dirs.append(numpy.get_include())

    self.distribution.ext_modules[:] = cythonize(
        self.distribution.ext_modules,
        compile_time_env={'OPENMP': use_openmp},
        compiler_directives={
            'warn.undeclared': True,
            'warn.unused': True,
            'warn.maybe_uninitialized': True,
            **cython_opts,
        }
    )
    __build_ext_finalize(self)
build_ext.finalize_options = _finalize_options

# configure OpenMP
use_openmp = os.environ.get('USE_OPENMP', 'yes').lower()
if use_openmp == 'no':
    use_openmp = False
elif use_openmp == 'yes':
    use_openmp = True
if sys.platform == 'darwin' and 'USE_OPENMP' not in os.environ:
    use_openmp = False


openmp_cflags = []
openmp_lflags = []
openmp_libs = []
if use_openmp:
    if sys.platform == 'win32':
        # MSVC - we do not yet support mingw or clang
        openmp_cflags = ['/openmp']
    elif use_openmp == 'intel':
        openmp_cflags = ['-fopenmp']
        openmp_libs = ['iomp5', 'pthread']
    else:
        # assume we are using GCC or compatible
        openmp_cflags = ['-fopenmp']
        openmp_lflags = ['-fopenmp']

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
    libs = []
    if openmp:
        comp_args += openmp_cflags
        link_args += openmp_lflags
        libs += openmp_libs
    parts = name.split('.')
    parts[-1] += '.pyx'
    path = os.path.join(*parts)
    return Extension(name, [path],
                     extra_compile_args=comp_args,
                     extra_link_args=link_args,
                     libraries=libs)


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
    ext_modules=[
        extmod('lenskit._cy_util'),
        extmod('lenskit.algorithms._item_knn', openmp=True),
        extmod('lenskit.algorithms._funksvd')
    ]
)
