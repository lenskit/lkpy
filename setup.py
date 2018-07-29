import sys
import os
from setuptools import setup, find_packages
from distutils.extension import Extension


# Set up to lazily resolve the NumPy include path
# On supported systems (python >= 3.6), this will enable setup_requires to work well
if hasattr(os, 'PathLike'):
    class LazyNPPath(os.PathLike):
        "NumPy include path, but evaluated lazily"
        def __fspath__(self):
            import numpy
            return numpy.get_include()

        def __str__(self):
            return self.__fspath__()
else:
    def LazyNPPath():
        import numpy
        return numpy.get_include()


if sys.platform == 'win32':
    # MSVC - we do not yet support mingw or clang
    openmp_cflags = ['/openmp']
    openmp_lflags = []
else:
    # assume we are using GCC or compatible
    openmp_cflags = ['-fopenmp']
    openmp_lflags = ['-fopenmp']

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
    return Extension(name, [path], include_dirs=[LazyNPPath()],
                     extra_compile_args=comp_args,
                     extra_link_args=link_args)


def maybe_cythonize(mods):
    "Pre-Cythonize modules if feasible"
    try:
        from Cython.Build import cythonize
    except ImportError:
        # use distutils' built-in Cythonization
        return mods

    # we have Cython! pre-cythonize so we can set options
    return cythonize(mods, compiler_directives={
        'warn.undeclared': True,
        'warn.unused': True,
        'warn.maybe_uninitialized': True,
        **cython_opts
    })


with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
    name="lenskit",
    version="0.1.0",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    long_description=readme,
    long_description_content_type='text/markdown',
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
        extmod('lenskit.algorithms._item_knn', openmp=True),
        extmod('lenskit.algorithms._funksvd')
    ]
)
