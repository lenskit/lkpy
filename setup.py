import os
from pathlib import Path
from setuptools import setup, find_packages
from distutils.cmd import Command
from distutils import ccompiler, sysconfig


class BuildHelperCommand(Command):
    description = 'compile helper modules'
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        cc = ccompiler.new_compiler()
        sysconfig.customize_compiler(cc)

        dir = Path('lenskit')

        mkl_src = dir / 'mkl_ops.c'
        mkl_obj = cc.object_filenames([os.fspath(mkl_src)])
        mkl_so = cc.shared_object_filename('mkl_ops')
        mkl_so = dir / mkl_so

        if mkl_so.exists():
            src_mt = mkl_src.stat().st_mtime
            so_mt = mkl_so.stat().st_mtime
            if so_mt > src_mt:
                return mkl_so

        print('compiling MKL support library')
        i_dirs = []
        l_dirs = []
        conda = Path(os.environ['CONDA_PREFIX'])
        if os.name == 'nt':
            lib = conda / 'Library'
            i_dirs.append(os.fspath(lib / 'include'))
            l_dirs.append(os.fspath(lib / 'lib'))
        else:
            i_dirs.append(os.fspath(conda / 'include'))
            l_dirs.append(os.fspath(conda / 'lib'))

        cc.compile([os.fspath(mkl_src)], include_dirs=i_dirs, debug=True)
        cc.link_shared_object(mkl_obj, os.fspath(mkl_so), libraries=['mkl_rt'],
                              library_dirs=l_dirs)


with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
    name="lenskit",
    version="0.9.0",
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

    python_requires='>= 3.6',
    setup_requires=[
        'pytest-runner'
    ],
    install_requires=[
        'pandas >= 0.24',
        'numpy',
        'scipy',
        'numba >= 0.43, < 0.49',
        'pyarrow',
        'cffi',
        'joblib'
    ],
    tests_require=[
        'pytest >= 3.9',
        'pytest-doctestplus'
    ],
    extras_require={
        'docs': [
            'sphinx >= 1.8',
            'sphinx_rtd_theme',
            'nbsphinx',
            'recommonmark',
            'ipython'
        ],
        'hpf': [
            'hpfrec'
        ],
        'implicit': [
            'implicit'
        ]
    },
    packages=find_packages(),
    package_data={
        'lenskit': ['*.dll', '*.so', '*.dylib', '*.h']
    },

    cmdclass={
        'build_helper': BuildHelperCommand
    }
)
