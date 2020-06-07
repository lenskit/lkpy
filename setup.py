import os
import sys
from pathlib import Path
from setuptools import setup
from distutils.cmd import Command
from distutils import ccompiler, sysconfig
from distutils.command.build import build
from textwrap import dedent
from setuptools._vendor.packaging.requirements import Requirement
import json


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

        m_dir = Path('lenskit') / 'matrix'

        mkl_src = m_dir / 'mkl_ops.c'
        mkl_obj = cc.object_filenames([os.fspath(mkl_src)])
        mkl_so = cc.shared_object_filename('mkl_ops')
        mkl_so = m_dir / mkl_so

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


class DepInfo(Command):
    description = 'get dependency information'
    user_options = [
        ('extras=', 'E', 'extras to include in dependency list'),
        ('all-extras', 'A', 'include all extras'),
        ('ignore-extras=', 'I', 'ignore an extra from all-extras'),
        ('conda-env=', 'c', 'write Conda environment file'),
        ('conda-requires=', None, 'extra Conda requirements (raw yaml)'),
        ('conda-forge', None, 'use conda-forge channel'),
        ('python-version=', None, 'use specified Python version')
    ]
    boolean_options = ['all-extras', 'conda-forge']
    MAIN_PACKAGE_MAP = {
        'nbsphinx': '$pip',
        'nbval': '$pip',
        'implicit': '$pip'
    }
    FORGE_PACKAGE_MAP = {}
    CONDA_DEPS = ['mkl-devel', 'tbb']

    def initialize_options(self):
        """Set default values for options."""
        self.conda_requires = None
        self.extras = None
        self.all_extras = False
        self.ignore_extras = None
        self.conda_env = None
        self.conda_requires = None
        self.conda_forge = None
        self.python_version = None

    def finalize_options(self):
        """Post-process options."""
        if self.extras is None:
            self.extras = []
        else:
            self.extras = self.extras.split(',')
        if self.ignore_extras is None:
            self.ignore_extras = []
        else:
            self.ignore_extras = self.ignore_extras.split(',')
        if self.all_extras:
            self.extras = [e for e in self.distribution.extras_require.keys()
                           if e not in self.ignore_extras]
        if self.conda_requires is None:
            self.conda_requires = ''
        self.PACKAGE_MAP = self.FORGE_PACKAGE_MAP if self.conda_forge else self.MAIN_PACKAGE_MAP

    def run(self):
        if self.conda_env:
            self._write_conda(self.conda_env)
        else:
            for req, src in self._get_reqs():
                if src:
                    msg = f'{req}  # {src}'
                else:
                    msg = req
                print(msg)

    def _write_conda(self, file):
        if self.python_version:
            pyver = f'={self.python_version}'
        else:
            pyver = self.distribution.python_requires
        pip_deps = []
        dep_spec = {
            'name': 'lkpy-dev',
        }
        if self.conda_forge:
            dep_spec['channels'] = ['conda-forge']
        else:
            dep_spec['channels'] = ['lenskit', 'default']

        dep_spec['dependencies'] = [
            f'python {pyver}',
            'pip'
        ] + self.CONDA_DEPS
        for req_str, src in self._get_reqs():
            req = Requirement(req_str)
            mapped = self.PACKAGE_MAP.get(req.name, req.name)
            if mapped == '$pip':
                pip_deps.append(self._spec_str(req))
            else:
                dep_spec['dependencies'].append(self._spec_str(req, mapped, src))
        if pip_deps:
            dep_spec['dependencies'].append({
                'pip': pip_deps
            })

        if file == '-':
            json.dump(dep_spec, sys.stdout, indent=2)
        else:
            with open(file, 'w') as f:
                json.dump(dep_spec, f, indent=2)

    def _spec_str(self, req, name=None, src=None):
        spec = name if name is not None else req.name
        if req.specifier:
            spec += ' ' + str(req.specifier)
        return spec

    def _get_reqs(self):
        for req in self.distribution.install_requires:
            yield req, None
        for req in self.distribution.tests_require:
            yield req, 'test'
        for ex in self.extras:
            ereqs = self.distribution.extras_require[ex]
            for req in ereqs:
                yield req, ex


def has_mkl(build):
    try:
        import mkl  # noqa: F401
        return True
    except ImportError:
        return False


build.sub_commands.append(('build_helper', has_mkl))


if __name__ == "__main__":
    setup(cmdclass={
        'build_helper': BuildHelperCommand,
        'dep_info': DepInfo
    })
