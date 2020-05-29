import os
from pathlib import Path
from setuptools import setup
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


if __name__ == "__main__":
    setup(cmdclass={
        'build_helper': BuildHelperCommand
    })
