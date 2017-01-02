#!/usr/bin/env python
# coding=utf-8
# Filename: setup.py
"""
KM3Pipe setup script.

"""
from setuptools import setup, Extension
import sys

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    # from Cython.Compiler.Options import directive_defaults
    import numpy
except ImportError:
    raise SystemExit("\nCython and Numpy are required to compile KM3Pipe.\n"
                     "You can install it easily via pip:\n\n"
                     "    > pip install cython numpy")


# This hack is "stolen" from numpy and allows to detect the setup procedure
# to avoid loading modules which are not compiled yet. Ugly but robust.
builtins.__KM3PIPE_SETUP__ = True

# Needed for line_profiler - disable for production code
directives = {
    'linetrace': True,
    'profile': True,
    'binding': True,
}
CYTHON_TRACE = '1'

from km3pipe import version  # noqa


tools = Extension('km3pipe.tools', sources=['km3pipe/tools.pyx'],
                  extra_compile_args=['-O3', '-march=native', '-w'],
                  include_dirs=[numpy.get_include()],
                  define_macros=[('CYTHON_TRACE', CYTHON_TRACE)])

core = Extension('km3pipe.core', sources=['km3pipe/core.pyx'],
                 extra_compile_args=['-O3', '-march=native', '-w'],
                 include_dirs=[numpy.get_include()],
                 define_macros=[('CYTHON_TRACE', CYTHON_TRACE)])

dataclasses = Extension('km3pipe.dataclasses',
                        sources=['km3pipe/dataclasses.pyx'],
                        extra_compile_args=['-O3', '-march=native', '-w'],
                        include_dirs=[numpy.get_include()],
                        define_macros=[('CYTHON_TRACE', CYTHON_TRACE)])

setup(name='km3pipe',
      version=version,
      url='http://github.com/tamasgal/km3pipe/',
      description='An analysis framework for KM3NeT',
      author='Tamas Gal and Moritz Lotze',
      author_email='tgal@km3net.de',
      packages=['km3pipe', 'km3pipe.testing', 'km3pipe.io', 'km3pipe.utils',
                'km3modules', 'pipeinspector'],
      ext_modules=cythonize([tools, core, dataclasses],
                            compiler_directives=directives),
      cmdclass={'build_ext': build_ext},
      include_package_data=True,
      platforms='any',
      install_requires=[
          'controlhost',
          'cython',
          'docopt',
          'matplotlib==2.0.0rc2',
          'mock',
          'numpy',
          'pandas',
          'pytz',
          'scipy>=0.18',
          'six',
          'tables',
          'tornado',
          'urwid',
          'websocket-client',
          'statsmodels',
      ],
      extra_require={
          'docs': ['sphinx >= 1.4', 'sphinx-gallery', 'numpydoc',
                   'scikit-learn', 'statsmodels', 'seaborn',
                   'pillow', 'ipython', 'sphinxcontrib-napoleon',
                   'astropy',
                   ],
          'jppy': ['jppy'],
          'testing': ['pytest', 'mock'],
      },
      entry_points={
          'console_scripts': [
              'km3pipe=km3pipe.cmd:main',
              'km3srv=km3pipe.srv:main',
              'tohdf5=km3pipe.utils.tohdf5:main',
              'hdf2root=km3pipe.utils.hdf2root:main',
              'pipeinspector=pipeinspector.app:main',
              'h5tree=km3pipe.utils.h5tree:main',
              'rtree=km3pipe.utils.rtree:main',
              'h5info=km3pipe.utils.h5info:main',
              'h5concat=km3pipe.utils.h5concat:main',
              'meantots=km3pipe.utils.meantots:main',
              'ztplot=km3pipe.utils.ztplot:main',
          ],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
      ],
      )

__author__ = 'Tamas Gal'
