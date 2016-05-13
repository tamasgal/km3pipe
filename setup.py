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
    import numpy
except ImportError:
    raise SystemExit("\nCython and Numpy are required to compile KM3Pipe.\n"
                     "You can install it easily via pip:\n\n"
                     "    > pip install cython numpy")


# This hack is "stolen" from numpy and allows to detect the setup procedure
# to avoid loading modules which are not compiled yet. Ugly but robust.
builtins.__KM3PIPE_SETUP__ = True

from km3pipe import version  # noqa


tools = Extension('km3pipe.tools', sources=['km3pipe/tools.pyx'],
                  extra_compile_args=['-O3', '-march=native', '-w'],
                  include_dirs=[numpy.get_include()])

core = Extension('km3pipe.core', sources=['km3pipe/core.pyx'],
                 extra_compile_args=['-O3', '-march=native', '-w'],
                 include_dirs=[numpy.get_include()])

dataclasses = Extension('km3pipe.dataclasses',
                        sources=['km3pipe/dataclasses.pyx'],
                        extra_compile_args=['-O3', '-march=native', '-w'],
                        include_dirs=[numpy.get_include()])

setup(name='km3pipe',
      version=version,
      url='http://github.com/tamasgal/km3pipe/',
      description='An analysis framework for KM3NeT',
      author='Tamas Gal',
      author_email='tgal@km3net.de',
      packages=['km3pipe', 'km3pipe.testing', 'km3pipe.pumps', 'km3pipe.utils',
                'km3modules', 'pipeinspector'],
      ext_modules=[tools, core, dataclasses],
      cmdclass={'build_ext': build_ext},
      include_package_data=True,
      platforms='any',
      install_requires=[
          'cython',
          'numpy',
          'controlhost',
          'urwid',
          'docopt',
          'tables',
          'pandas',
          'seaborn',
          'pytz',
          'six',
      ],
      entry_points={
          'console_scripts': [
              'km3pipe=km3pipe.cmd:main',
              'pipeinspector=pipeinspector.app:main',
              'h5tree=km3pipe.utils.h5tree:main',
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
