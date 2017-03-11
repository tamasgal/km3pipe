#!/usr/bin/env python
# coding=utf-8
# Filename: setup.py
"""
KM3Pipe setup script.

"""
from setuptools import setup, Extension
from itertools import chain
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

require_groups = {
          'docs': ['numpydoc', 'pillow',
                   'scikit-learn', 'sphinx-gallery',
                   'sphinx>=1.5.1', 'sphinxcontrib-napoleon', ],
          'base': ['cython', 'docopt', 'numpy>=1.12', 'pandas', 'pytz',
                   'six', ],
          'analysis': ['matplotlib>=2.0.0', 'sklearn', 'statsmodels>=0.8',
                       'scipy', 'seaborn', 'ipython', 'patsy', ],
          'daq': ['controlhost', ],
          'io': ['tables', 'h5py', ],
          'jpp': ['jppy>=1.3.1', ],
          'web': ['tornado', 'websocket-client', ],
          'testing': ['pytest', 'mock', ],
          'utils': ['urwid', ],
}
require_groups['most'] = list(chain.from_iterable(
    [require_groups[k] for k in ('base', 'io', 'web', 'utils')],
))
require_groups['full'] = list(chain.from_iterable(
    [require_groups[k] for k in ('base', 'io', 'web', 'utils', 'analysis',
                                 'testing', 'daq', 'docs')],
))

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
      install_requires=['cython', 'docopt', 'numpy>=1.12', 'pandas', 'pytz',
                        'six', ],
      extras_require=require_groups,
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
              'ptconcat=km3pipe.utils.ptconcat:main',
              'meantots=km3pipe.utils.meantots:main',
              'pushover=km3pipe.utils.pushover:main',
              'ztplot=km3pipe.utils.ztplot:main',
              'k40calib=km3pipe.utils.k40calib:main',
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
