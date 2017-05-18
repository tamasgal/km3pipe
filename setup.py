#!/usr/bin/env python
# coding=utf-8
# Filename: setup.py
"""
KM3Pipe setup script.

"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

from itertools import chain
import sys
import os

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

# This hack is "stolen" from numpy and allows to detect the setup procedure
# to avoid loading modules which are not compiled yet. Ugly but robust.
builtins.__KM3PIPE_SETUP__ = True


try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# Needed for line_profiler - disable for production code
CYTHON_TRACE = int(os.getenv('KM3PIPE_DEBUG', False))
directives = {
    'linetrace': CYTHON_TRACE,
    'profile': CYTHON_TRACE,
    'binding': CYTHON_TRACE,
}

from km3pipe import version  # noqa


tools = Extension('km3pipe.tools', sources=['km3pipe/tools.pyx'],
                  extra_compile_args=['-O3', '-march=native', '-w'],
                  define_macros=[('CYTHON_TRACE', str(CYTHON_TRACE))])

core = Extension('km3pipe.core', sources=['km3pipe/core.pyx'],
                 extra_compile_args=['-O3', '-march=native', '-w'],
                 define_macros=[('CYTHON_TRACE', str(CYTHON_TRACE))])

dataclasses = Extension('km3pipe.dataclasses',
                        sources=['km3pipe/dataclasses.pyx'],
                        extra_compile_args=['-O3', '-march=native', '-w'],
                        define_macros=[('CYTHON_TRACE', str(CYTHON_TRACE))])

if CYTHON_TRACE and cythonize:
    print("Building KM3Pipe with line tracing and profiling extensions.")
    ext_modules = cythonize([tools, core, dataclasses],
                            compiler_directives=directives)
else:
    ext_modules = [tools, core, dataclasses]


require_groups = {
          'docs': ['numpydoc', 'pillow',
                   'scikit-learn', 'sphinx-gallery',
                   'sphinx==1.5.6', 'sphinxcontrib-napoleon', ],
          'base': ['cython', 'docopt', 'numpy>=1.12', 'pandas', 'pytz',
                   'six', ],
          'analysis': ['matplotlib>=2.0.0', 'sklearn', 'statsmodels>=0.8',
                       'scipy>=0.19', 'seaborn', 'ipython', 'patsy', ],
          'daq': ['controlhost', ],
          'io': ['tables==3.4.0', 'h5py', ],
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
      ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext},
      include_package_data=True,
      platforms='any',
      setup_requires=['setuptools>=18.0', 'cython', 'numpy'],
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
              'totmonitor=km3pipe.utils.totmonitor:main',
          ],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
      ],
      )

__author__ = 'Tamas Gal and Moritz Lotze'
