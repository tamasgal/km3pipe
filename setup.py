#!/usr/bin/env python
# Filename: setup.py
"""
KM3Pipe setup script.

"""

from setuptools import setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins
# so we can detect in __init__.py that it's called from setup.py
builtins.__KM3PIPE_SETUP__ = True

from km3pipe import version    # noqa

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(
    name='km3pipe',
    version=version,
    url='http://github.com/tamasgal/km3pipe/',
    description='An analysis framework for KM3NeT',
    author='Tamas Gal and Moritz Lotze',
    author_email='tgal@km3net.de',
    packages=[
        'km3pipe', 'km3pipe.io', 'km3pipe.utils', 'km3modules', 'pipeinspector'
    ],
    include_package_data=True,
    platforms='any',
    setup_requires=[
        'numpy>=1.12',
    ],
    install_requires=requirements,
    python_requires='>=2.7',
    entry_points={
        'console_scripts': [
            'km3pipe=km3pipe.cmd:main',
            'km3srv=km3pipe.srv:main',
            'tohdf5=km3pipe.utils.tohdf5:main',
            'hdf2root=km3pipe.utils.hdf2root:main',
            'pipeinspector=pipeinspector.app:main',
            'rtree=km3pipe.utils.rtree:main',
            'h5info=km3pipe.utils.h5info:main',
            'h5tree=km3pipe.utils.h5tree:main',
            'h5header=km3pipe.utils.h5header:main',
            'ptconcat=km3pipe.utils.ptconcat:main',
            'meantots=km3pipe.utils.meantots:main',
            'pushover=km3pipe.utils.pushover:main',
            'ztplot=km3pipe.utils.ztplot:main',
            'k40calib=km3pipe.utils.k40calib:main',
            'totmonitor=km3pipe.utils.totmonitor:main',
            'calibrate=km3pipe.utils.calibrate:main',
            'rba=km3pipe.utils.rba:main',
            'i3toroot=km3pipe.utils.i3toroot:main',
            'i3root2hdf5=km3pipe.utils.i3root2hdf5:main',
            'i3shower2hdf5=km3pipe.utils.i3shower2hdf5:main',
            'streamds=km3pipe.utils.streamds:main',
            'triggermap=km3pipe.utils.triggermap:main',
            'nb2sphx=km3pipe.utils.nb2sphx:main',
            'km3h5concat=km3pipe.utils.km3h5concat:main',
            'triggersetup=km3pipe.utils.triggersetup:main',
            'ligiermirror=km3pipe.utils.ligiermirror:main',
            'runtable=km3pipe.utils.runtable:main',
            'runinfo=km3pipe.utils.runinfo:main',
            'qrunprocessor=km3pipe.utils.qrunprocessor:main',
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
