#!/usr/bin/env python
# Filename: setup.py
"""
KM3Pipe setup script.

"""
import os
from setuptools import setup
import sys

import builtins

# so we can detect in __init__.py that it's called from setup.py
builtins.__KM3PIPE_SETUP__ = True


def read_requirements(kind):
    """Return a list of stripped lines from a file"""
    with open(os.path.join("requirements", kind + ".txt")) as fobj:
        requirements = [l.strip() for l in fobj.readlines()]
    v = sys.version_info
    if (v.major, v.minor) < (3, 6):
        try:
            requirements.pop(requirements.index("black"))
        except ValueError:
            pass
    return requirements


try:
    with open("README.rst") as fh:
        long_description = fh.read()
except UnicodeDecodeError:
    long_description = "KM3Pipe"

setup(
    name="km3pipe",
    url="http://git.km3net.de/km3py/km3pipe",
    description="An analysis framework for KM3NeT",
    long_description=long_description,
    author="Tamas Gal and Moritz Lotze",
    author_email="tgal@km3net.de",
    packages=["km3pipe", "km3pipe.io", "km3pipe.utils", "km3modules", "pipeinspector"],
    include_package_data=True,
    platforms="any",
    setup_requires=["numpy>=1.12", "setuptools_scm"],
    install_requires=read_requirements("install"),
    extras_require={kind: read_requirements(kind) for kind in ["dev", "extras"]},
    use_scm_version=True,
    python_requires=">=3.5",
    entry_points={
        "console_scripts": [
            "km3pipe=km3pipe.cmd:main",
            "pipeinspector=pipeinspector.app:main",
            "h5extract=km3pipe.utils.h5extract:main",
            "h5info=km3pipe.utils.h5info:main",
            "h5tree=km3pipe.utils.h5tree:main",
            "h5header=km3pipe.utils.h5header:main",
            "meantots=km3pipe.utils.meantots:main",
            "ztplot=km3pipe.utils.ztplot:main",
            "k40calib=km3pipe.utils.k40calib:main",
            "triggermap=km3pipe.utils.triggermap:main",
            "nb2sphx=km3pipe.utils.nb2sphx:main",
            "ligiermirror=km3pipe.utils.ligiermirror:main",
            "qrunprocessor=km3pipe.utils.qrunprocessor:main",
            "qrunqaqc=km3pipe.utils.qrunqaqc:main",
            "daqsample=km3pipe.utils.daqsample:main",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
    ],
)

__author__ = "Tamas Gal and Moritz Lotze"
