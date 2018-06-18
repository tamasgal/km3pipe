#!/usr/bin/env python
# vim:set ts=4 sts=4 sw=4 et:

from setuptools import setup
from pydemo import __version__

setup(
    name='pydemo',
    version=__version__,
    description='Astro Utils',
    url='http://git.km3net.de/moritz/pydemo',
    author='Moritz Lotze',
    author_email='mlotze@km3net.de',
    license='BSD-3',
    packages=[
        'pydemo',
    ],
    install_requires=[
        'astropy',
        'numpy',
        'pandas',
        'km3pipe',
    ]
)
