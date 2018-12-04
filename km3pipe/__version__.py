#!/usr/bin/env python
# Filename: __version__.py
# pylint: disable=C0103
"""
Pep 386 compliant version info.

    (major, minor, micro, alpha/beta/rc/final, #)
    (1, 1, 2, 'alpha', 0) => "1.1.2.dev"
    (1, 2, 0, 'beta', 2) => "1.2b2"

"""
from __future__ import absolute_import, print_function, division

from setuptools_scm import get_version
version = get_version(root='..', relative_to=__file__)
