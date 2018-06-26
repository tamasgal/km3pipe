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

import subprocess
import os
from os.path import dirname, join, exists

KP_PATH = join(dirname(__file__), '..')


def get_git_revision_hash(short=False):
    """Try to retrieve the hash of the last git commit"""
    infix = '--short' if short else ''
    try:
        return subprocess.check_output(
            ['git', '-C', KP_PATH, 'rev-parse', infix, 'HEAD'],
            stderr=subprocess.PIPE,
        ).strip().decode()
    except subprocess.CalledProcessError:
        pass
    fpath = join(
        KP_PATH, 'km3pipe/.git_revision_{}hash'
        .format('short_' if short else '')
    )
    if exists(fpath):
        with open(fpath, 'r') as fobj:
            return fobj.read().strip()
    else:
        return 'no-git-revision-hash'


VERSION_INFO = (8, 1, 4, 'final', 0)

__author__ = 'tamasgal'


def _get_version(version_info):
    """Return a PEP 386-compliant version number."""
    assert len(version_info) == 5
    assert version_info[3] in ('alpha', 'beta', 'rc', 'final')

    parts = 2 if version_info[2] == 0 else 3
    main = '.'.join(map(str, version_info[:parts]))

    sub = ''
    if version_info[3] == 'alpha' and version_info[4] == 0:
        sub = '.dev'
    elif version_info[3] != 'final':
        mapping = {'alpha': 'a', 'beta': 'b', 'rc': 'c'}
        sub = mapping[version_info[3]] + str(version_info[4])

    return str(main + sub)


def _get_latest_version():
    from urllib import urlopen
    import json

    response = urlopen('https://pypi.python.org/pypi/km3pipe/json')
    content = response.read()
    latest_version = json.loads(content.decode())['info']['version']
    return str(latest_version)


def check_for_update():
    try:
        latest_version = _get_latest_version()
    except IOError:
        pass
    else:
        version = _get_version(VERSION_INFO)
        if latest_version != version:
            print(
                "There is an update for km3pipe available.\n" +
                "    Installed: {0}\n"
                "    Latest: {1}\n".format(version, latest_version) +
                "Please run `pip install --upgrade km3pipe`."
            )


version = _get_version(VERSION_INFO)
