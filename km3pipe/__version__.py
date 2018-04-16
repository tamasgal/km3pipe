#!/usr/bin/env python
# Filename: __version__.py
# pylint: disable=C0103
"""
Pep 386 compliant version info.

    (major, minor, micro, alpha/beta/rc/final, #)
    (1, 1, 2, 'alpha', 0) => "1.1.2.dev"
    (1, 2, 0, 'beta', 2) => "1.2b2"

"""

VERSION_INFO = (8, 0, 0, 'alpha', 0)

import json
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

from km3pipe.logger import logging

__author__ = 'tamasgal'

log = logging.getLogger(__name__)  # pylint: disable=C0103


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
            log.warning("There is an update for km3pipe available.\n" +
                        "    Installed: {0}\n"
                        "    Latest: {1}\n".format(version, latest_version) +
                        "Please run `pip install --upgrade km3pipe`.")


version = _get_version(VERSION_INFO)
