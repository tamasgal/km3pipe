# coding=utf-8
# Filename: config.py
# pylint: disable=locally-disabled
"""
Tools for global configuration.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import ConfigParser, os

import logging
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


CONFIG_PATH='~/.km3net'


class Config(object):
    def __init__(self):
        """Configuration manager for KM3NeT stuff"""
        self.config = ConfigParser.ConfigParser()
        self.config.readfp(open(os.path.expanduser(CONFIG_PATH)))

    @property
    def db_credentials(self):
        """Return username and password for the KM3NeT WebDB."""
        username = self.config.get('DB', 'username')
        password = self.config.get('DB', 'password')
        return username, password

