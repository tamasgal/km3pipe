# coding=utf-8
# Filename: config.py
# pylint: disable=locally-disabled
"""
Tools for global configuration.

"""
from __future__ import division, absolute_import, print_function

import os
from six.moves.configparser import ConfigParser, NoSectionError
import getpass
try:
    input = raw_input
except NameError:
    pass

from km3pipe.logger import logging

__author__ = 'tamasgal'

log = logging.getLogger(__name__)  # pylint: disable=C0103

CONFIG_PATH = '~/.km3net'


class Config(object):
    def __init__(self):
        """Configuration manager for KM3NeT stuff"""
        self.config = ConfigParser()
        try:
            self.config.readfp(open(os.path.expanduser(CONFIG_PATH)))
        except IOError:
            log.warn("No configuration found at '{0}'".format(CONFIG_PATH))

    @property
    def db_credentials(self):
        """Return username and password for the KM3NeT WebDB."""
        try:
            username = self.config.get('DB', 'username')
            password = self.config.get('DB', 'password')
        except NoSectionError:
            username = input("Please enter your KM3NeT DB username: ")
            password = getpass.getpass("Password: ")
        return username, password
