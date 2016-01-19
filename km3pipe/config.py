# coding=utf-8
# Filename: config.py
# pylint: disable=locally-disabled
"""
Tools for global configuration.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import ConfigParser, os
import getpass

import logging
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


CONFIG_PATH='~/.km3net'


class Config(object):
    def __init__(self):
        """Configuration manager for KM3NeT stuff"""
        self.config = ConfigParser.ConfigParser()
        try:
            self.config.readfp(open(os.path.expanduser(CONFIG_PATH)))
        except IOError:
            log.error("No configuration found at '{0}'".format(CONFIG_PATH))

    @property
    def db_credentials(self):
        """Return username and password for the KM3NeT WebDB."""
        try:
            username = self.config.get('DB', 'username')
            password = self.config.get('DB', 'password')
        except ConfigParser.NoSectionError:
            username = raw_input("Please enter your KM3NeT DB username: ")
            password = getpass.getpass("Password: ")
        return username, password

