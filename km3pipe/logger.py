# coding=utf-8
# Filename: logger.py
# pylint: disable=locally-disabled,C0103
"""
The logging facility.

"""
from __future__ import division, absolute_import, print_function

import logging
import logging.config

__author__ = 'tamasgal'

try:
    logging.config.fileConfig('logging.conf')
except Exception:
    logging.basicConfig()

logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" %
                     logging.getLevelName(logging.INFO))
logging.addLevelName(logging.DEBUG, "\033[1;34m%s\033[1;0m" %
                     logging.getLevelName(logging.DEBUG))
logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" %
                     logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" %
                     logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.CRITICAL, "\033[1;101m%s\033[1;0m" %
                     logging.getLevelName(logging.CRITICAL))

ch = logging.StreamHandler()

# pylint: disable=C0103
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)
