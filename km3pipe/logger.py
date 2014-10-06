# coding=utf-8
# Filename: logger.py
# pylint: disable=locally-disabled,C0103
"""
The logging facility.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import logging

logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" %
                     logging.getLevelName(logging.INFO))
logging.addLevelName(logging.DEBUG, "\033[1;34m%s\033[1;0m" %
                     logging.getLevelName(logging.DEBUG))
logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" %
                     logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" %
                     logging.getLevelName(logging.ERROR))

# pylint: disable=C0103
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')


def get_logger(name, log_level=logging.DEBUG):
    """Create a fancy logger with the given name.

    If you use it in your own module, give it __name__ as name.

    Usage example:
    --------
    >>> from km3pipe.logger import get_logger
    >>> log = get_logger('foo_logger')
    >>> log.debug("This is a foo debug.")
    [DEBUG] foo_logger: This is a foo debug.
    >>> log.info("This is a foo info.")
    [INFO] foo_logger: This is a foo info.
    >>> log.warn("This is a foo warning.")
    [WARNING] foo_logger: This is a foo warning.
    >>> log.error("This is a foo error.")
    [ERROR] foo_logger: This is a foo error.

    """
    log = logging.getLogger(name)
    log.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    log.addHandler(console_handler)
    return log
