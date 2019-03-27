# Filename: core.py
# pylint: disable=locally-disabled
"""
The core of the KM3Pipe framework.

"""
from __future__ import absolute_import, print_function, division

import gzip
from thepipe import Module

from .logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid", "Johannes Schumann"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)


class Pump(Module):
    """The pump with basic file or socket handling."""

    def __init__(self, *args, **kwargs):
        self.blob_file = None
        if args:
            log.warning(
                "Non-keywords argument passed. Please use keyword "
                "arguments to supress this warning. I will assume the "
                "first argument to be the `filename`."
            )
            Module.__init__(self, filename=args[0], **kwargs)
        else:
            Module.__init__(self, **kwargs)

    def open_file(self, filename):
        """Open the file with filename"""
        try:
            if filename.endswith('.gz'):
                self.blob_file = gzip.open(filename, 'rb')
            else:
                self.blob_file = open(filename, 'rb')
        except TypeError:
            log.error("Please specify a valid filename.")
            raise SystemExit
        except IOError as error_message:
            log.error(error_message)
            raise SystemExit

    def process(self, blob):
        """Create a blob"""
        raise NotImplementedError("The pump has no process() method!")

    def rewind_file(self):
        """Put the file pointer to position 0"""
        self.blob_file.seek(0, 0)

    def close_file(self):
        """Close file."""
        if self.blob_file:
            self.blob_file.close()

    def finish(self):
        pass

    def pre_finish(self):
        """Clean up open file or socket-handlers."""
        out = self.finish()
        self.close_file()
        return out

    def close(self):
        self.finish()

    def next(self):
        """Python 2 compatibility for iterators"""
        return self.__next__()

    def __enter__(self, *args, **kwargs):
        self.configure(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.finish()
