# Filename: time.py
# pylint: disable=C0103
"""
Manipulating time and so...

"""
from __future__ import absolute_import, print_function, division

from datetime import datetime
import numpy as np
import time
from timeit import default_timer as timer

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def total_seconds(td):
    """Convert the timedelta to seconds."""
    return td.total_seconds()


class Timer(object):
    """A very simple, accurate and easy to use timer context"""

    def __init__(self, message='It', precision=3, callback=print):
        self.message = message
        self.precision = precision
        self.callback = callback

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.__start = timer()
        self.__start_cpu = time.clock()

    def stop(self):
        self.__finish = timer()
        self.__finish_cpu = time.clock()
        if self.callback is not None:
            self.log()
        return self.seconds

    @property
    def seconds(self):
        return self.__finish - self.__start

    @property
    def cpu_seconds(self):
        return self.__finish_cpu - self.__start_cpu

    def log(self):
        self.callback(
            "{0} took {1:.{3}f}s (CPU {2:.{3}f}s).".format(
                self.message, self.seconds, self.cpu_seconds, self.precision
            )
        )


class Cuckoo(object):
    "A timed callback caller, which only executes once in a given interval."

    def __init__(self, interval=0, callback=print):
        "Setup with interval in seconds and a callback function"
        self.interval = interval
        self.callback = callback
        self.timestamp = None

    def msg(self, *args, **kwargs):
        "Only execute callback when interval is reached."
        if self.timestamp is None or self._interval_reached():
            self.callback(*args, **kwargs)
            self.reset()

    def reset(self):
        "Reset the timestamp"
        self.timestamp = datetime.now()

    def _interval_reached(self):
        "Check if defined interval is reached"
        return total_seconds(datetime.now() - self.timestamp) > self.interval

    def __call__(self, *args, **kwargs):
        "Run the msg function when called directly."
        self.msg(*args, **kwargs)


def tai_timestamp():
    """Return current TAI timestamp."""
    timestamp = time.time()
    date = datetime.utcfromtimestamp(timestamp)
    if date.year < 1972:
        return timestamp
    offset = 10 + timestamp
    leap_seconds = [
        (1972, 1, 1),
        (1972, 7, 1),
        (1973, 1, 1),
        (1974, 1, 1),
        (1975, 1, 1),
        (1976, 1, 1),
        (1977, 1, 1),
        (1978, 1, 1),
        (1979, 1, 1),
        (1980, 1, 1),
        (1981, 7, 1),
        (1982, 7, 1),
        (1983, 7, 1),
        (1985, 7, 1),
        (1988, 1, 1),
        (1990, 1, 1),
        (1991, 1, 1),
        (1992, 7, 1),
        (1993, 7, 1),
        (1994, 7, 1),
        (1996, 1, 1),
        (1997, 7, 1),
        (1999, 1, 1),
        (2006, 1, 1),
        (2009, 1, 1),
        (2012, 7, 1),
        (2015, 7, 1),
        (2017, 1, 1),
    ]
    for idx, leap_date in enumerate(leap_seconds):
        if leap_date >= (date.year, date.month, date.day):
            return idx - 1 + offset
    return len(leap_seconds) - 1 + offset


def np_to_datetime(intime):
    """Convert numpy/pandas datetime64 to list[datetime]."""
    nptime = np.atleast_1d(intime)
    np_corr = (nptime - np.datetime64('1970-01-01T00:00:00')) / \
        np.timedelta64(1, 's')
    return [datetime.utcfromtimestamp(t) for t in np_corr]
