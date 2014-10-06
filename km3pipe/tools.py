# coding=utf-8
# Filename: tools.py
# pylint: disable=locally-disabled
"""
Some frequently used logic.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'


def unpack_nfirst(seq, nfirst):
    """Unpack the nfrist items from the list and return the rest.

    >>> a, b, c, rest = unpack_nfirst((1, 2, 3, 4, 5), 3)
    >>> a, b, c
    (1, 2, 3)
    >>> rest
    (4, 5)

    """
    iterator = iter(seq)
    for item in xrange(nfirst):
        yield next(iterator, None)
    yield tuple(iterator)


def split(string, callback=None):
    """Split the string and execute the callback function on each part.

    >>> string = "1 2 3 4"
    >>> parts = split(string, int)
    >>> parts
    [1, 2, 3, 4]

    """
    if callback:
        return [callback(i) for i in string.split()]
    else:
        return string.split()

