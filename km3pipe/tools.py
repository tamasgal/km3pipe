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
    it = iter(seq)
    for x in xrange(nfirst):
        yield next(it, None)
    yield tuple(it)