# coding=utf-8
# cython: profile=True
# Filename: dev.py
# cython: embedsignature=True
# pylint: disable=C0103
"""
Development stuff.

"""
from __future__ import division, absolute_import, print_function

import collections
import re
import os
import warnings

from .logger import logging

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Konstantin Lepa <konstantin.lepa@gmail.com> for termcolor"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


def unpack_nfirst(seq, nfirst):
    """Unpack the nfrist items from the list and return the rest.

    >>> a, b, c, rest = unpack_nfirst((1, 2, 3, 4, 5), 3)
    >>> a, b, c
    (1, 2, 3)
    >>> rest
    (4, 5)

    """
    iterator = iter(seq)
    for _ in range(nfirst):
        yield next(iterator, None)
    yield tuple(iterator)


def split(string, callback=None, sep=' '):
    """Split the string and execute the callback function on each part.

    >>> string = "1 2 3 4"
    >>> parts = split(string, int)
    >>> parts
    [1, 2, 3, 4]

    """
    if callback is not None:
        return [callback(i) for i in string.split(sep)]
    else:
        return string.split(sep)


def namedtuple_with_defaults(typename, field_names, default_values=[]):
    """Create a namedtuple with default values

    >>> Node = namedtuple_with_defaults('Node', 'val left right')
    >>> Node()
    Node(val=None, left=None, right=None)
    >>> Node = namedtuple_with_defaults('Node', 'val left right', [1, 2, 3])
    >>> Node()
    Node(val=1, left=2, right=3)
    >>> Node = namedtuple_with_defaults('Node', 'val left right', {'right':7})
    >>> Node()
    Node(val=None, left=None, right=7)
    >>> Node(4)
    Node(val=4, left=None, right=7)
    """
    the_tuple = collections.namedtuple(typename, field_names)
    the_tuple.__new__.__defaults__ = (None,) * len(the_tuple._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = the_tuple(**default_values)
    else:
        prototype = the_tuple(*default_values)
    the_tuple.__new__.__defaults__ = tuple(prototype)
    return the_tuple


def remain_file_pointer(function):
    """Remain the file pointer position after calling the decorated function

    This decorator assumes that the last argument is the file handler.

    """
    def wrapper(*args, **kwargs):
        """Wrap the function and remain its parameters and return values"""
        file_obj = args[-1]
        old_position = file_obj.tell()
        return_value = function(*args, **kwargs)
        file_obj.seek(old_position, 0)
        return return_value
    return wrapper


try:
    dict.iteritems
except AttributeError:
    # for Python 3

    def itervalues(d):
        return iter(d.values())

    def iteritems(d):
        return iter(d.items())
else:
    # for Python 2
    def itervalues(d):
        return d.itervalues()

    def iteritems(d):
        return d.iteritems()


def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def camelise(text, capital_first=True):
    """Convert lower_underscore to CamelCase."""
    def camelcase():
        if not capital_first:
            yield str.lower
        while True:
            yield str.capitalize

    c = camelcase()
    return "".join(next(c)(x) if x else '_' for x in text.split("_"))


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    >>> @deprecated('some warning')
    ... def some_function(): pass
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.
    # and stolen again from sklearn.utils

    def __init__(self, extra=''):
        """
        Parameters
        ----------
        extra: string
          to be added to the deprecation messages
        """
        self.extra = extra

    def __call__(self, obj):
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


ATTRIBUTES = dict(
        list(zip([
            'bold',
            'dark',
            '',
            'underline',
            'blink',
            '',
            'reverse',
            'concealed'
            ],
            list(range(1, 9))
            ))
        )
del ATTRIBUTES['']

ATTRIBUTES_RE = '\033\[(?:%s)m' % '|'  \
                .join(['%d' % v for v in ATTRIBUTES.values()])

HIGHLIGHTS = dict(
        list(zip([
            'on_grey',
            'on_red',
            'on_green',
            'on_yellow',
            'on_blue',
            'on_magenta',
            'on_cyan',
            'on_white'
            ],
            list(range(40, 48))
            ))
        )

HIGHLIGHTS_RE = '\033\[(?:%s)m' % '|'  \
                .join(['%d' % v for v in HIGHLIGHTS.values()])

COLORS = dict(
        list(zip([
            'grey',
            'red',
            'green',
            'yellow',
            'blue',
            'magenta',
            'cyan',
            'white',
            ],
            list(range(30, 38))
            ))
        )

COLORS_RE = '\033\[(?:%s)m' % '|'.join(['%d' % v for v in COLORS.values()])

RESET = '\033[0m'
RESET_RE = '\033\[0m'


def colored(text, color=None, on_color=None, attrs=None):
    """Colorize text, while stripping nested ANSI color sequences.

    Author:  Konstantin Lepa <konstantin.lepa@gmail.com> / termcolor

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.
    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.
    Available attributes:
        bold, dark, underline, blink, reverse, concealed.
    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv('ANSI_COLORS_DISABLED') is None:
        fmt_str = '\033[%dm%s'
        if color is not None:
            text = re.sub(COLORS_RE + '(.*?)' + RESET_RE, r'\1', text)
            text = fmt_str % (COLORS[color], text)
        if on_color is not None:
            text = re.sub(HIGHLIGHTS_RE + '(.*?)' + RESET_RE, r'\1', text)
            text = fmt_str % (HIGHLIGHTS[on_color], text)
        if attrs is not None:
            text = re.sub(ATTRIBUTES_RE + '(.*?)' + RESET_RE, r'\1', text)
            for attr in attrs:
                text = fmt_str % (ATTRIBUTES[attr], text)
        return text + RESET
    else:
        return text


def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Print colorize text.

    Author:  Konstantin Lepa <konstantin.lepa@gmail.com> / termcolor

    It accepts arguments of print function.
    """
    print((colored(text, color, on_color, attrs)), **kwargs)
