#!/usr/bin/env python3


def pandas():
    """Imports and returns ``pandas``."""
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "install the 'pandas' package with:\n\n"
            "    pip install pandas\n\n"
            "or\n\n"
            "    conda install pandas"
        )
    else:
        return pandas


def scipy():
    """Imports and returns ``scipy``."""
    try:
        import scipy
    except ImportError:
        raise ImportError(
            "install the 'scipy' package with:\n\n"
            "    pip install scipy\n\n"
            "or\n\n"
            "    conda install scipy"
        )
    else:
        return scipy


def pypandoc():
    """Imports and returns ``pypandoc``."""
    try:
        import pypandoc
    except ImportError:
        raise ImportError(
            "install the 'pypandoc' package with:\n\n"
            "    pip install pypandoc\n\n"
            "or\n\n"
            "    conda install pypandoc"
        )
    else:
        return pypandoc


def urwid():
    """Imports and returns ``urwid``."""
    try:
        import urwid
    except ImportError:
        raise ImportError(
            "install the 'urwid' package with:\n\n"
            "    pip install urwid\n\n"
            "or\n\n"
            "    conda install urwid"
        )
    else:
        return urwid
