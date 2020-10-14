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
