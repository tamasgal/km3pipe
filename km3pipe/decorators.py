# coding=utf-8
# Filename: decorators.py
# pylint: disable=locally-disabled
"""
Function decorators.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'


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

