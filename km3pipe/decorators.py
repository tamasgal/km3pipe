from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'


def remain_file_pointer(f):
    """Remain the file pointer position after calling the decorated function

    This decorator assumes that the last argument is the file handler.

    """
    def wrapper(*args, **kwargs):
        file_obj = args[-1]
        old_position = file_obj.tell()
        f(*args, **kwargs)
        file_obj.seek(old_position, 0)
    return wrapper