from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six


def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def guess_pos_columns(f):
    """ Guess the position columns from a given feature DataFrame """
    if 'z' in f:
        pos_columns = ['z', 'y', 'x']
    else:
        pos_columns = ['y', 'x']
    return pos_columns
