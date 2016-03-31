from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from functools import wraps
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from scipy.spatial import cKDTree


def sort_positions(actual, expected):
    assert_equal(len(actual), len(expected))
    tree = cKDTree(actual)
    devs, argsort = tree.query([expected])
    return devs, actual[argsort][0]


def assert_coordinates_close(actual, expected, atol):
    _, sorted_actual = sort_positions(actual, expected)
    assert_allclose(sorted_actual, expected, atol)


def repeat_test_std(func):
    @wraps(func)
    def wrapper(test_obj, *args, **kwargs):
        global result_table
        repeats = test_obj.repeats
        actual = []
        expected = []
        for i in range(repeats):
            result = func(test_obj, *args, **kwargs)
            if result is None:
                continue
            a, e = result
            if not hasattr(a, '__iter__'):
                a = (a,)
            if not hasattr(e, '__iter__'):
                e = (e,)
            assert len(a) == len(e)
            actual.append(a)
            expected.append(e)
        actual = np.array(actual, dtype=np.float).T
        expected = np.array(expected, dtype=np.float).T
        n_tests = actual.shape[0]

        for name in ['names', 'rtol', 'atol', 'fails']:
            try:
                _var = getattr(test_obj, name)
            except AttributeError:
                _var = None
            if hasattr(_var, '__iter__'):
                _var = list(_var)
            else:
                _var = [_var] * n_tests
            if len(_var) < n_tests:
                if n_tests % len(_var) == 0:
                    if name == 'names':
                        new_var = []
                        for i in range(int(n_tests // len(_var))):
                            new_var.extend([n + '_' + str(i) for n in _var])
                        _var = new_var
                    else:
                        _var *= int(n_tests // len(_var))  # broadcast
                else:
                    raise ValueError('{} has the wrong length'.format(name))
            setattr(test_obj, name, _var)

        _exceptions = []
        _result_table = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            if test_obj.atol[i] is None and test_obj.rtol[i] is None:
                continue
            n_failed = np.sum(~np.isfinite(a))
            rms_dev = np.sqrt(np.sum((a - e)**2))
            rms_dev_rel = np.sqrt(np.sum((a / e - 1)**2))
            name = test_obj.names[i]
            if name is None:
                name = func.__name__ + ' ({})'.format(i)
            else:
                name = func.__name__ + ' ({})'.format(name)
            fails = test_obj.fails[i]
            if fails is None:
                fails = 0
            if n_failed > fails:
                mesg = '{0:.0f}% of the tests in "{1}" failed'
                _exceptions.append(mesg.format(n_failed/repeats*100, name))
            if test_obj.atol[i] is not None:
                if rms_dev > test_obj.atol[i]:
                    mesg = 'rms deviation in "{2}" is too large ({0} > {1})'
                    _exceptions.append(mesg.format(rms_dev, test_obj.atol[i],
                                                   name))
            if test_obj.rtol[i] is not None:
                mesg = 'rms rel. deviation in "{2}" is too large ({0} > {1})'
                if rms_dev_rel > test_obj.rtol[i]:
                     _exceptions.append(mesg.format(rms_dev_rel,
                                                    test_obj.rtol[i], name))
            res = pd.Series([n_failed, rms_dev, rms_dev_rel], name=name)
            _result_table.append(res)

        try:
            result_table.extend(_result_table)
        except NameError:
            result_table = _result_table

        if len(_exceptions) > 0:
            raise AttributeError('\n'.join(_exceptions))

    return wrapper

class RepeatedUnitTests(object):
    N = 10
    @classmethod
    def setUpClass(cls):
        global result_table
        result_table = []

    @classmethod
    def tearDownClass(cls):
        global result_table
        results_table = pd.DataFrame(result_table)
        results_table.columns = ['fails', 'rms_dev', 'rms_rel_dev']
        print('Tests results from {}:'.format(cls.__name__))
        print(results_table)
