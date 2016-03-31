""" Nosetests for finding features """
from __future__ import (division, unicode_literals)

import unittest
import nose
from functools import wraps
from numpy.testing import assert_allclose, assert_equal
import numpy as np
import pandas as pd
from circletracking import (ellipse_grid, ellipsoid_grid, locate_ellipse,
                            draw_ellipse, draw_ellipsoid, fit_ellipse,
                            fit_ellipsoid, find_ellipsoid, find_ellipse,
                            locate_ellipsoid, locate_ellipsoid_fast,
                            SimulatedImage, find_disks, locate_disks)
from scipy.spatial import cKDTree


def sort_positions(actual, expected):
    tree = cKDTree(actual)
    deviations, argsort = tree.query([expected])
    return deviations, actual[argsort][0]


class repeat_test_std(object):
    def __init__(self, repeats, names=None, atol=None, rtol=None, fails=None):
        self.repeats = repeats
        self.atol = atol
        self.rtol = rtol
        self.fails = fails
        self.names = names

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global result_table
            actual = []
            expected = []
            for i in range(self.repeats):
                result = func(*args, **kwargs)
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
            atol, rtol, names, fails = self.atol, self.rtol, self.names, self.fails
            if not hasattr(self.atol, '__iter__'):
                atol = [atol] * n_tests
            if not hasattr(self.rtol, '__iter__'):
                rtol = [rtol] * n_tests
            if not hasattr(self.fails, '__iter__'):
                fails = [fails] * n_tests
            if not hasattr(self.names, '__iter__'):
                names = [names] * n_tests
            else:
                names = list(names)
            _result_table = []
            for i, (a, e) in enumerate(zip(actual, expected)):
                n_failed = np.sum(~np.isfinite(a))
                rms_dev = np.sqrt(np.sum((a - e)**2))
                rms_dev_rel = np.sqrt(np.sum((a / e - 1)**2))
                if names[i] is None:
                    names[i] = ''
                names[i] = func.__name__ + ' ({})'.format(names[i])
                res = pd.Series([n_failed, rms_dev, rms_dev_rel], name=names[i])
                _result_table.append(res)

            try:
                result_table.extend(_result_table)
            except NameError:
                result_table = _result_table

            for i, res in enumerate(_result_table):
                if fails[i] is None:
                    fails[i] = 0
                if n_failed > fails[i]:
                    raise AssertionError('{0:.0f}% of the tests in "{1}" failed'.format(n_failed/self.repeats*100, names[i]))
                if atol[i] is not None:
                    if rms_dev > atol[i]:
                        raise AssertionError('rms deviation in "{2}" is too large ({0} > {1})'.format(rms_dev, atol[i], names[i]))
                if rtol[i] is not None:
                    if rms_dev_rel > rtol[i]:
                        raise AssertionError('rms relative deviation in "{2}" is too large ({0} > {1})'.format(rms_dev_rel, rtol[i], names[i]))
        return wrapper

class RepeatedUnitTests(unittest.TestCase):
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


class TestFits(RepeatedUnitTests):
    N = 100

    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=1E-7)
    def test_fit_circle(self):
        noise = 0
        radius = (np.random.random() * 10 + 5,) * 2
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='xy')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=[0.15, 0.15, 0.2, 0.2])
    def test_fit_circle_noisy(self):
        noise = 0.2
        radius = (np.random.random() * 10 + 5,) * 2
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='xy')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=1E-7)
    def test_fit_ellipse_straight(self):
        noise = 0
        radius = np.random.random((2)) * 10 + 5
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='0')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=[0.15, 0.15, 0.2, 0.2])
    def test_fit_ellipse_straight_noisy(self):
        noise = 0.2
        radius = np.random.random((2)) * 10 + 5
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='0')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])

    #
    # def test_fit_ellipse(self):
    #     noise = 0.2
    #     radius = np.random.random((2)) * 10 + 5
    #     center = np.random.random((2)) * 10 + radius
    #     points, _ = ellipse_grid(radius, [0, 0])
    #     radius = [10, 2]
    #
    #     # rotation
    #     angle = np.random.random() * np.pi/6
    #     x_rot = points[1] * np.cos(angle) - points[0] * np.sin(angle)
    #     y_rot = points[1] * np.sin(angle) + points[0] * np.cos(angle)
    #     points[1] = x_rot
    #     points[0] = y_rot
    #
    #     # translation
    #     points += center[:, np.newaxis]
    #     #points += (np.random.random(points.shape) - 0.5) * noise

        # radius_fit, center_fit, angle_fit = gt.fit_ellipse(points, mode='')
        # assert_allclose(radius, radius_fit, atol=0.1)
        # assert_allclose(center, center_fit, atol=0.1)
        # assert_allclose(angle, angle_fit, atol=0.1)

    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                                'center_z', 'center_y', 'center_x'),
                     atol=1E-7)
    def test_fit_ellipsoid_straight(self):
        radius = np.random.random((3)) * 10 + 5
        center = np.random.random((3)) * 10 + radius
        points, _ = ellipsoid_grid(radius, center)

        radius_fit, center_fit, _ = fit_ellipsoid(points, mode='0')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                                'center_z', 'center_y', 'center_x'),
                     atol=0.1)
    def test_fit_ellipsoid_straight_noisy(self):
        noise = 0.2
        radius = np.random.random((3)) * 10 + 5
        center = np.random.random((3)) * 10 + radius
        points, _ = ellipsoid_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipsoid(points, mode='0')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])

    # def test_fit_ellipsoid_rotated(self):
    #     noise = 0.2
    #     radius = np.random.random((3)) * 10 + 5
    #     center = np.random.random((3)) * 10 + radius
    #     points, _ = ellipsoid_grid(radius, [0, 0, 0])
    #
    #     # rotation
    #     angle = np.random.random() * np.pi/6
    #     x_rot = points[2] * np.cos(angle) - points[1] * np.sin(angle)
    #     y_rot = points[2] * np.sin(angle) + points[1] * np.cos(angle)
    #     points[2] = x_rot
    #     points[1] = y_rot
    #
    #     # translation
    #     points += center[:, np.newaxis]
    #
    #     points += (np.random.random(points.shape) - 0.5) * noise
    #
    #     result = gt.fit_ellipsoid(points, mode='', return_mode='euler')
    #     assert_allclose([angle, 0, 0], result[2], atol=np.pi/100)
    #     assert_allclose(radius, result[0], atol=0.1)
    #     assert_allclose(center, result[1], atol=0.1)

    # def test_fit_ellipsoid_tilted(self):
    #     noise = 0.2
    #     radius = np.random.random((3)) * 10 + 5
    #     center = np.random.random((3)) * 10 + radius
    #     points, _ = ellipsoid_grid(radius, [0, 0, 0])
    #
    #     # tilt
    #     angle_z = np.random.random() * np.pi/6
    #     x_rot = points[2] * np.cos(angle_z) - points[0] * np.sin(angle_z)
    #     z_rot = points[2] * np.sin(angle_z) + points[0] * np.cos(angle_z)
    #     points[2] = x_rot
    #     points[0] = z_rot
    #
    #     # rotation
    #     angle = np.random.random() * np.pi/6
    #     x_rot = points[2] * np.cos(angle) - points[1] * np.sin(angle)
    #     y_rot = points[2] * np.sin(angle) + points[1] * np.cos(angle)
    #     points[2] = x_rot
    #     points[1] = y_rot
    #
    #     # translation
    #     points += center[:, np.newaxis]
    #
    #     points += (np.random.random(points.shape) - 0.5) * noise
    #
    #     result = gt.fit_ellipsoid(points, mode='', return_mode='euler')
    #     assert_allclose([angle, angle_z, 0], result[2], atol=np.pi/100)
    #     assert_allclose(radius, result[0], atol=0.1)
    #     assert_allclose(center, result[1], atol=0.1)


    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                                'center_z', 'center_y', 'center_x',
                                'skew_y', 'skew_x'),
                     atol=1E-7)
    def test_fit_ellipsoid_skewed(self):
        radius = np.random.random((3)) * 10 + 5
        radius[1] = radius[2]

        center = np.random.random((3)) * 10 + radius
        points, _ = ellipsoid_grid(radius, [0, 0, 0])
        skew = np.random.random((2)) * 4 - 2  # -2 - 2

        # rotation
        angle = np.random.random() * np.pi/6
        x_rot = points[2] * np.cos(angle) - points[1] * np.sin(angle)
        y_rot = points[2] * np.sin(angle) + points[1] * np.cos(angle)
        points[2] = x_rot
        points[1] = y_rot

        # skew
        points[1] += skew[0]*points[0]
        points[2] += skew[1]*points[0]

        # translation
        points += center[:, np.newaxis]

        result = fit_ellipsoid(points, mode='xy', return_mode='skew')

        return np.concatenate(result), \
               np.concatenate([radius, center, skew])

    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                                'center_z', 'center_y', 'center_x',
                                'skew_y', 'skew_x'),
                     atol=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.03, 0.03])
    def test_fit_ellipsoid_skewed_noisy(self):
        noise = 0.2
        radius = np.random.random((3)) * 10 + 5
        radius[1] = radius[2]

        center = np.random.random((3)) * 10 + radius
        points, _ = ellipsoid_grid(radius, [0, 0, 0])
        skew = np.random.random((2)) * 4 - 2  # -2 - 2

        # rotation
        angle = np.random.random() * np.pi/6
        x_rot = points[2] * np.cos(angle) - points[1] * np.sin(angle)
        y_rot = points[2] * np.sin(angle) + points[1] * np.cos(angle)
        points[2] = x_rot
        points[1] = y_rot

        # skew
        points[1] += skew[0]*points[0]
        points[2] += skew[1]*points[0]

        # translation
        points += center[:, np.newaxis]
        points += (np.random.random(points.shape) - 0.5) * noise

        result = fit_ellipsoid(points, mode='xy', return_mode='skew')

        return np.concatenate(result), \
               np.concatenate([radius, center, skew])


class TestEllipse(RepeatedUnitTests):
    N = 10
    shape = (300, 300)
    FWHM = 5

    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=[1, 1, 0.1, 0.1])
    def test_circle_no_refine(self):
        noise = 0.02
        radius = (np.random.random() * 50 + 50,) * 2
        padding = [r + self.FWHM * 3 for r in radius]
        center = tuple([np.random.random() * (s - 2 * p) + p
                       for (s, p) in zip(self.shape, padding)])
        im = draw_ellipse(self.shape, radius, center,
                          FWHM=self.FWHM, noise=noise)
        result = find_ellipse(im)

        return result, np.concatenate([radius, center])

    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=[None, None, 0.1, 0.1], rtol=[0.01, 0.01, None, None])
    def test_circle_noisy(self):
        noise = 0.2
        radius = (np.random.random() * 50 + 50,) * 2
        padding = [r + self.FWHM * 3 for r in radius]
        center = tuple([np.random.random() * (s - 2 * p) + p
                       for (s, p) in zip(self.shape, padding)])
        im = draw_ellipse(self.shape, radius, center,
                          FWHM=self.FWHM, noise=noise)
        result, _ = locate_ellipse(im)

        return result[['yr', 'xr', 'yc', 'xc']].values, \
               np.concatenate([radius, center])

    @repeat_test_std(N, names=('radius_y', 'radius_x', 'center_y', 'center_x'),
                     atol=[None, None, 0.1, 0.1], rtol=[0.01, 0.01, None, None])
    def test_ellipse_noisy(self):
        noise = 0.2
        radius = tuple(np.random.random(2) * 50 + 50,)
        padding = [r + self.FWHM * 3 for r in radius]
        center = tuple([np.random.random() * (s - 2 * p) + p
                       for (s, p) in zip(self.shape, padding)])
        im = draw_ellipse(self.shape, radius, center,
                          FWHM=self.FWHM, noise=noise)
        result, _ = locate_ellipse(im)
        return result[['yr', 'xr', 'yc', 'xc']].values, \
               np.concatenate([radius, center])


class TestEllipsoid(RepeatedUnitTests):
    N = 10
    shape = (250, 300, 300)
    FWHM = 5

    def gen_center_radius(self):
        R = np.random.random() * 50 + 50
        ar = np.random.random() * 4 + 1
        radius = (R/ar, R, R)
        padding = [r + self.FWHM * 3 for r in radius]
        center = tuple([np.random.random() * (s - 2 * p) + p
                       for (s, p) in zip(self.shape, padding)])
        return center, radius

    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                               'center_z', 'center_y', 'center_x'),
                     atol=[5, 5, 5, 1, 1, 1])
    def test_no_refine_noisy(self):
        center, radius = self.gen_center_radius()
        im = draw_ellipsoid(self.shape, radius, center,
                            FWHM=self.FWHM, noise=0.2)
        result = find_ellipsoid(im)
        return result, radius + center

    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                           'center_z', 'center_y', 'center_x'),
                     atol=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    def test_locate_fast(self):
        center, radius = self.gen_center_radius()
        im = draw_ellipsoid(self.shape, radius, center,
                            FWHM=self.FWHM, noise=0.02)
        result, _ = locate_ellipsoid_fast(im)
        return result[['zr', 'yr', 'xr', 'zc', 'yc', 'xc']].values, \
               np.concatenate([radius, center])

    @repeat_test_std(N, names=('radius_z', 'radius_y', 'radius_x',
                               'center_z', 'center_y', 'center_x'),
                     atol=[1, 0.5, 0.5, 0.1, 0.1, 0.1])
    def test_locate_noisy(self):
        center, radius = self.gen_center_radius()
        im = draw_ellipsoid(self.shape, radius, center,
                            FWHM=self.FWHM, noise=0.2)
        result, _ = locate_ellipsoid(im)
        return result[['zr', 'yr', 'xr', 'zc', 'yc', 'xc']].values, \
               np.concatenate([radius, center])


class TestDisks(RepeatedUnitTests):
    """ Test case for finding circular disks """
    number = 1
    radii = [15.0, 20.0, 25.0]

    def generate_image(self, radius, n, noise=0.02):
        """ Generate the test image """
        image = SimulatedImage(shape=(300, 300), radius=radius,
                               noise=noise)
        image.draw_features(n, margin=2*radius, separation=2*radius + 2)
        return image

    @repeat_test_std(number, names=('radius', 'y_coord', 'x_coord'), atol=5)
    def test_find_single(self):
        """ Test finding single particle """
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 1)

        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          number_of_disks=1)

        y_coord, x_coord = generated_image.coords[0]
        if len(fits) != 1:  # Particle number mismatch
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, x, y = fits[['r', 'x', 'y']].values[0]

        return (r, y, x), (radius, y_coord, x_coord)

    @repeat_test_std(number, names=('radius', 'y_coord', 'x_coord'), atol=5)
    def test_find_single_noisy(self):
        """ Test find single noisy particle """
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 1, noise=0.2)

        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          number_of_disks=1)

        y_coord, x_coord = generated_image.coords[0]
        if len(fits) != 1: # Particle number mismatch
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, x, y = fits[['r', 'x', 'y']].values[0]

        return (r, y, x), (radius, y_coord, x_coord)

    @repeat_test_std(number, names=('radii', 'y_coords', 'x_coords'), atol=5)
    def test_find_multiple_noisy(self):
        """ Test finding multiple particles (noisy) """
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 10, noise=0.2)
        actual_number = len(generated_image.coords)
        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          number_of_disks=actual_number)

        _, coords = sort_positions(generated_image.coords,
                                   np.array([fits['y'].values,
                                             fits['x'].values]).T)

        if len(fits) == 0:  # Nothing found
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, y, x = fits[['r', 'y', 'x']].values.astype(np.float64).T

        return (r, y, x), (np.full(actual_number, radius, np.float64),
                           coords[:, 0], coords[:, 1])

    @repeat_test_std(number, names=('radius', 'y_coord', 'x_coord'),
                     atol=0.1)
    def test_locate_single_noisy(self):
        """ Test locating single particle (noisy) """
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 1, noise=0.2)

        fits = locate_disks(generated_image.image, (radius / 2.0,
                                                    radius * 2.0),
                            number_of_disks=1)

        y_coord, x_coord = generated_image.coords[0]
        if len(fits) != 1:  # Particle number mismatch
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, x, y = fits[['r', 'x', 'y']].values[0]

        return (r, y, x), (radius, y_coord, x_coord)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
