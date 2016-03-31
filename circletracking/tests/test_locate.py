""" Nosetests for finding features """
from __future__ import (division, unicode_literals)

import nose
import numpy as np
import unittest
from circletracking import (ellipse_grid, ellipsoid_grid, locate_ellipse,
                            draw_ellipse, draw_ellipsoid, fit_ellipse,
                            fit_ellipsoid, find_ellipsoid, find_ellipse,
                            locate_ellipsoid, locate_ellipsoid_fast,
                            SimulatedImage, find_disks, locate_disks)
from circletracking.tests.common import (RepeatedUnitTests, repeat_test_std,
                                         sort_positions)


class TestFits(RepeatedUnitTests, unittest.TestSuite):
    repeats = 100
    names = ('radius_y', 'radius_x', 'center_y', 'center_x')

    @repeat_test_std
    def test_fit_circle(self):
        self.atol = 1E-7
        noise = 0
        radius = (np.random.random() * 10 + 5,) * 2
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='xy')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std
    def test_fit_circle_noisy(self):
        self.atol = [0.15, 0.15, 0.2, 0.2]
        noise = 0.2
        radius = (np.random.random() * 10 + 5,) * 2
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='xy')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std
    def test_fit_ellipse_straight(self):
        self.atol = 1E-7
        noise = 0
        radius = np.random.random((2)) * 10 + 5
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='0')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std
    def test_fit_ellipse_straight_noisy(self):
        self.atol = [0.15, 0.15, 0.2, 0.2]
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

    @repeat_test_std
    def test_fit_ellipsoid_straight(self):
        self.names = ('radius_z', 'radius_y', 'radius_x',
                                'center_z', 'center_y', 'center_x')
        self.atol = 1E-7
        radius = np.random.random((3)) * 10 + 5
        center = np.random.random((3)) * 10 + radius
        points, _ = ellipsoid_grid(radius, center)

        radius_fit, center_fit, _ = fit_ellipsoid(points, mode='0')

        return np.concatenate([radius_fit, center_fit]), \
               np.concatenate([radius, center])


    @repeat_test_std
    def test_fit_ellipsoid_straight_noisy(self):
        self.names = ('radius_z', 'radius_y', 'radius_x',
                                'center_z', 'center_y', 'center_x')
        self.atol = 0.1
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


    @repeat_test_std
    def test_fit_ellipsoid_skewed(self):
        self.names = ('radius_z', 'radius_y', 'radius_x',
                      'center_z', 'center_y', 'center_x',
                      'skew_y', 'skew_x')
        self.atol = 1E-7
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

    @repeat_test_std
    def test_fit_ellipsoid_skewed_noisy(self):
        self.names = ('radius_z', 'radius_y', 'radius_x',
                      'center_z', 'center_y', 'center_x',
                      'skew_y', 'skew_x')
        self.atol = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.03, 0.03]
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


class TestEllipse(RepeatedUnitTests, unittest.TestSuite):
    repeats = 10
    shape = (300, 300)
    FWHM = 5
    names = ('radius_y', 'radius_x', 'center_y', 'center_x')
    @repeat_test_std
    def test_circle_no_refine(self):
        self.atol = [1, 1, 0.1, 0.1]
        noise = 0.02
        radius = (np.random.random() * 50 + 50,) * 2
        padding = [r + self.FWHM * 3 for r in radius]
        center = tuple([np.random.random() * (s - 2 * p) + p
                       for (s, p) in zip(self.shape, padding)])
        im = draw_ellipse(self.shape, radius, center,
                          FWHM=self.FWHM, noise=noise)
        result = find_ellipse(im)

        return result, np.concatenate([radius, center])

    @repeat_test_std
    def test_circle_noisy(self):
        self.atol = [None, None, 0.1, 0.1]
        self.rtol = [0.01, 0.01, None, None]
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

    @repeat_test_std
    def test_ellipse_noisy(self):
        self.atol = [None, None, 0.1, 0.1]
        self.rtol = [0.01, 0.01, None, None]
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


class TestEllipsoid(RepeatedUnitTests, unittest.TestSuite):
    repeats = 10
    shape = (250, 300, 300)
    FWHM = 5
    names = ('radius_z', 'radius_y', 'radius_x',
             'center_z', 'center_y', 'center_x')

    def gen_center_radius(self):
        R = np.random.random() * 50 + 50
        ar = np.random.random() * 4 + 1
        radius = (R/ar, R, R)
        padding = [r + self.FWHM * 3 for r in radius]
        center = tuple([np.random.random() * (s - 2 * p) + p
                       for (s, p) in zip(self.shape, padding)])
        return center, radius

    @repeat_test_std
    def test_no_refine_noisy(self):
        self.atol = [5, 5, 5, 1, 1, 1]
        center, radius = self.gen_center_radius()
        im = draw_ellipsoid(self.shape, radius, center,
                            FWHM=self.FWHM, noise=0.2)
        result = find_ellipsoid(im)
        return result, radius + center

    @repeat_test_std
    def test_locate_fast(self):
        self.atol = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        center, radius = self.gen_center_radius()
        im = draw_ellipsoid(self.shape, radius, center,
                            FWHM=self.FWHM, noise=0.02)
        result, _ = locate_ellipsoid_fast(im)
        return result[['zr', 'yr', 'xr', 'zc', 'yc', 'xc']].values, \
               np.concatenate([radius, center])

    @repeat_test_std
    def test_locate_noisy(self):
        self.atol = [1, 0.5, 0.5, 0.1, 0.1, 0.1]
        center, radius = self.gen_center_radius()
        im = draw_ellipsoid(self.shape, radius, center,
                            FWHM=self.FWHM, noise=0.2)
        result, _ = locate_ellipsoid(im)
        return result[['zr', 'yr', 'xr', 'zc', 'yc', 'xc']].values, \
               np.concatenate([radius, center])


class TestDisks(RepeatedUnitTests, unittest.TestCase):
    """ Test case for finding circular disks """
    repeats = 20
    names = ('radius', 'y_coord', 'x_coord')

    def generate_image(self, radius, n, noise=0.02):
        """ Generate the test image """
        image = SimulatedImage(shape=(300, 300), radius=radius,
                               noise=noise)
        image.draw_features(n, margin=2*radius, separation=2*radius + 2)
        return image

    @repeat_test_std
    def test_find_single(self):
        """ Test finding single particle """
        self.atol = 5
        radius = np.random.uniform(15, 30)
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

    @repeat_test_std
    def test_find_single_noisy(self):
        """ Test find single noisy particle """
        self.atol = 5
        radius = np.random.uniform(15, 30)
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

    @repeat_test_std
    def test_find_multiple_noisy(self):
        """ Test finding multiple particles (noisy) """
        self.atol = 5
        radius = np.random.uniform(15, 30)
        generated_image = self.generate_image(radius, 10, noise=0.2)
        actual_number = len(generated_image.coords)
        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          number_of_disks=actual_number)

        _, coords = sort_positions(generated_image.coords,
                                   np.array([fits['y'].values,
                                             fits['x'].values]).T)

        if len(fits) == 0:  # Nothing found
            actual = np.repeat([[np.nan, np.nan, np.nan]], actual_number,
                               axis=0)
        else:
            actual = fits[['r', 'y', 'x']].values.astype(np.float64)

        expected = np.array([np.full(actual_number, radius, np.float64),
                             coords[:, 0], coords[:, 1]]).T

        return np.sqrt(((actual - expected)**2).mean(0)), [0] * 3

    @repeat_test_std
    def test_locate_single_noisy(self):
        """ Test locating single particle (noisy) """
        self.atol = 0.5
        radius = np.random.uniform(15, 30)
        generated_image = self.generate_image(radius, 1, noise=0.2)

        fits = locate_disks(generated_image.image, (radius / 2.0,
                                                    radius * 2.0),
                            number_of_disks=1)

        y_coord, x_coord = generated_image.coords[0]
        if len(fits) != 1:  # Particle number mismatch
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, y, x = fits[['r', 'y', 'x']].values[0]

        return (r, y, x), (radius, y_coord, x_coord)

    @repeat_test_std
    def test_locate_multiple_noisy(self):
        """ Test locating multiple particles (noisy) """
        self.atol = 0.5
        radius = np.random.uniform(15, 30)
        generated_image = self.generate_image(radius, 10, noise=0.2)
        actual_number = len(generated_image.coords)
        fits = locate_disks(generated_image.image, (radius / 2.0,
                                                    radius * 2.0),
                            number_of_disks=actual_number)

        _, coords = sort_positions(generated_image.coords,
                                   np.array([fits['y'].values,
                                             fits['x'].values]).T)

        if len(fits) == 0:  # Nothing found
            actual = np.repeat([[np.nan, np.nan, np.nan]], actual_number,
                               axis=0)
        else:
            actual = fits[['r', 'y', 'x']].values.astype(np.float64)

        expected = np.array([np.full(actual_number, radius, np.float64),
                             coords[:, 0], coords[:, 1]]).T

        return np.sqrt(((actual - expected)**2).mean(0)), [0] * 3

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
