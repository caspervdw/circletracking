from __future__ import (division, unicode_literals)

import numpy as np
from circletracking import (ellipse_grid, ellipsoid_grid,
                            gen_artificial_ellipsoid, fit_ellipse,
                            fit_ellipsoid, find_ellipsoid,
                            locate_ellipsoid, locate_ellipsoid_fast)
import unittest
import nose
from numpy.testing import assert_allclose


class TestFits(unittest.TestCase):
    def test_fit_circle(self):
        noise = 0.2
        radius = (np.random.random() * 10 + 5,) * 2
        center = np.random.random((2)) * 10 + radius

        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='xy')
        assert_allclose(radius, radius_fit, atol=0.1)
        assert_allclose(center, center_fit, atol=0.1)

    def test_fit_ellipse_straight(self):
        noise = 0.2
        radius = np.random.random((2)) * 10 + 5
        center = np.random.random((2)) * 10 + radius
        points, _ = ellipse_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipse(points, mode='0')
        assert_allclose(radius, radius_fit, atol=0.1)
        assert_allclose(center, center_fit, atol=0.1)
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

    def test_fit_ellipsoid_straight(self):
        noise = 0.2
        radius = np.random.random((3)) * 10 + 5
        center = np.random.random((3)) * 10 + radius
        points, _ = ellipsoid_grid(radius, center)
        points += (np.random.random(points.shape) - 0.5) * noise

        radius_fit, center_fit, _ = fit_ellipsoid(points, mode='0')
        assert_allclose(radius, radius_fit, atol=0.1)
        assert_allclose(center, center_fit, atol=0.1)

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

    def test_fit_ellipsoid_skewed(self):
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
        assert_allclose(skew, result[2], atol=0.01)
        assert_allclose(radius, result[0], atol=0.1)
        assert_allclose(center, result[1], atol=0.1)

class TestEllipsoid(unittest.TestCase):
    def setUp(self):
        self.shape = (250, 300, 300)
        self.FWHM = 5
        R = np.random.random() * 50 + 50
        ar = np.random.random() * 4 + 1
        self.radius = (R/ar, R, R)
        padding = [r + self.FWHM * 3 for r in self.radius]
        self.center = tuple([np.random.random() * (s - 2 * p) + p
                            for (s, p) in zip(self.shape, padding)])
        self.noise = 0.02

    def test_no_refine(self):
        NOREFINE_CENTER_ATOL = 1
        NOREFINE_RADIUS_RTOL = 0.1
        im = gen_artificial_ellipsoid(self.shape, self.radius, self.center,
                                      FWHM=self.FWHM, noise=self.noise)
        result = find_ellipsoid(im)
        assert_allclose(result[0], self.radius[0], rtol=NOREFINE_RADIUS_RTOL)
        assert_allclose(result[1], self.radius[1], rtol=NOREFINE_RADIUS_RTOL)
        assert_allclose(result[2], self.radius[2], rtol=NOREFINE_RADIUS_RTOL)
        assert_allclose(result[3], self.center[0], atol=NOREFINE_CENTER_ATOL)
        assert_allclose(result[4], self.center[1], atol=NOREFINE_CENTER_ATOL)
        assert_allclose(result[5], self.center[2], atol=NOREFINE_CENTER_ATOL)

    def test_locate_fast(self):
        CENTER_ATOL_FAST = 1
        RADIUS_RTOL_FAST = 0.05
        im = gen_artificial_ellipsoid(self.shape, self.radius, self.center,
                                      FWHM=self.FWHM, noise=self.noise)
        result, _ = locate_ellipsoid_fast(im)
        assert_allclose(result['zc'], self.center[0], atol=CENTER_ATOL_FAST)
        assert_allclose(result['yc'], self.center[1], atol=CENTER_ATOL_FAST)
        assert_allclose(result['xc'], self.center[2], atol=CENTER_ATOL_FAST)
        assert_allclose(result['zr'], self.radius[0], rtol=RADIUS_RTOL_FAST)
        assert_allclose(result['yr'], self.radius[1], rtol=RADIUS_RTOL_FAST)
        assert_allclose(result['xr'], self.radius[2], rtol=RADIUS_RTOL_FAST)

    def test_locate_noisy(self):
        NOISY_CENTER_ATOL = 0.5
        NOISY_RADIUS_RTOL = 0.05
        noise = 0.2
        im = gen_artificial_ellipsoid(self.shape, self.radius, self.center,
                                      FWHM=self.FWHM, noise=noise)
        result, _ = locate_ellipsoid(im)
        assert_allclose(result['zc'], self.center[0], atol=NOISY_CENTER_ATOL)
        assert_allclose(result['yc'], self.center[1], atol=NOISY_CENTER_ATOL)
        assert_allclose(result['xc'], self.center[2], atol=NOISY_CENTER_ATOL)
        assert_allclose(result['zr'], self.radius[0], rtol=NOISY_RADIUS_RTOL)
        assert_allclose(result['yr'], self.radius[1], rtol=NOISY_RADIUS_RTOL)
        assert_allclose(result['xr'], self.radius[2], rtol=NOISY_RADIUS_RTOL)

    def test_locate(self):
        CENTER_ATOL = 0.1
        RADIUS_RTOL = 0.03
        im = gen_artificial_ellipsoid(self.shape, self.radius, self.center,
                                      FWHM=self.FWHM, noise=self.noise)
        result, _ = locate_ellipsoid(im)
        assert_allclose(result['zc'], self.center[0], atol=CENTER_ATOL)
        assert_allclose(result['yc'], self.center[1], atol=CENTER_ATOL)
        assert_allclose(result['xc'], self.center[2], atol=CENTER_ATOL)
        assert_allclose(result['zr'], self.radius[0], rtol=RADIUS_RTOL)
        assert_allclose(result['yr'], self.radius[1], rtol=RADIUS_RTOL)
        assert_allclose(result['xr'], self.radius[2], rtol=RADIUS_RTOL)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
