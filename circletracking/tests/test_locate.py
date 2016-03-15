""" Nosetests for finding features """
from __future__ import (division, unicode_literals)

import unittest
import nose
from numpy.testing import assert_allclose, assert_equal
import numpy as np
import pandas
from circletracking import (ellipse_grid, ellipsoid_grid, locate_ellipse,
                            draw_ellipse, draw_ellipsoid, fit_ellipse,
                            fit_ellipsoid, find_ellipsoid, find_ellipse,
                            locate_ellipsoid, locate_ellipsoid_fast,
                            SimulatedImage, find_disks)


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


class TestEllipse(unittest.TestCase):
    def setUp(self):
        self.shape = (300, 300)
        self.FWHM = 5
        self.N = 10

    def test_circle_no_refine(self):
        NOREFINE_CENTER_ATOL = 1
        NOREFINE_RADIUS_RTOL = 0.1
        noise = 0.02
        for _ in range(self.N):
            radius = (np.random.random() * 50 + 50,) * 2
            padding = [r + self.FWHM * 3 for r in radius]
            center = tuple([np.random.random() * (s - 2 * p) + p
                           for (s, p) in zip(self.shape, padding)])
            im = draw_ellipse(self.shape, radius, center,
                              FWHM=self.FWHM, noise=noise)
            result = find_ellipse(im)
            assert_allclose(result[0], radius[0], rtol=NOREFINE_RADIUS_RTOL)
            assert_allclose(result[1], radius[1], rtol=NOREFINE_RADIUS_RTOL)
            assert_allclose(result[2], center[0], atol=NOREFINE_CENTER_ATOL)
            assert_allclose(result[3], center[1], atol=NOREFINE_CENTER_ATOL)

    def test_circle_noisy(self):
        NOISY_CENTER_ATOL = 0.1
        NOISY_RADIUS_RTOL = 0.01
        noise = 0.2
        for _ in range(self.N):
            radius = (np.random.random() * 50 + 50,) * 2
            padding = [r + self.FWHM * 3 for r in radius]
            center = tuple([np.random.random() * (s - 2 * p) + p
                           for (s, p) in zip(self.shape, padding)])
            im = draw_ellipse(self.shape, radius, center,
                              FWHM=self.FWHM, noise=noise)
            result, _ = locate_ellipse(im)
            assert_allclose(result['yc'], center[0], atol=NOISY_CENTER_ATOL)
            assert_allclose(result['xc'], center[1], atol=NOISY_CENTER_ATOL)
            assert_allclose(result['yr'], radius[0], rtol=NOISY_RADIUS_RTOL)
            assert_allclose(result['xr'], radius[1], rtol=NOISY_RADIUS_RTOL)


    def test_circle(self):
        CENTER_ATOL = 0.1
        RADIUS_RTOL = 0.01
        noise = 0.02
        for _ in range(self.N):
            radius = (np.random.random() * 50 + 50,) * 2
            padding = [r + self.FWHM * 3 for r in radius]
            center = tuple([np.random.random() * (s - 2 * p) + p
                           for (s, p) in zip(self.shape, padding)])
            im = draw_ellipse(self.shape, radius, center,
                              FWHM=self.FWHM, noise=noise)
            result, _ = locate_ellipse(im)
            assert_allclose(result['yc'], center[0], atol=CENTER_ATOL)
            assert_allclose(result['xc'], center[1], atol=CENTER_ATOL)
            assert_allclose(result['yr'], radius[0], rtol=RADIUS_RTOL)
            assert_allclose(result['xr'], radius[1], rtol=RADIUS_RTOL)

    def test_ellipse(self):
        CENTER_ATOL = 0.1
        RADIUS_RTOL = 0.01
        noise = 0.02
        for _ in range(self.N):
            radius = tuple(np.random.random(2) * 50 + 50,)
            padding = [r + self.FWHM * 3 for r in radius]
            center = tuple([np.random.random() * (s - 2 * p) + p
                           for (s, p) in zip(self.shape, padding)])
            im = draw_ellipse(self.shape, radius, center,
                              FWHM=self.FWHM, noise=noise)
            result, _ = locate_ellipse(im)
            assert_allclose(result['yc'], center[0], atol=CENTER_ATOL)
            assert_allclose(result['xc'], center[1], atol=CENTER_ATOL)
            assert_allclose(result['yr'], radius[0], rtol=RADIUS_RTOL)
            assert_allclose(result['xr'], radius[1], rtol=RADIUS_RTOL)


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
        im = draw_ellipsoid(self.shape, self.radius, self.center,
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
        im = draw_ellipsoid(self.shape, self.radius, self.center,
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
        im = draw_ellipsoid(self.shape, self.radius, self.center,
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
        im = draw_ellipsoid(self.shape, self.radius, self.center,
                            FWHM=self.FWHM, noise=self.noise)
        result, _ = locate_ellipsoid(im)
        assert_allclose(result['zc'], self.center[0], atol=CENTER_ATOL)
        assert_allclose(result['yc'], self.center[1], atol=CENTER_ATOL)
        assert_allclose(result['xc'], self.center[2], atol=CENTER_ATOL)
        assert_allclose(result['zr'], self.radius[0], rtol=RADIUS_RTOL)
        assert_allclose(result['yr'], self.radius[1], rtol=RADIUS_RTOL)
        assert_allclose(result['xr'], self.radius[2], rtol=RADIUS_RTOL)


class TestCircles(unittest.TestCase):
    """
    Test case for finding circular disks
    """
    def setUp(self):
        """
        Setup test image
        """
        self.number = 100
        self.radius = 15.0
        self.image = self.generate_image()

    def generate_image(self):
        """
        Generate the test image

        :rtype : semtracking.SimulatedImage
        :return:
        """
        image = SimulatedImage(shape=(1024, 943), radius=self.radius,
                               noise=0.3)
        image.draw_features(self.number, separation=3 * self.radius)
        return image

    def get_coords_dataframe(self, add_r=False):
        """
        Get a dataframe with the generated coordinates
        :rtype : pandas.DataFrame
        """
        coords = self.image.coords
        coords_df = pandas.DataFrame(data=coords, columns=['y', 'x'])

        # Add r
        if add_r:
            coords_df['r'] = pandas.Series(self.radius, index=coords_df.index)

        return coords_df

    @staticmethod
    def find_closest_index(expected_row, actual_df, intersection, columns,
                           tolerance=5.0):
        """
        Find the row in the dataframe that has the closest resemblence to the
        expected row.
        """
        if len(columns) != 2:
            return False
        column1, column2 = columns

        diffs = {i: abs(expected_row[column1] - actual_df.loc[i][column1])
                    + abs(expected_row[column2] - actual_df.loc[i][column2])
                 for i in intersection}

        key = min(diffs, key=diffs.get)

        if diffs[key] / 2.0 < tolerance:
            return key
        else:
            return False

    def find_intersections(self, expected_row, actual_df, closest_dict, columns,
                           tolerance=5.0):
        """
        Find intersections between the expected and actual rows.
        """
        if len(columns) != 2:
            return False
        column1, column2 = columns

        intersection = list(set(closest_dict[column1]).intersection(
            closest_dict[column2]))

        if len(intersection) == 0:
            return False
        else:
            return self.find_closest_index(expected_row, actual_df,
                                           intersection, columns, tolerance)

    def find_closest_row_index(self, expected_row, actual_df, columns,
                               tolerance=5.0):
        """
        Find the index of the row closest to expected_row
        """
        closest = {c: (actual_df[c] - expected_row[c]).abs().argsort()[:10]
                   for c in columns}
        return self.find_intersections(expected_row, actual_df, closest,
                                       columns, tolerance)

    def check_frames_difference(self, actual, expected, sizecol='r'):
        """
        Compare DataFrame items by index and column and
        raise AssertionError if any item is not equal.

        Ordering is unimportant, items are compared only by label.
        NaN and infinite values are supported.

        Parameters
        ----------
        actual : pandas.DataFrame
        expected : pandas.DataFrame
        use_close : bool, optional
            If True, use numpy.testing.assert_allclose instead of
            numpy.testing.assert_equal.

        """
        unmatched = 0
        value_diff = pandas.DataFrame(columns=expected.columns)

        for _, exp_row in expected.iterrows():
            # tolerance in pixels
            tolerance = max(exp_row[sizecol] * 0.1, 5.0)
            act_row_index = self.find_closest_row_index(exp_row, actual,
                                                        (u'x', u'y'),
                                                        tolerance)
            if act_row_index is False:
                unmatched += 1
                continue

            act_row = actual.loc[act_row_index]
            diff = exp_row - act_row[expected.columns]
            value_diff = value_diff.append(diff, ignore_index=True)
        return unmatched, value_diff

    def test_locate_particles(self):
        """
        Test locating particles
        """
        generated_image = self.image()

        fits = find_disks(generated_image, (self.radius / 2.0,
                                            self.radius * 2.0),
                          number_of_disks=self.number)

        coords_df = self.get_coords_dataframe(True)

        unmatched, value_diff = self.check_frames_difference(fits, coords_df)

        assert_allclose(value_diff['r'], np.zeros_like(value_diff['r']),
                        atol=0.03*self.radius, err_msg='Radius not accurate.')
        assert_allclose(value_diff['x'], np.zeros_like(value_diff['x']),
                        atol=1, err_msg='X not accurate.')
        assert_allclose(value_diff['y'], np.zeros_like(value_diff['y']),
                        atol=1, err_msg='Y not accurate.')
        assert_equal(unmatched, 0, 'Not all particles found, %d unmatched'
                     % unmatched)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
