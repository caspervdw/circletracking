""" Nosetests for finding and locating disks """
from __future__ import (division, unicode_literals)

import nose
import numpy as np
import unittest
from circletracking import (SimulatedImage, find_disks, locate_disks)
from circletracking.tests.common import (RepeatedUnitTests, repeat_check_std,
                                         sort_positions)


class TestDisks(RepeatedUnitTests, unittest.TestCase):
    """ Test case for finding circular disks """
    repeats = 20
    names = ('radius', 'y_coord', 'x_coord')

    def generate_image(self, n=1, noise=0.02, shape=(300, 300),
                       radius_range=(15, 30)):
        """ Generate the test image """
        radius = np.random.uniform(*radius_range)
        image = SimulatedImage(shape=shape, radius=radius,
                               noise=noise)
        image.draw_features(n, margin=2 * radius, separation=2 * radius + 2)
        return image

    def find_or_locate_single(self, noise=0.02, mode='find'):
        """ Find single particles, with optional noise """
        im = self.generate_image(n=1, noise=noise)

        if mode == 'find':
            fits = find_disks(im.image, (im.radius / 2.0,
                                         im.radius * 2.0),
                              number_of_disks=1)
        else:
            fits = locate_disks(im.image, (im.radius / 2.0,
                                           im.radius * 2.0),
                                number_of_disks=1)

        y_coord, x_coord = im.coords[0]
        if len(fits) != 1:  # Particle number mismatch
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, x, y = fits[['r', 'x', 'y']].values[0]

        if mode == 'locate':
            assert fits.refined.all()

        return (r, y, x), (im.radius, y_coord, x_coord)

    def find_or_locate_multiple(self, noise=0.02, mode='find'):
        """ Find multiple particles, with optional noise """
        im = self.generate_image(n=10, noise=noise)
        actual_number = len(im.coords)

        if mode == 'find':
            fits = find_disks(im.image, (im.radius / 2.0,
                                         im.radius * 2.0),
                              number_of_disks=actual_number)
        else:
            fits = locate_disks(im.image, (im.radius / 2.0,
                                           im.radius * 2.0),
                                number_of_disks=actual_number)

        _, coords = sort_positions(im.coords,
                                   np.array([fits['y'].values,
                                             fits['x'].values]).T)

        if len(fits) == 0:  # Nothing found
            actual = np.repeat([[np.nan, np.nan, np.nan]], actual_number,
                               axis=0)
        else:
            actual = fits[['r', 'y', 'x']].values.astype(np.float64)

        expected = np.array([np.full(actual_number, im.radius, np.float64),
                             coords[:, 0], coords[:, 1]]).T

        if mode == 'locate':
            assert fits.refined.all()

        return np.sqrt(((actual - expected)**2).mean(0)), [0] * 3

    @repeat_check_std
    def test_find_single(self):
        """ Test finding single particle """
        self.atol = 5
        return self.find_or_locate_single(noise=0.02, mode='find')

    @repeat_check_std
    def test_find_single_noisy(self):
        """ Test find single noisy particle """
        self.atol = 5
        return self.find_or_locate_single(noise=0.2, mode='find')

    @repeat_check_std
    def test_find_multiple(self):
        """ Test finding multiple particles """
        self.atol = 5
        return self.find_or_locate_multiple(noise=0.02, mode='find')

    @repeat_check_std
    def test_find_multiple_noisy(self):
        """ Test finding multiple particles (noisy) """
        self.atol = 5
        return self.find_or_locate_multiple(noise=0.2, mode='find')

    @repeat_check_std
    def test_locate_single(self):
        """ Test locating single particle """
        self.atol = 0.5
        return self.find_or_locate_single(noise=0.02, mode='locate')

    @repeat_check_std
    def test_locate_single_noisy(self):
        """ Test locating single particle (noisy) """
        self.atol = 0.5
        return self.find_or_locate_single(noise=0.2, mode='locate')

    @repeat_check_std
    def test_locate_multiple(self):
        """ Test locating multiple particles """
        self.atol = 0.5
        return self.find_or_locate_multiple(noise=0.02, mode='locate')

    @repeat_check_std
    def test_locate_multiple_noisy(self):
        """ Test locating multiple particles (noisy) """
        self.atol = 0.5
        return self.find_or_locate_multiple(noise=0.2, mode='locate')

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
