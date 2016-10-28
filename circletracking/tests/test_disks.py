""" Nosetests for finding features """
from __future__ import (division, unicode_literals)

import nose
import numpy as np
import unittest
from circletracking import SimulatedImage, find_disks, locate_disks
from circletracking.tests.common import (RepeatedUnitTests, repeat_test_std,
                                         sort_positions)

class TestDisks(RepeatedUnitTests, unittest.TestCase):
    """ Test case for finding circular disks """
    radii = [15.0, 20.0, 25.0]
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
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 1)

        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          maximum=1)

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
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 1, noise=0.2)

        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          maximum=1)

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
        radius = np.random.random() * 15 + 15
        generated_image = self.generate_image(radius, 10, noise=0.2)
        actual_number = len(generated_image.coords)
        fits = find_disks(generated_image.image, (radius / 2.0,
                                                  radius * 2.0),
                          maximum=actual_number)

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
                            maximum=1)

        y_coord, x_coord = generated_image.coords[0]
        if len(fits) != 1:  # Particle number mismatch
            r, y, x = np.nan, np.nan, np.nan
        else:
            r, y, x = fits[['r', 'y', 'x']].values[0]

        return (r, y, x), (radius, y_coord, x_coord)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
