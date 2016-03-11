from __future__ import division
import numpy as np
import trackpy as tp
import pims


class SimulatedImage(object):
    """ This class makes it easy to generate artificial pictures.

    Parameters
    ----------
    shape : tuple of int
    dtype : numpy.dtype, default np.float64
    saturation : maximum value in image
    radius : default radius of particles, used for determining the
                  distance between particles in clusters
    feat_dict : dictionary of arguments passed to tp.artificial.draw_feature

    Attributes
    ----------
    image : ndarray containing pixel values
    center : the center [y, x] to use for radial coordinates

    Examples
    --------
    image = SimulatedImage(shape=(50, 50), dtype=np.float64, radius=7,
                           feat_dict={'diameter': 20, 'max_value': 100,
                                      'feat_func': SimulatedImage.feat_hat,
                                      'disc_size': 0.2})
    image.draw_feature((10, 10))
    image.draw_dimer((32, 35), angle=75)
    image.add_noise(5)
    image()
    """

    def __init__(self, shape,
                 radius=None, noise=0.0,
                 feat_func=tp.artificial.feat_step, **feat_kwargs):
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = np.float64
        self.image = pims.Frame(np.zeros(shape, dtype=self.dtype))
        self.feat_func = feat_func
        self.feat_kwargs = feat_kwargs
        self.noise = float(noise)
        self.center = tuple([float(s) / 2.0 for s in shape])
        self.radius = float(radius)
        self._coords = []
        self.pos_columns = ['z', 'y', 'x'][-self.ndim:]
        self.size_columns = ['size']

    def __call__(self):
        # so that you can checkout the image with image() instead of image.image
        return self.noisy_image(self.noise)

    def clear(self):
        """Clears the current image"""
        self._coords = []
        self.image = np.zeros_like(self.image)

    def normalize_image(self, image):
        """
        Normalize image
        :param image:
        :return:
        """
        image = image.astype(self.dtype)
        abs_max = np.max(np.abs(image))
        return image / abs_max

    def noisy_image(self, noise_level):
        """Adds noise to the current image, uniformly distributed
        between 0 and `noise_level`, not including noise_level."""
        if noise_level <= 0:
            return self.image

        noise = np.random.random(self.shape) * noise_level
        noisy_image = self.normalize_image(self.image + noise)
        return pims.Frame(np.array(noisy_image, dtype=self.dtype))

    @property
    def coords(self):
        if len(self._coords) == 0:
            return np.zeros((0, self.ndim), dtype=self.dtype)
        return np.array(self._coords)

    def draw_feature(self, pos):
        """Draws a feature at `pos`."""
        pos = [float(p) for p in pos]
        self._coords.append(pos)
        tp.artificial.draw_feature(image=self.image, position=pos, diameter=2.0 * self.radius, max_value=1.0,
                                   feat_func=self.feat_func, **self.feat_kwargs)

    def draw_features(self, count, separation=0.0, margin=None):
        """Draws N features at random locations, using minimum separation
        and a margin. If separation > 0, less than N features may be drawn."""
        if margin is None:
            margin = float(self.radius)
        if margin is None:
            margin = 0.0
        pos = self.gen_nonoverlapping_locations(self.shape, count, separation, margin)
        for p in pos:
            self.draw_feature(p)
        return pos

    @staticmethod
    def gen_random_locations(shape, count, margin=0.0):
        """ Generates `count` number of positions within `shape`. If a `margin` is
        given, positions will be inside this margin. Margin may be tuple-valued.
        """
        margin = tp.artificial.validate_tuple(margin, len(shape))
        pos = [np.abs(s - m) * np.random.random_sample(count,) + np.min([m, s])
               for (s, m) in zip(shape, margin)]
        return np.array(pos).T

    def gen_nonoverlapping_locations(self, shape, count, separation, margin=0.0):
        """ Generates `count` number of positions within `shape`, that have minimum
        distance `separation` from each other. The number of positions returned may
        be lower than `count`, because positions too close to each other will be
        deleted. If a `margin` is given, positions will be inside this margin.
        Margin may be tuple-valued.
        """
        positions = self.gen_random_locations(shape, count, margin)
        return tp.artificial.eliminate_overlapping_locations(positions, separation)
