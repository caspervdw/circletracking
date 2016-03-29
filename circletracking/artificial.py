from __future__ import (division, unicode_literals)

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.ndimage.interpolation import zoom, shift
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import cKDTree


def crop_pad(image, corner, shape):
    ndim = len(corner)
    corner = [int(round(c)) for c in corner]
    shape = [int(round(s)) for s in shape]
    original = image.shape[-ndim:]
    zipped = zip(corner, shape, original)

    if np.any(c < 0 or c + s > o for (c, s, o) in zipped):
        no_padding = [(0, 0)] * (image.ndim - ndim)
        padding = [(max(-c, 0), max(c + s - o, 0)) for (c, s, o) in zipped]
        corner = [c + max(-c, 0) for c in corner]
        image_temp = np.pad(image, no_padding + padding, mode=str('constant'))
    else:
        image_temp = image

    no_crop = [slice(o+1) for o in image.shape[:-ndim]]
    crop = [slice(c, c+s) for (c, s) in zip(corner, shape)]
    return image_temp[no_crop + crop]


def draw_ellipse(shape, radius, center, FWHM, noise=0):
    sigma = FWHM / 2.35482
    cutoff = 2 * FWHM

    # draw a circle
    R = max(radius)
    zoom_factor = np.array(radius) / R
    size = int((R + cutoff)*2)
    c = size // 2
    y, x = np.meshgrid(*([np.arange(size)] * 2), indexing='ij')
    h = np.sqrt((y - c)**2+(x - c)**2) - R
    mask = np.abs(h) < cutoff
    im = np.zeros((size,)*2, dtype=np.float)
    im[mask] += np.exp((h[mask] / sigma)**2/-2)/(sigma*np.sqrt(2*np.pi))

    # zoom so that radii are ok
    im = zoom(im, zoom_factor)

    # shift and make correct shape
    center_diff = center - np.array(center_of_mass(im))
    left_padding = np.round(center_diff).astype(np.int)
    subpx_shift = center_diff - left_padding

    im = shift(im, subpx_shift)
    im = crop_pad(im, -left_padding, shape)
    im[im < 0] = 0

    assert_almost_equal(center_of_mass(im), center, decimal=2)

    if noise > 0:
        im += np.random.random(shape) * noise * im.max()

    return (im / im.max() * 255).astype(np.uint8)


def draw_ellipsoid(shape, radius, center, FWHM, noise=0):
    sigma = FWHM / 2.35482
    cutoff = 2 * FWHM

    # draw a sphere
    R = max(radius)
    zoom_factor = np.array(radius) / R
    size = int((R + cutoff)*2)
    c = size // 2
    z, y, x = np.meshgrid(*([np.arange(size)] * 3), indexing='ij')
    h = np.sqrt((z - c)**2+(y - c)**2+(x - c)**2) - R
    mask = np.abs(h) < cutoff
    im = np.zeros((size,)*3, dtype=np.float)
    im[mask] += np.exp((h[mask] / sigma)**2/-2)/(sigma*np.sqrt(2*np.pi))

    # zoom so that radii are ok
    im = zoom(im, zoom_factor)

    # shift and make correct shape
    center_diff = center - np.array(center_of_mass(im))
    left_padding = np.round(center_diff).astype(np.int)
    subpx_shift = center_diff - left_padding

    im = shift(im, subpx_shift)
    im = crop_pad(im, -left_padding, shape)
    im[im < 0] = 0

    assert_almost_equal(center_of_mass(im), center, decimal=2)

    if noise > 0:
        im += np.random.random(shape) * noise * im.max()

    return (im / im.max() * 255).astype(np.uint8)

def feat_step(r):
    """ Solid disc. """
    return r <= 1


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
                 feat_func=feat_step, **feat_kwargs):
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = np.float64
        self.image = np.zeros(shape, dtype=self.dtype)
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
        """ Normalize image """
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
        return np.array(noisy_image, dtype=self.dtype)

    @property
    def coords(self):
        if len(self._coords) == 0:
            return np.zeros((0, self.ndim), dtype=self.dtype)
        return np.array(self._coords)

    def draw_feature(self, pos):
        """Draws a feature at `pos`."""
        pos = [float(p) for p in pos]
        self._coords.append(pos)
        draw_feature(image=self.image, position=pos, diameter=2.0 * self.radius,
                     max_value=1.0, feat_func=self.feat_func,
                     **self.feat_kwargs)

    def draw_features(self, count, separation=0, margin=None):
        """Draws N features at random locations, using minimum separation
        and a margin. If separation > 0, less than N features may be drawn."""
        if margin is None:
            margin = float(self.radius)
        pos = self.gen_nonoverlapping_locations(self.shape, count, separation,
                                                margin)
        for p in pos:
            self.draw_feature(p)
        return pos

    @staticmethod
    def gen_random_locations(shape, count, margin=0.0):
        """ Generates `count` number of positions within `shape`. If a `margin` is
        given, positions will be inside this margin. Margin may be tuple-valued.
        """
        margin = validate_tuple(margin, len(shape))
        pos = [np.random.uniform(m, s - m, count)
               for (s, m) in zip(shape, margin)]
        return np.array(pos).T

    def gen_nonoverlapping_locations(self, shape, count, separation,
                                     margin=0.0):
        """ Generates `count` number of positions within `shape`, that have minimum
        distance `separation` from each other. The number of positions returned may
        be lower than `count`, because positions too close to each other will be
        deleted. If a `margin` is given, positions will be inside this margin.
        Margin may be tuple-valued.
        """
        positions = self.gen_random_locations(shape, count, margin)
        if len(positions) > 1:
            return eliminate_overlapping_locations(positions, separation)
        else:
            return positions


def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def draw_feature(image, position, diameter, max_value=None,
                 feat_func=feat_step, ecc=None, **kwargs):
    """ Draws a radial symmetric feature and adds it to the image at given
    position. The given function will be evaluated at each pixel coordinate,
    no averaging or convolution is done.

    Parameters
    ----------
    image : ndarray
        image to draw features on
    position : iterable
        coordinates of feature position
    diameter : number
        defines the box that will be drawn on
    max_value : number
        maximum feature value. should be much less than the max value of the
        image dtype, to avoid pixel wrapping at overlapping features
    feat_func : function. Default: feat_gauss
        function f(r) that takes an ndarray of radius values
        and returns intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    kwargs : keyword arguments are passed to feat_func
    """
    if len(position) != image.ndim:
        raise ValueError("Number of position coordinates should match image"
                         " dimensionality.")
    diameter = validate_tuple(diameter, image.ndim)
    if ecc is not None:
        if len(diameter) != 2:
            raise ValueError("Eccentricity is only defined in 2 dimensions")
        if diameter[0] != diameter[1]:
            raise ValueError("Diameter is already anisotropic; eccentricity is"
                             " not defined.")
        diameter = (diameter[0] / (1 - ecc), diameter[1] * (1 - ecc))
    radius = tuple([d / 2 for d in diameter])
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    rect = []
    vectors = []
    for (c, r, lim) in zip(position, radius, image.shape):
        if (c >= lim) or (c < 0):
            raise ValueError("Position outside of image.")
        lower_bound = max(int(np.floor(c - r)), 0)
        upper_bound = min(int(np.ceil(c + r + 1)), lim)
        rect.append(slice(lower_bound, upper_bound))
        vectors.append(np.arange(lower_bound - c, upper_bound - c) / r)
    coords = np.meshgrid(*vectors, indexing='ij', sparse=True)
    r = np.sqrt(np.sum(np.array(coords)**2, axis=0))
    spot = max_value * feat_func(r, **kwargs)
    image[rect] += spot.astype(image.dtype)


def gen_random_locations(shape, count, margin=0):
    """ Generates `count` number of positions within `shape`. If a `margin` is
    given, positions will be inside this margin. Margin may be tuple-valued.
    """
    margin = validate_tuple(margin, len(shape))
    np.random.seed(0)
    pos = [np.random.randint(round(m), round(s - m), count)
           for (s, m) in zip(shape, margin)]
    return np.array(pos).T


def eliminate_overlapping_locations(f, separation):
    """ Makes sure that no position is within `separation` from each other, by
    deleting one of the that are to close to each other.
    """
    separation = validate_tuple(separation, f.shape[1])
    assert np.greater(separation, 0).all()
    # Rescale positions, so that pairs are identified below a distance of 1.
    f = f / separation
    while True:
        duplicates = cKDTree(f, 30).query_pairs(1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            to_drop.append(pair[1])
        f = np.delete(f, to_drop, 0)
    return f * separation
