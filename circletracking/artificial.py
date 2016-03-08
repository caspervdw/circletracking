from __future__ import (division, unicode_literals)

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.ndimage.interpolation import zoom, shift
from scipy.ndimage.measurements import center_of_mass


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
        image_temp = np.pad(image, no_padding + padding, mode='constant')
    else:
        image_temp = image

    no_crop = [slice(o+1) for o in image.shape[:-ndim]]
    crop = [slice(c, c+s) for (c, s) in zip(corner, shape)]
    return image_temp[no_crop + crop]


def gen_artificial_ellipsoid(shape, radius, center, FWHM, noise=0):
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
