from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
try:
    from skimage.filters import threshold_otsu
except ImportError:
    from skimage.filter import threshold_otsu  # skimage <= 0.10
from skimage.measure import find_contours
import pandas
from numpy.testing import assert_allclose

from .algebraic import fit_ellipse
from .locate import locate_multiple_disks
from .refine import refine_multiple


def find_disks(image, size_range, number_of_disks=100):
    """
    Find circular particles in the image
    :param size_range:
    :rtype : pandas.DataFrame
    :param n:
    :return:
    """
    blobs = locate_multiple_disks(image, size_range, number_of_disks)

    if blobs.empty:
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    return refine_multiple(image, blobs)

def find_ellipse(image, mode='ellipse_aligned', min_length=24):
    """ Thresholds the image, finds the longest contour and fits an ellipse
    to this contour.

    Parameters
    ----------
    image : 2D numpy array of numbers
    mode : {'ellipse', 'ellipse_aligned', 'circle'}
    min_length : number
    minimum length of contour

    Returns
    -------
    yr, xr, yc, xc when dimension order was y, x (common)
    xr, yr, xc, yc when dimension order was x, y
    """
    assert image.ndim == 2
    thresh = threshold_otsu(image)
    binary = image > thresh
    contours = find_contours(binary, 0.5, fully_connected='high')
    if len(contours) == 0:
        raise ValueError('No contours found')

    # eliminate short contours
    contours = [c for c in contours if len(c) >= min_length]

    # fit circles to the rest, keep the one with lowest residual deviation
    result = [np.nan] * 4
    residual = None
    for c in contours:
        try:
            (xr, yr), (xc, yc), _ = fit_ellipse(c.T, mode=mode)
            if np.any(np.isnan([xr, yr, xc, yc])):
                continue
            x, y = c.T
            r = np.sum((((xc - x)/xr)**2 + ((yc - y)/yr)**2 - 1)**2)/len(c)
            if residual is None or r < residual:
                result = xr, yr, xc, yc
                residual = r
        except np.linalg.LinAlgError:
            pass

    return result


def find_ellipsoid(image3d, center_atol=None):
    """ Finds ellipses in all three projections of the 3D image and returns
    center coordinates and priciple radii.

    The function uses the YX projection for the yr, xr, yc, xc and the ZX
    projection for zr, xc. The ZY projection can be used for sanity checking
    the found center.

    Parameters
    ----------
    image3d : 3D numpy array of numbers
    center_atol : float, optional
    the maximum absolute tolerance for the difference between the found
    centers, Default None

    Returns
    -------
    zr, yr, xr, zc, yc, xc
    """
    assert image3d.ndim == 3

    # Y, X projection, use y radius because resonant scanning in x direction.
    image = np.mean(image3d, axis=0)
    yr, xr, yc, xc = find_ellipse(image, mode='ellipse_aligned')

    # Z, X projection
    image = np.mean(image3d, axis=1)
    zr, xr2, zc, xc2 = find_ellipse(image, mode='ellipse_aligned')

    if center_atol is not None:
        # Z, Y projection (noisy with resonant scanning)
        image = np.mean(image3d, axis=2)
        zr2, yr2, zc2, yc2 = find_ellipse(image, mode='ellipse_aligned')

        assert_allclose([xc, yc, zc],
                        [xc2, yc2, zc2], rtol=0, atol=center_atol,
                        err_msg='Found centers have inconsistent values.')

    return zr, yr, xr, zc, yc, xc
