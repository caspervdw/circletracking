""" Find features in image """
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from numpy.testing import assert_allclose
import skimage
try:
    from skimage.filters import threshold_otsu
except ImportError:
    from skimage.filter import threshold_otsu  # skimage <= 0.10
try:
    from skimage.feature import canny
except ImportError:
    from skimage.filter import canny  # skimage <= 0.10
from skimage.measure import find_contours
import pandas
import scipy

from .algebraic import fit_ellipse

def find_disks(image, size_range, number_of_disks=100):
    """ Locate blobs in the image by using a Laplacian of Gaussian method """
    number_of_disks = int(np.round(number_of_disks))
    radii = np.linspace(size_range[0], size_range[1],
                        num=min(abs(size_range[0] - size_range[1]) * 2.0, 30))
    radii = radii.astype(np.float)

    # Find edges
    edges = canny(image)
    circles = skimage.transform.hough_circle(edges, radii)

    fit = pandas.DataFrame(columns=['r', 'y', 'x', 'accum'])
    for radius, hough_circle in zip(radii, circles):
        peaks = skimage.feature.peak_local_max(hough_circle, threshold_rel=0.5,
                                               num_peaks=number_of_disks)
        accumulator = hough_circle[peaks[:, 0], peaks[:, 1]]
        fit = pandas.concat([fit, pandas.DataFrame(data={'r': [radius] * peaks.shape[0], 'y': peaks[:, 0], 'x': peaks[:, 1], 'accum': accumulator})], ignore_index=True)

    fit = merge_hough_same_values(fit, number_of_disks)

    return fit

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


def merge_hough_same_values(data, number_to_keep=100):
    while True:
        # Rescale positions, so that pairs are identified below a distance
        # of 1. Do so every iteration (room for improvement?)
        positions = data[['x', 'y']].values
        mass = data['accum'].values
        duplicates = scipy.spatial.cKDTree(positions, 30).query_pairs(
            np.mean(data['r']), p=2.0, eps=0.1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            # Drop the dimmer one.
            if np.equal(*mass.take(pair, 0)):
                # Rare corner case: a tie!
                # Break ties by sorting by sum of coordinates, to avoid
                # any randomness resulting from cKDTree returning a set.
                dimmer = np.argsort(np.sum(positions.take(pair, 0), 1))[0]
            else:
                dimmer = np.argmin(mass.take(pair, 0))
            to_drop.append(pair[dimmer])
        data.drop(to_drop, inplace=True)

    # Keep only brightest n circles
    data = data.sort_values(by=['accum'], ascending=False)
    data = data.head(number_to_keep)

    return data
