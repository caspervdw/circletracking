from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import skimage
import scipy.spatial
import pandas as pd
from .find import find_ellipse, find_ellipsoid
from .refine import (refine_ellipse, refine_ellipsoid, refine_ellipsoid_fast)


def locate_ellipse(frame, mode='ellipse_aligned', n=None, rad_range=None,
                   maxfit_size=2, spline_order=3, threshold=0.1):
    """Locates an ellipse in a 2D image and returns center coordinates and
    radii along x, y.

    Parameters
    ----------
    frame: ndarray
    n : int
    number of points on the ellipse that are used for refine
    spacing: float
    spacing between points on an xy circle, for grid
    rad_range: tuple of floats
    length of the line (distance inwards, distance outwards)
    maxfit_size: integer
    pixels around maximum pixel that will be used in linear regression
    spline_order: integer
    interpolation order for edge crossections
    threshold: float
    a threshold is calculated based on the global maximum
    fitregions are rejected if their average value is lower than this

    Returns
    -------
    Series with yr, xr, yc, xc indices
    ndarray with (y, x) contour
    """
    assert frame.ndim == 2
    columns = ['yr', 'xr', 'yc', 'xc']
    try:
        params = find_ellipse(frame, mode)
        params, r = refine_ellipse(frame, params, mode, n, rad_range,
                                   maxfit_size, spline_order, threshold)
    except Exception:
        params = [np.nan] * 4
        r = None

    return pd.Series(params, index=columns), r


def locate_ellipsoid_fast(frame, n_xy=None, n_xz=None, rad_range=None,
                          maxfit_size=2, spline_order=3, threshold=0.1,
                          radius_rtol=0.5, radius_atol=30.0, center_atol=30.0):
    """Locates an ellipsoid in a 3D image and returns center coordinates and
    radii along x, y, z. The function only analyzes YX and ZX middle slices.

    Parameters
    ----------
    image3d: 3D ndarray
    n_xy: integer
    number of points on the ellipse that are used for refine in xy plane
    n_xz: integer
    number of points on the ellipse that are used for refine in xz plane
    rad_range: tuple of floats
    length of the line (distance inwards, distance outwards)
    maxfit_size: integer
    pixels around maximum pixel that will be used in linear regression
    spline_order: integer
    interpolation order for edge crossections
    threshold: float
    a threshold is calculated based on the global maximum
    fitregions are rejected if their average value is lower than this
    radius_rtol : float, optional
    the maximum relative tolerance for the difference between initial
    and refined radii, Default 0.5
    radius_atol : float, optional
    the maximum absolute tolerance for the difference between initial
    and refined radii, Default 30.
    center_atol : float, optional
    the maximum absolute tolerance for the difference between initial
    and refined radii, Default 30.

    Returns
    -------
    Series with zr, yr, xr, zc, yc, xc indices, ndarray with (y, x) contour
    """
    assert frame.ndim == 3
    columns = ['zr', 'yr', 'xr', 'zc', 'yc', 'xc']
    try:
        params = find_ellipsoid(frame)
        params, r = refine_ellipsoid_fast(frame, params, n_xy, n_xz, rad_range,
                                          maxfit_size, spline_order, threshold,
                                          radius_rtol, radius_atol, center_atol)
    except Exception:
        params = [np.nan] * 6
        r = None

    return pd.Series(params, index=columns), r


def locate_ellipsoid(frame, spacing=1, rad_range=None, maxfit_size=2,
                     spline_order=3, threshold=0.1):
    """Locates an ellipsoid in a 3D image and returns center coordinates and
    radii along x, y, z. The function fully analyzes the vesicle.

    Parameters
    ----------
    image3d: 3D ndarray
    spacing: float
    spacing between points on an xy circle, for grid
    rad_range: tuple of floats
    length of the line (distance inwards, distance outwards)
    maxfit_size: integer
    pixels around maximum pixel that will be used in linear regression
    spline_order: integer
    interpolation order for edge crossections
    threshold: float
    a threshold is calculated based on the global maximum
    fitregions are rejected if their average value is lower than this

    Returns
    -------
    Series with zr, yr, xr, zc, yc, xc, skew_y, skew_x indices
    ndarray with (y, x) contour
    """
    assert frame.ndim == 3
    columns = ['zr', 'yr', 'xr', 'zc', 'yc', 'xc', 'skew_y', 'skew_x']
    try:
        params = find_ellipsoid(frame)
        params, r = refine_ellipsoid(frame, params, spacing, rad_range,
                                     maxfit_size, spline_order, threshold)
        r = r[np.abs(r[:, 0] - params[3]) < 0.5]  # extract center coords
    except Exception:
        params = [np.nan] * 8
        r = None

    return pd.Series(params, index=columns), r


def locate_multiple_disks(image, size_range, number_of_disks=100):
    """
    Locate blobs in the image by using a Laplacian of Gaussian method
    :rtype : pd.DataFrame
    :return:
    """
    number_of_disks = int(np.round(number_of_disks))
    radii = np.linspace(size_range[0], size_range[1],
                        num=min(abs(size_range[0] - size_range[1]) * 2.0, 30),
                        dtype=np.float)

    # Find edges
    edges = skimage.feature.canny(image)
    circles = skimage.transform.hough_circle(edges, radii)

    fit = pd.DataFrame(columns=['r', 'y', 'x', 'accum'])
    for radius, hough_circle in zip(radii, circles):
        peaks = skimage.feature.peak_local_max(hough_circle, threshold_rel=0.5,
                                               num_peaks=number_of_disks)
        accumulator = hough_circle[peaks[:, 0], peaks[:, 1]]
        fit = pd.concat([fit,
                         pd.DataFrame(data={'r': [radius] * peaks.shape[0],
                                            'y': peaks[:, 0],
                                            'x': peaks[:, 1],
                                            'accum': accumulator})
                        ], ignore_index=True)

        fit = merge_hough_same_values(fit, number_of_disks)

        return fit

def merge_hough_same_values(data, number_to_keep=100):
    """

    :param data:
    :return:
    """
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
