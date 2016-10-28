""" Locate features in images: combine find and refine steps """
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import pandas as pd
from .find import find_ellipse, find_ellipsoid, find_disks
from .refine import (refine_ellipse, refine_ellipsoid,
                     refine_ellipsoid_fast, refine_disks)


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
    params = find_ellipse(frame, mode)
    params, r = refine_ellipse(frame, params, mode, n, rad_range,
                               maxfit_size, spline_order, threshold)
        # params = [np.nan] * 4
        # r = None

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
    params = find_ellipsoid(frame)
    params, r = refine_ellipsoid_fast(frame, params, n_xy, n_xz, rad_range,
                                      maxfit_size, spline_order, threshold,
                                      radius_rtol, radius_atol, center_atol)
    # except Exception:
    #     params = [np.nan] * 6
    #     r = None

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
    params = find_ellipsoid(frame)
    params, r = refine_ellipsoid(frame, params, spacing, rad_range,
                                 maxfit_size, spline_order, threshold)
    r = r[np.abs(r[:, 0] - params[3]) < 0.5]  # extract center coords
    # except Exception:
    #     params = [np.nan] * 8
    #     r = None

    return pd.Series(params, index=columns), r


def locate_disks(image, size_range, maximum=100, rad_range=None,
                 threshold=0.5, max_dev=1, canny_sigma=1):
    """ Find circular particles in the image """
    blobs = find_disks(image, size_range, maximum, canny_sigma)

    if blobs.empty:
        return pd.DataFrame(columns=['r', 'y', 'x', 'dev'])

    result = refine_disks(image, blobs, rad_range, threshold, max_dev)
    result = result.dropna()
    result.reset_index(drop=True, inplace=True)
    return result
