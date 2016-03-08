from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from scipy.ndimage.interpolation import map_coordinates

from numpy.testing import assert_allclose

from .algebraic import (ellipse_grid, ellipsoid_grid, fit_ellipse,
                        fit_ellipsoid)


def fit_max_2d(arr, maxfit_size=2, threshold=0.1):
    """ Finds the maxima along axis 0 using linear regression of the
    difference list.

    Parameters
    ----------
    arr : numpy array
    maxfit_size : integer, optional
        defines the fitregion around the maximum value:
        range(argmax - maxfit_size, argmax + maxfitsize + 1). Default 2.
    threshold :
        discard points when the mean of the fitregion < threshold * global max

    Returns
    -------
    1D array with locations of maxima.
    Elements are NaN in all of the following cases:
     - any pixel in the fitregion is 0
     - the mean of the fitregion < threshold * global max
     - regression returned infinity
     - maximum is outside of the fit region.
    """
    # identify the regions around the max value
    maxes = np.argmax(arr[:, maxfit_size:-maxfit_size],
                      axis=1) + maxfit_size
    ind = maxes[:, np.newaxis] + range(-maxfit_size, maxfit_size+1)
    fitregion = np.array([_int.take(_ind) for _int, _ind in zip(arr, ind)],
                         dtype=np.int32)

    # fit max using linear regression
    intdiff = np.diff(fitregion, 1)
    x_norm = np.arange(-maxfit_size + 0.5, maxfit_size + 0.5) # is normed because symmetric, x_mean = 0
    y_mean = np.mean(intdiff, axis=1, keepdims=True)
    y_norm = intdiff - y_mean
    slope = np.sum(x_norm[np.newaxis, :] * y_norm, 1) / np.sum(x_norm * x_norm)
    r_dev = - y_mean[:, 0] / slope

    # mask invalid fits
    threshold *= fitregion.max()  # relative to global maximum
    valid = (np.isfinite(r_dev) &   # finite result
             (fitregion > 0).all(1) &  # all pixels in fitregion > 0
             (fitregion.mean(1) > threshold) & # fitregion mean > threshold
             (r_dev > -maxfit_size + 0.5) &  # maximum inside fit region
             (r_dev < maxfit_size - 0.5))
    r_dev[~valid] = np.nan
    return r_dev + maxes


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


def refine_ellipse(image, params, mode='ellipse_aligned', n=None,
                   rad_range=None, maxfit_size=2, spline_order=3,
                   threshold=0.1):
    """ Interpolates the image along lines perpendicular to the ellipse. 
    The maximum along each line is found using linear regression of the
    descrete derivative.

    Parameters
    ----------
    image : 2d numpy array of numbers
        Image indices are interpreted as (y, x)
    params : yr, xr, yc, xc
    mode : {'ellipse', 'ellipse_aligned', 'circle'}
    n: integer
        number of points on the ellipse that are used for refine
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
    yr, xr, yc, xc

    """
    if not np.all([x > 0 for x in params]):
        raise ValueError("All yc, xc, yr, xr params should be positive")
    assert image.ndim == 2
    yr, xr, yc, xc = params
    if rad_range is None:
        rad_range = (-min(yr, xr) / 2, min(yr, xr) / 2)
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)
    pos, normal = ellipse_grid((yr, xr), (yc, xc), n=n, spacing=1)
    coords = normal[:, :, np.newaxis] * steps[np.newaxis, np.newaxis, :] + \
             pos[:, :, np.newaxis]

    # interpolate the image on calculated coordinates
    intensity = map_coordinates(image, coords, order=spline_order)

    # identify the regions around the max value
    r_dev = fit_max_2d(intensity, maxfit_size, threshold)

    # calculate new coords
    coord_new = pos + (r_dev + rad_range[0])*normal
    coord_new = coord_new[:, np.isfinite(coord_new).all(0)]

    # fit ellipse
    radius, center, _ = fit_ellipse(coord_new, mode=mode)
    return tuple(radius) + tuple(center), coord_new.T


def refine_ellipsoid_fast(image3d, p, n_xy=None, n_xz=None, rad_range=None,
                          maxfit_size=2, spline_order=3, threshold=0.1,
                          radius_rtol=0.5, radius_atol=30., center_atol=30.):
    """ Refines coordinates of a 3D ellipsoid, starting from given parameters.
    For fast analysis, it only analyzes YX and ZX middle slices.

    Parameters
    ----------
    image3d : 3D numpy array
    p: tuple of floats
        (zr, yr, xr, zc, yr, xr) coordinates of ellipsoid center
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
    (zr, yr, xr, zc, yc, xc), contour
    """
    assert image3d.ndim == 3
    zr0, yr0, xr0, zc0, yc0, xc0 = p
    # refine X, Y radius and center on XY middle
    middle_slice = image3d[int(zc0)] * (1 - zc0 % 1) + \
                   image3d[int(zc0) + 1] * (zc0 % 1)

    (yr, xr, yc, xc), r = refine_ellipse(middle_slice, (yr0, xr0, yc0, xc0),
                                         'ellipse_aligned', n_xy, rad_range,
                                         maxfit_size, spline_order, threshold)

    # refine Z radius and center on ZX middle (not ZY, is blurred by resonant)
    middle_slice = image3d[:, int(yc0)] * (1 - yc0 % 1) + \
                   image3d[:, int(yc0) + 1] * (yc0 % 1)

    (zr, _, zc, _), _ = refine_ellipse(middle_slice, (zr0, xr0, zc0, xc0),
                                       'ellipse_aligned', n_xz, rad_range,
                                       maxfit_size, spline_order, threshold)


    assert_allclose([xr, yr, zr],
                    [xr0, yr0, zr0], radius_rtol, radius_atol,
                    err_msg='Refined value differs extremely from initial value.')
                    
    assert_allclose([xc, yc, zc],
                    [xc0, yc0, zc0], rtol=0, atol=center_atol,
                    err_msg='Refined value differs extremely from initial value.')

    return (zr, yr, xr, zc, yc, xc), r


def refine_ellipsoid(image3d, params, spacing=1, rad_range=None, maxfit_size=2,
                     spline_order=3, threshold=0.1):
    """ Refines coordinates of a 3D ellipsoid, starting from given parameters.

    Interpolates the image along lines perpendicular to the ellipsoid.
    The maximum along each line is found using linear regression of the
    descrete derivative.

    Parameters
    ----------
    image3d : 3d numpy array of numbers
        Image indices are interpreted as (z, y, x)
    params : tuple
        zr, yr, xr, zc, yc, xc
    spacing: number
        spacing along radial direction
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
    - zr, yr, xr, zc, yc, xc, skew_y, skew_x
    - contour coordinates at z = 0

    """
    if not np.all([x > 0 for x in params]):
        raise ValueError("All zc, yc, xc, zr, yr, xr params should be positive")
    assert image3d.ndim == 3
    zr, yr, xr, zc, yc, xc = params
    if rad_range is None:
        rad_range = (-min(zr, yr, xr) / 2, min(zr, yr, xr) / 2)
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)
    pos, normal = ellipsoid_grid((zr, yr, xr), (zc, yc, xc), spacing=spacing)
    coords = normal[:, :, np.newaxis] * steps[np.newaxis, np.newaxis, :] + \
             pos[:, :, np.newaxis]

    # interpolate the image on calculated coordinates
    intensity = map_coordinates(image3d, coords, order=spline_order)

    # identify the regions around the max value
    r_dev = fit_max_2d(intensity, maxfit_size, threshold)

    # calculate new coords
    coord_new = pos + (r_dev + rad_range[0])*normal
    coord_new = coord_new[:, np.isfinite(coord_new).all(0)]

    # fit ellipsoid
    radius, center, skew = fit_ellipsoid(coord_new, mode='xy',
                                         return_mode='skew')
    return tuple(radius) + tuple(center) + tuple(skew), coord_new.T


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
