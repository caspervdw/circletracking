"""Refinement steps for refining the 'crude' fits """
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from numpy.testing import assert_allclose
from scipy.ndimage.interpolation import map_coordinates
import scipy.ndimage
try:
    from skimage.filters import threshold_otsu
except ImportError:
    from skimage.filter import threshold_otsu  # skimage <= 0.10
import pandas as pd

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
    # is normed because symmetric, x_mean = 0
    x_norm = np.arange(-maxfit_size + 0.5, maxfit_size + 0.5)
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


def refine_disks(image, blobs, num_points_circle=100):
    """ Refine multiple Hough detected blobs """
    if blobs is None:
        return pd.DataFrame(colums=['r', 'y', 'x'])

    result = blobs.copy()
    result.drop('accum', axis=1, inplace=True)
    result['refined'] = True
    for i in result.index:
        fit = fit_edge_2d(image, blobs.loc[i],
                          num_points_circle=num_points_circle)
        # This is a refinement function, only refine if successfull,
        # otherwise use the original fit
        if fit is not None:
            result.loc[i, ['r', 'y', 'x']] = fit[0], fit[2], fit[3]
        else:
            result.loc[i, ['refined']] = False

    return result


def fit_edge_2d(image, params, threshold=None, num_points_circle=100):
    """ Find a circle based on the blob """
    rad_range = (-params.r, params.r)

    # Get intensity in spline representation
    coords = (params.r, params.r, params.y, params.x)
    intensity, pos, normal = get_intensity_interpolation(image, coords,
                                                         rad_range,
                                                         num_points_circle)

    if not check_intensity_interpolation(intensity, threshold):
        return None

    # Find the coordinates of the edge
    edge_positions = find_edge(intensity)

    # Convert to cartesian
    coord_new = mapped_coords_to_normal_coords(pos, edge_positions,
                                               rad_range, normal)

    # Remove coords with NaN values
    coord_new = coord_new[~np.isnan(coord_new).any(axis=1)]

    # Fit the circle
    radius, center, _ = fit_ellipse(coord_new, mode='xy')

    return tuple(radius) + tuple(center)


def find_edge(intensity):
    """ Find the edge of the particle """
    mask = create_binary_mask(intensity)

    # Take first falling edge where np.diff == -1
    xcoords = np.asarray(np.argmin(np.diff(np.asarray(mask, dtype=np.float),
                                           axis=1), axis=1), dtype=np.float)

    # 0.0 means edge at left or edge not found, replace with NaN
    xcoords[xcoords == 0.0] = np.nan

    # +0.5 to get the correct pixel position
    xcoords = xcoords + 0.5

    return xcoords


def create_binary_mask(intensity, threshold=None):
    if threshold is None:
        threshold = threshold_otsu(intensity)
    mask = intensity > threshold

    # Fill holes in binary mask
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)

    return mask


def check_intensity_interpolation(intensity, threshold=None):
    """
    Check whether the intensity interpolation is bright on left,
    dark on right
    """
    binary_mask = create_binary_mask(intensity, threshold)
    parts = np.array_split(binary_mask, 2, axis=1)
    mean_left = np.mean(parts[0])
    mean_right = np.mean(parts[1])
    return mean_left > mean_right


def get_intensity_interpolation(image, params, rad_range, num_points,
                                spline_order=3):
    """ Generate the intensity interpolation used for edge finding """
    yr, xr, yc, xc = params
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)
    pos, normal = ellipse_grid((yr, xr), (yc, xc), n=num_points, spacing=1)
    coords = normal[:, :, np.newaxis] * steps[np.newaxis, np.newaxis, :] + \
        pos[:, :, np.newaxis]

    # interpolate the image on calculated coordinates
    intensity = map_coordinates(image, coords, order=spline_order)
    return intensity, pos, normal


def mapped_coords_to_normal_coords(pos, r_dev, rad_range, normal):
    """ Generate the intensity interpolation used for edge finding """
    # calculate new coords
    coord_new = pos + (r_dev + rad_range[0]) * normal
    coord_new = coord_new[:, np.isfinite(coord_new).all(0)]
    return coord_new


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

    # interpolate the image on calculated coordinates
    intensity, pos, normal = get_intensity_interpolation(image, params,
                                                         rad_range, n)

    # identify the regions around the max value
    r_dev = fit_max_2d(intensity, maxfit_size, threshold)
    coord_new = mapped_coords_to_normal_coords(pos, r_dev, rad_range, normal)

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
