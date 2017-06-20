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

from .algebraic import (ellipse_grid, ellipsoid_grid, fit_ellipse,
                        fit_ellipsoid, max_linregress, max_edge)
from .masks import slice_image, get_mask


def unwrap_ellipse(image, params, rad_range, num_points=None, spline_order=3):
    """ Unwraps an circular or ellipse-shaped feature into elliptic coordinates.

    Transforms an image in (y, x) space to (theta, r) space, using elliptic
    coordinates. The theta coordinate is tangential to the ellipse, the r
    coordinate is normal to the ellipse. r=0 at the ellipse: inside the ellipse,
    r < 0.

    Parameters
    ----------
    image : ndarray, 2d
    params : (yr, xr, yc, xc)
    rad_range : tuple
        A tuple defining the range of r to interpolate.
    num_points : number, optional
        The number of ``theta`` values. By default, this equals the
        ellipse circumference: approx. every pixel there is an interpolation.
    spline_order : number, optional
        The order of the spline interpolation. Default 3.

    Returns
    -------
    intensity : the interpolated image in (theta, r) space
    pos : the (y, x) positions of the ellipse grid
    normal : the (y, x) unit vectors normal to the ellipse grid
    """
    yr, xr, yc, xc = params
    # compute the r coordinates
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)
    # compute the (y, x) positions and unit normals of the ellipse
    pos, normal = ellipse_grid((yr, xr), (yc, xc), n=num_points, spacing=1)
    # calculate all the (y, x) coordinates on which the image interpolated.
    # this is a 3D array of shape [n_theta, n_r, 2], with 2 being y and x.
    coords = normal[:, :, np.newaxis] * steps[np.newaxis, np.newaxis, :] + \
        pos[:, :, np.newaxis]
    # interpolate the image on computed coordinates
    intensity = map_coordinates(image, coords, order=spline_order,
                                output=np.float)
    return intensity, pos, normal


def to_cartesian(r_dev, pos, normal):
    """ Transform radial deviations from an ellipsoidal grid to Cartesian

    Parameters
    ----------
    r_dev : ndarray, shape (N, )
        Array containing the N radial deviations from the ellipse. r < 0 means
        inside the ellipse.
    pos : ndarray, shape (2, N)
        The N (y, x) positions of the ellipse (as given by ``ellipse_grid``)
    normal : ndarray, shape (2, N)
        The N (y, x) unit normals of the ellipse (as given by ``ellipse_grid``)
    """
    coord_new = pos + r_dev * normal
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
    intensity, pos, normal = unwrap_ellipse(image, params, rad_range, n)

    # identify the regions around the max value
    r_dev = max_linregress(intensity, maxfit_size, threshold) + rad_range[0]
    coord_new = to_cartesian(r_dev, pos, normal)

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
    r_dev = max_linregress(intensity, maxfit_size, threshold)

    # calculate new coords
    coord_new = pos + (r_dev + rad_range[0])*normal
    coord_new = coord_new[:, np.isfinite(coord_new).all(0)]

    # fit ellipsoid
    radius, center, skew = fit_ellipsoid(coord_new, mode='xy',
                                         return_mode='skew')
    return tuple(radius) + tuple(center) + tuple(skew), coord_new.T


def refine_disks(image, blobs, rad_range=None, threshold=0.5, max_dev=1):
    """ Refine the position and size of multiple bright disks in an image """
    result = blobs.copy()
    if 'accum' in result:
        del result['accum']
    result['mass'] = np.nan
    result['signal'] = np.nan
    for i in result.index:
        fit, _ = _refine_disks(image, blobs.loc[i], rad_range, threshold,
                               max_dev)
        if fit is None:
            result.loc[i, ['r', 'y', 'x']] = np.nan
            result.loc[i, ['r', 'y', 'x']] = np.nan
            continue

        r, _, yc, xc = fit
        result.loc[i, ['r', 'y', 'x']] = r, yc, xc
        coords = np.array([(yc, xc)])
        square, origin = slice_image(coords, image, r+1)
        if origin is None:  # outside of image
            continue
        mask = get_mask(coords - origin, square.shape, r)
        result.loc[i, 'mass'] = square[mask].sum()
        result.loc[i, 'signal'] = result.loc[i, 'mass'] / mask.sum()

    return result


def _refine_disks(image, params, rad_range=None, threshold=0.5, max_dev=1):
    if rad_range is None:
        rad_range = (-params.r / 2, params.r / 2)

    # Get intensity in spline representation
    coords = (params.r, params.r, params.y, params.x)
    intensity, pos, normal = unwrap_ellipse(image, coords, rad_range)

    # Check whether the intensity interpolation is bright on left, dark on right
    if np.mean(intensity[:, 0]) < 2 * np.mean(intensity[:, -1]):
        return None, None

    # Find the coordinates of the edge
    r_dev = max_edge(intensity, threshold) + rad_range[0]
    if np.sum(np.isnan(r_dev)) / len(r_dev) > 0.5:
        return None, None

    # Set outliers to mean of rest of x coords
    # r_dev = remove_outliers(r_dev)

    # Convert to cartesian
    coord_new = to_cartesian(r_dev, pos, normal)

    # Fit the circle
    try:
        (radius, _), (yc, xc), _ = fit_ellipse(coord_new, mode='xy')
    except np.linalg.LinAlgError:
        return None, None
    if np.any(np.isnan([radius, yc, xc])):
        return None, None
    if not rad_range[0] < radius - params.r < rad_range[1]:
        return None, None

    # calculate deviations from circle
    y, x = coord_new
    deviations2 = (np.sqrt((xc - x)**2 + (yc - y)**2) - radius)**2
    mask = deviations2 < max_dev**2
    if np.sum(mask) / len(mask) < 0.5:
        return None, None

    if np.any(~mask):
        try:
            (radius, _), (yc, xc), _ = fit_ellipse(coord_new[:, mask],
                                                   mode='xy')
        except np.linalg.LinAlgError:
            return None, None
        if np.any(np.isnan([radius, yc, xc])):
            return None, None

    return (radius, radius, yc, xc), coord_new.T
