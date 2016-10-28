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
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
import pandas as pd
from scipy.spatial import cKDTree
from .algebraic import fit_ellipse
from trackpy.utils import validate_tuple


def find_disks(image, size_range, maximum=100, canny_sigma=1):
    """ Find circular edges in a 2D image using hough transforms.

    An edge is a sharp light-dark or dark-light transition. These are found
    using a canny edge filter. Subsequently, the edges undergo a circular
    Hough transformation for a range of circle radii. Peaks in the hough
    transformed image correspond to circle centers.

    Parameters
    ----------
    image : ndarray, 2d
    size_range : tuple of numbers
        the range of circle radii to look for, in pixels
    maximum : number, optional
        The maximum number of disks
    canny_sigma : number, optional
        The sigma value used in the Canny edge filter. Default 1.

    See also
    --------
    http://scikit-image.org/docs/dev/auto_examples/plot_canny.html
    http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    """
    # Define the radius at for which hough transforms are done. Take integer
    # values and a maximum of 30 intermediate steps.
    step = max(int(round(abs(size_range[1] - size_range[0]) / 30)), 1)
    radii = np.arange(size_range[0], size_range[1], step=step, dtype=np.intp)

    # Find edges in the image
    edges = canny(image, sigma=canny_sigma)
    # Perform a circular hough transform of the edges
    circles = hough_circle(edges, radii)

    # Collect the peaks in hough space, these are circle centers
    data = []
    for radius, circle in zip(radii, circles):
        peaks = peak_local_max(circle, threshold_rel=0.5,
                               num_peaks=int(maximum))
        try:
            accumulator = circle[peaks[:, 0], peaks[:, 1]]
        except TypeError:
            continue
        data.append(pd.DataFrame(dict(r=[radius] * peaks.shape[0],
                                      y=peaks[:, 0],
                                      x=peaks[:, 1],
                                      accum=accumulator)))
    if len(data) == 0:
        return pd.DataFrame(columns=['r', 'y', 'x', 'accum'])
    data = pd.concat(data, ignore_index=True)

    # drop features that are closer than the average radius together
    # keep the ones that are brightest in hough space (= the most circular ones)
    to_drop = where_close(data[['y', 'x']].values, data['r'].mean(),
                          intensity=data['accum'].values)
    data.drop(to_drop, inplace=True)

    # Keep only brightest n circles
    try:  # work around API change in pandas 0.17
        data = data.sort_values(by=['accum'], ascending=False)
    except AttributeError:
        data = data.sort(columns=['accum'], ascending=False)

    return data.head(maximum).copy()


def find_ellipse(image, mode='ellipse_aligned', min_length=24):
    """ Find bright ellipse contours on a black background.

    This routine thresholds the image (using the Otsu threshold), finds the
    longest contour and fits an ellipse to this contour.

    Parameters
    ----------
    image : ndarray, 2d
    mode : {'ellipse', 'ellipse_aligned', 'circle'}
        'ellipse' or None finds an arbitrary ellipse (default)
        'circle' finds a circle
        'ellipse_aligned' finds an ellipse with its axes aligned along [x y] axes
    min_length : number, optional
        minimum length of the ellipse contour, in pixels. Default 24.

    Returns
    -------
    yr, xr, yc, xc when dimension order was y, x (most common)
    xr, yr, xc, yc when dimension order was x, y
    """
    assert image.ndim == 2
    # Threshold the image
    thresh = threshold_otsu(image)
    binary = image > thresh

    # Find the contours of 0.5 value. For a thresholded ellipse contour, this
    # likely finds 2 contours: the inner and the outer.
    contours = find_contours(binary, 0.5, fully_connected='high')
    if len(contours) == 0:
        raise ValueError('No contours found')

    # Eliminate short contours
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
    """ Finds a bright ellipsoid contour on a black background in a 3D image.

    Finds the ellipses in all three projections of the 3D image and returns
    center coordinates and priciple radii.

    The function uses the YX projection for the yr, xr, yc, xc and the ZX
    projection for zr, xc. The ZY projection can be used for sanity checking
    the found center.

    Parameters
    ----------
    image3d : ndarray, 3d
    center_atol : float, optional
        the maximum absolute tolerance for the difference between the found
        centers in different projections. Default None

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


def where_close(pos, separation, intensity=None):
    """ Returns indices of features that are closer than separation from other
    features. When intensity is given, the one with the lowest intensity is
    returned: else the most topleft is returned (to avoid randomness)

    To be implemented in trackpy v0.4"""
    if len(pos) == 0:
        return []
    separation = validate_tuple(separation, pos.shape[1])
    if any([s == 0 for s in separation]):
        return []
    # Rescale positions, so that pairs are identified below a distance
    # of 1.
    pos_rescaled = pos / separation
    duplicates = cKDTree(pos_rescaled, 30).query_pairs(1 - 1e-7)
    if len(duplicates) == 0:
        return []
    index_0 = np.fromiter((x[0] for x in duplicates), dtype=int)
    index_1 = np.fromiter((x[1] for x in duplicates), dtype=int)
    if intensity is None:
        to_drop = np.where(np.sum(pos_rescaled[index_0], 1) >
                           np.sum(pos_rescaled[index_1], 1),
                           index_1, index_0)
    else:
        intensity_0 = intensity[index_0]
        intensity_1 = intensity[index_1]
        to_drop = np.where(intensity_0 > intensity_1, index_1, index_0)
        edge_cases = intensity_0 == intensity_1
        if np.any(edge_cases):
            index_0 = index_0[edge_cases]
            index_1 = index_1[edge_cases]
            to_drop[edge_cases] = np.where(np.sum(pos_rescaled[index_0], 1) >
                                           np.sum(pos_rescaled[index_1], 1),
                                           index_1, index_0)
    return np.unique(to_drop)

