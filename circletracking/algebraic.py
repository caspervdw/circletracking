""" Functions for algebraic fitting """
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np

MODE_DICT_ELLIPSE = {'circle': 'xy', 'ellipse_aligned': '0', 'ellipse': ''}
MODE_DICT_ELLIPSOID = {'sphere': 'xyz', 'prolate': 'xy', 'oblate': 'xy',
                       'ellipsoid': '', 'ellipsoid_aligned': '0',
                       'prolate_aligned': '0xy', 'oblate_aligned': '0xy'}


def fit_ellipse(coords, mode=''):
    """ Fits an ellipse to datapoints using an algebraic method.

    This method is different from least squares minimization of the distnace
    between each datapoint and the fitted ellipse (so-called geometrical
    approach). For circles, this makes no difference. The higher the aspect
    ratio of the ellipse, the worse the approximation gets.

    Parameters
    ----------
    coords : numpy array of floats
        array of shape (N, 2) containing datapoints
    mode : {'circle', 'ellipse', 'ellipse_aligned'}
        'ellipse' or None fits an arbitrary ellipse (default)
        'circle' fits a circle
        'ellipse_aligned' fits an ellipse with its axes aligned along [x y] axes

    Returns
    -------
    center, radii, angle

    References
    ----------
    .. [1] Bertoni B (2010) Multi-dimensional ellipsoidal fitting.
    """
    if coords.shape[0] != 2:
        raise ValueError('Input data must have two columns!')
    if mode in MODE_DICT_ELLIPSE:
        mode = MODE_DICT_ELLIPSE[mode]

    x = coords[0, :, np.newaxis]
    y = coords[1, :, np.newaxis]

    if mode == '':
        D = np.hstack((x**2 - y**2, 2*x*y, 2*x, 2*y, np.ones_like(x)))
    elif mode == '0':
        D = np.hstack((x**2 - y**2, 2*x, 2*y, np.ones_like(x)))
    elif mode == 'xy':
        D = np.hstack((2*x, 2*y, np.ones_like(x)))

    d2 = x**2 + y**2  # the RHS of the llsq problem (y's)
    u = np.linalg.solve(np.dot(D.T, D), (np.dot(D.T, d2)))[:, 0]
    v = np.empty((6), dtype=u.dtype)

    if mode == '':
        v[0] = u[0] - 1
        v[1] = -u[0] - 1
        v[2:] = u[1:]
    elif mode == '0':
        v[0] = u[0] - 1
        v[1] = -u[0] - 1
        v[2] = 0
        v[3:] = u[1:]
    elif mode == 'xy':
        v[:2] = -1
        v[2] = 0
        v[3:] = u

    A = np.array([[v[0], v[2], v[3]],
                  [v[2], v[1], v[4]],
                  [v[3], v[4], v[5]]])
    # find the center of the ellipse
    center = -np.linalg.solve(A[:2, :2], v[3:5])

    # translate to the center
    T = np.identity(3, dtype=A.dtype)
    T[2, :2] = center
    R = np.dot(np.dot(T, A), T.T)

    # solve the eigenproblem
    evals, evecs = np.linalg.eig(R[:2, :2] / -R[2, 2])
    radius = (np.sqrt(1 / np.abs(evals)) * np.sign(evals))

    if mode == '':
        new_order = np.argmax(np.abs(evecs), 1)
        radius = radius[new_order]
        evecs = evecs[:, new_order]
        r11, r12, r21, r22 = evecs.T.flat
        angle = np.arctan(-r12/r11)
    else:
        angle = 0

    return radius, center, angle


def fit_ellipsoid(coords, mode='', return_mode=''):
    """
    Fit an ellipsoid/sphere/paraboloid/hyperboloid to a set of xyz data points:

    Parameters
    ----------
    coords : ndarray
        Cartesian coordinates, 3 x n array
    mode : {'', 'xy', 'xz', 'xyz', '0', '0xy', '0xz'} t
        '' or None fits an arbitrary ellipsoid (default)
        'xy' fits a spheroid with x- and y- radii equal
        'xz' fits a spheroid with x- and z- radii equal
        'xyz' fits a sphere
        '0' fits an ellipsoid with its axes aligned along [x y z] axes
        '0xy' the same with x- and y- radii equal
        '0xz' the same with x- and z- radii equal
    return_mode : {'', 'euler', 'skew'}
        '' returns the directions of the radii as 3x3 array
        'euler' returns euler angles
        'skew' returns skew in xy

    Returns
    -------
    radius : ndarray
        ellipsoid radii [zr, yr, xr]
    center : ndarray
        ellipsoid center coordinates [zc, yc, xc]
    value :
        return_mode == '': the radii directions as columns of the 3x3 matrix
        return_mode == 'euler':
            euler angles, applied in x, y, z order [z, y, x]
            the y value is the angle with the z axis (tilt)
            the z value is the angle around the z axis (rotation)
            the x value is the 3rd rotation, should be around 0
        return_mode == 'skew':
            skew in y, x order

    Notes
    -----
    Author: Yury Petrov, Oculus VR Date: September, 2015
    ported to python by Casper van der Wel, December 2015
    added euler angles and skew by Casper van der Wel
    """
    if coords.shape[0] != 3:
        raise ValueError('Input data must have three columns!')
    if mode in MODE_DICT_ELLIPSOID:
        mode = MODE_DICT_ELLIPSOID[mode]
    if return_mode == 'skew' and 'xy' not in mode:
        raise ValueError('Cannot return skew when x, y radii are not equal')
    if return_mode == 'euler':
        raise ValueError('Euler mode is not implemented fully')
    z = coords[0, :, np.newaxis]
    y = coords[1, :, np.newaxis]
    x = coords[2, :, np.newaxis]

    # fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx +
    # 2Hy + 2Iz + J = 0 and A + B + C = 3 constraint removing one extra param
    if mode == '':
        D = np.hstack((x**2 + y**2 - 2 * z**2, x**2 + z**2 - 2 * y**2,
                       2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z,
                       np.ones_like(x)))
    elif mode == 'xy':
        D = np.hstack((x**2 + y**2 - 2 * z**2, 2 * x * y,  2 * x * z, 2 * y * z,
                       2 * x, 2 * y, 2 * z, np.ones_like(x)))
    elif mode == 'xz':
        D = np.hstack((x**2 + z**2 - 2 * y**2, 2 * x * y, 2 * x * z, 2 * y * z,
                       2 * x, 2 * y, 2 * z, np.ones_like(x)))

    # fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1
    elif mode == '0':
        D = np.hstack((x**2 + y**2 - 2 * z**2, x**2 + z**2 - 2 * y**2,
                       2 * x, 2 * y, 2 * z, np.ones_like(x)))

    # fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1,
    # where A = B or B = C or A = C
    elif mode == '0xy':
        D = np.hstack((x**2 + y**2 - 2 * z**2, 2 * x, 2 * y, 2 * z,
                       np.ones_like(x)))
    elif mode == '0xz':
        D = np.hstack((x**2 + z**2 - 2 * y**2, 2 * x, 2 * y, 2 * z,
                       np.ones_like(x)))

    # fit sphere in the form A(x^2 + y^2 + z^2) + 2Gx + 2Hy + 2Iz = 1
    elif mode == 'xyz':
        D = np.hstack((2 * x, 2 * y, 2 * z, np.ones_like(x)))
    else:
        raise ValueError('Unknown mode "{}"'.format(mode))

    if D.shape[0] < D.shape[1]:
        raise ValueError('Not enough datapoints')

    # solve the normal system of equations
    d2 = x**2 + y**2 + z**2  # the RHS of the llsq problem (y's)
    u = np.linalg.solve(np.dot(D.T, D), (np.dot(D.T, d2)))[:, 0]

    # find the ellipsoid parameters
    # convert back to the conventional algebraic form
    v = np.empty((10), dtype=u.dtype)
    if mode == '':
        v[0] = u[0] +     u[1] - 1
        v[1] = u[0] - 2 * u[1] - 1
        v[2] = u[1] - 2 * u[0] - 1
        v[3:10] = u[2:9]
    elif mode == 'xy':
        v[0] = u[0] - 1
        v[1] = u[0] - 1
        v[2] = -2 * u[0] - 1
        v[3:10] = u[1:8]
    elif mode == 'xz':
        v[0] = u[0] - 1
        v[1] = -2 * u[0] - 1
        v[2] = u[0] - 1
        v[3:10] = u[1:8]
    elif mode == '0':
        v[0] = u[0] +     u[1] - 1
        v[1] = u[0] - 2 * u[1] - 1
        v[2] = u[1] - 2 * u[0] - 1
        v[3:6] = 0
        v[6:10] = u[2:6]
    elif mode == '0xy':
        v[0] = u[0] - 1
        v[1] = u[0] - 1
        v[2] = -2 * u[0] - 1
        v[3:6] = 0
        v[6:10] = u[2:6]
    elif mode == '0xz':
        v[0] = u[0] - 1
        v[1] = -2 * u[0] - 1
        v[2] = u[0] - 1
        v[3:6] = 0
        v[6:10] = u[2:6]
    elif mode == 'xyz':
        v[:3] = -1
        v[3:6] = 0
        v[6:10] = u[:4]

    # form the algebraic form of the ellipsoid
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])
    # find the center of the ellipsoid
    center = -np.linalg.solve(A[:3, :3], v[6:9])

    # form the corresponding translation matrix
    T = np.identity(4, dtype=A.dtype)
    T[3, :3] = center
    # translate to the center
    R = np.dot(np.dot(T, A), T.T)
    if return_mode == 'skew':
        # extract the xy skew (ignoring a parameter here!)
        skew_xy = -R[2, :2] / np.diag(R[:2, :2])
        radius = np.diag(R[:3, :3]) / R[3, 3]

        # do some trick to make radius_z be the unskewed radius
        radius[2] -= np.sum(radius[:2] * skew_xy**2)
        radius = np.sqrt(1 / np.abs(radius))
        return radius[::-1], center[::-1], skew_xy[::-1]

    # solve the eigenproblem
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    radii = (np.sqrt(1 / np.abs(evals)) * np.sign(evals))

    if return_mode == 'euler':
        # sort the vectors so that -> z, y, x
        new_order = np.argmax(np.abs(evecs), 1)
        radii = radii[new_order]
        evecs = evecs[:, new_order]

        # Discover Euler angle vector from 3x3 matrix
        cy_thresh = np.finfo(evecs.dtype).eps * 4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = evecs.T.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = np.sqrt(r33*r33 + r23*r23)
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            # z: atan2(cos(y)*sin(z), cos(y)*cos(z)),
            # y: atan2(sin(y), cy), atan2(cos(y)*sin(x),
            # x: cos(x)*cos(y))
            angles = np.array([np.arctan(r12/r11), np.arctan(-r13/cy),
                               np.arctan(r23/r33)])
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            # y: atan2(sin(y), cy)
            angles = np.array([np.arctan(-r21/r22), np.arctan(-r13/cy), 0.0])

        return radii[::-1], center[::-1], angles

    return radii[::-1], center[::-1], evecs[::-1]


def ellipse_grid(radius, center, rotation=0, skew=0, n=None, spacing=1):
    """ Returns points and normal (unit) vectors on an ellipse.

    Parameters
    ----------
    radius : tuple
        (yr, xr) the two principle radii of the ellipse
    center : tuple
        (yc, xc) the center coordinate of the ellipse
    rotation : float, optional
        angle of xr with the x-axis, in radians. Rotates clockwise in image.
    skew : float, optional
        skew: y -> y + skew * x
    n : int, optional
        number of points
    spacing : float, optional
        When `n` is not given then the spacing is determined by `spacing`.

    Returns
    -------
    two arrays of shape (2, N), being the coordinates and unit normals
    """
    yr, xr = radius
    yc, xc = center
    if n is None:
        n = int(2*np.pi*np.sqrt((yr**2 + xr**2) / 2) / spacing)

    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    pos = np.array([yr * np.sin(phi), xr * np.cos(phi)])

    normal = np.array([np.sin(phi) / yr, np.cos(phi) / xr])
    normal /= np.sqrt((normal**2).sum(0))

    mask = np.isfinite(pos).all(0) & np.isfinite(normal).all(0)
    pos = pos[:, mask]
    normal = normal[:, mask]

    if rotation != 0:
        R = np.array([[ np.cos(rotation), np.sin(rotation)],
                      [-np.sin(rotation), np.cos(rotation)]])
        pos = np.dot(pos.T, R).T
    elif skew != 0:
        pos[0] += pos[1] * skew

    # translate
    pos[0] += yc
    pos[1] += xc
    return pos, normal  # both in y_list, x_list format


def ellipsoid_grid(radius, center, spacing=1):
    """ Returns points and normal (unit) vectors on an ellipse on only
    integer values of z.

    Parameters
    ----------
    radius : tuple
        (zr, yr, xr) the three principle radii of the ellipsoid
    center : tuple
        (zc, yc, xc) the center coordinate of the ellipsoid
    spacing : float, optional
        Distance between points

    Returns
    -------
    two arrays of shape (3, N), being the coordinates and unit normals
    """
    zc, yc, xc = center
    zr, yr, xr = radius

    pos = np.empty((3, 0))
    for z in range(int(zc - zr + 1), int(zc + zr) + 1):
        n = int(2*np.pi*np.sqrt((yr**2 + xr**2) / 2) / spacing)
        if n == 0:
            continue
        phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
        factor = np.sqrt(1 - ((zc - z) / zr)**2)  # = sin(arccos((zc/z)/zr))
        pos = np.append(pos,
                        np.array([[float(z)] * n,
                                  yr * factor * np.sin(phi) + yc,
                                  xr * factor * np.cos(phi) + xc]),
                        axis=1)
    normal = (pos - np.array(center)[:, np.newaxis]) / np.array(radius)[:, np.newaxis]
    normal /= np.sqrt((normal**2).sum(0))

    mask = np.isfinite(pos).all(0) & np.isfinite(normal).all(0)
    return pos[:, mask], normal[:, mask]


def max_linregress(arr, maxfit_size=2, threshold=0.1, axis=1):
    """ Locates maxima by fitting parabolas to values around the maximum.

    This function is optimized for two-dimensional numpy arrays. For each row
    in the array, the index of the maximum value is located. Then some values
    around it (given by ``maxfit_size``) are taken, the first (discrete)
    derivative is taken, and linear regression is done. This gives the location
    of the maximum with sub-pixel precision. Effectively, a parabola is fitted.

    Parameters
    ----------
    arr : ndarray
    maxfit_size : integer, optional
        Defines the fit region around the maximum value. By default, this value
        is 2, that is, two pixels before and two pixels after the maximum are
        used for the fit (a total of 5).
    threshold :
        Discard points when the average value of the fit region is lower than
        ``threshold`` times the maximum in the whole fit array. This helps
        discarding low-intensity peaks. Default 0.1: if the average intensity
        in the fitregion is below 10% of the global maximum, the point is
        discarded.
    axis : {0, 1}
        axis along which the maxima are fitted. Default 1.

    Returns
    -------
    ndarray with the locations of the maxima.
    Elements are NaN in all of the following cases:
     - any pixel in the fitregion is 0
     - the mean of the fitregion < threshold * global max
     - regression returned infinity
     - maximum is outside of the fit region.
    """
    if axis == 0:
        arr = arr.T
    # identify the regions around the max value
    maxes = np.argmax(arr[:, maxfit_size:-maxfit_size],
                      axis=1) + maxfit_size
    ind = maxes[:, np.newaxis] + range(-maxfit_size, maxfit_size+1)

    # must cast dtype from unsigned to signed integer
    dtype = np.dtype(arr.dtype)
    if dtype.kind == 'u':
        if dtype.itemsize == 1:
            dtype = np.int16
        elif dtype.itemsize == 2:
            dtype = np.int32
        else:
            dtype = np.int64
    else:
        dtype = arr.dtype

    fitregion = np.array([_int.take(_ind) for _int, _ind in zip(arr, ind)],
                         dtype=dtype)

    # fit max using linear regression
    intdiff = np.diff(fitregion, 1)
    x_norm = np.arange(-maxfit_size + 0.5, maxfit_size + 0.5)
    y_mean = np.mean(intdiff, axis=1, keepdims=True)
    y_norm = intdiff - y_mean
    slope = np.sum(x_norm[np.newaxis, :] * y_norm, 1) / np.sum(x_norm * x_norm)
    slope[slope == 0] = np.nan  # protect against division by zero
    r_dev = - y_mean[:, 0] / slope

    # mask invalid fits
    threshold = threshold * fitregion.max()    # relative to global maximum
    with np.errstate(invalid='ignore'):  # ignore comparison with np.nan
        valid = (np.isfinite(r_dev) &              # finite result
                 (fitregion > 0).all(1) &          # all pixels in fitregion > 0
                 (fitregion.mean(1) > threshold) & # fitregion mean > threshold
                 (r_dev > -maxfit_size + 0.5) &    # maximum inside fit region
                 (r_dev < maxfit_size - 0.5))
    r_dev[~valid] = np.nan
    return r_dev + maxes


def max_edge(arr, threshold=0.5, axis=1):
    """ Find strongest decreasing edge on each row """
    if axis == 0:
        arr = arr.T
    derivative = -1*np.diff(arr)
    index = np.argmax(derivative, axis=1)
    values = np.max(derivative, axis=1)
    r_dev = index + 0.5
    r_dev[values < threshold * values.max()] = np.nan
    return r_dev
