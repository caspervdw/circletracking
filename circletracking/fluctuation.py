from __future__ import (division, unicode_literals)

import numpy as np
from .algebraic import fit_ellipse
import matplotlib.pyplot as plt


def circle_deviation(coords):
    """ Fits a circle to given coordinates using an algebraic fit. Additionally
    returns the deviations from the circle radius in a list sorted on angle.

    Parameters
    ----------
    coords :
        (n, 2) array of (y, x) coordinates

    Returns
    -------
    tuple of center, radius, array of theta, array of deviatory radius

    """
    # algebraicly fit a circle to all coords. don't check for outliers because
    # refine_ellipse should only look in a small deviatory radius interval,
    # it already rejected radii that were to close to the interval boundary.
    (r, _), (yc, xc), _ = fit_ellipse(coords.T, mode='circle')

    # calculate deviatory radius and angle
    y = coords[:, 0] - yc
    x = coords[:, 1] - xc
    r_dev = np.sqrt(y**2 + x**2) - r
    theta = np.arctan2(y, x)

    # sort the radial coordinates on the value of theta (which is -pi to +pi)
    sortindices = np.argsort(theta)
    theta = theta[sortindices]
    r_dev = r_dev[sortindices]
    return (yc, xc), r, theta, r_dev


def power_spectrum(theta, r_dev, r, modes=None, part=None):
    """ From deviatory radius as function of theta, calculates the power
    spectrum of fluctuations upto a certain mode. Fast Fourier Transform is not
    used because theta values could be unevenly spaced. Instead, numerical
    integration is performed using the trapezoid rule.

    Theta must be sorted, but the phase shift is irrelevant as the absolute
    value is taken (phase information is lost). -pi to pi works, 0-2pi too.

    Parameters
    ----------
    theta :
        array of angles, in radius, sorted, rangeing from -pi to pi
    r_dev :
        array of deviatory radius, in um, belonging to theta values
    r :
        avarage radius, in um
    modes :
        array of integers. the modes for which the DFT is done

    Returns
    -------
    array of numbers, power spectrum (squared of absolute value) of DFT. The
    wavenumber values belonging to each mode are given by mode / <R>, in which
    R is the average radius of the fluctuating circle.
    """
    if modes is None:
        modes = np.arange(1, 101)
    if part is None:
        part = 1

    if part <= 2:
        Ntheta = len(theta)
        fft = np.sum(r_dev[np.newaxis, :] *
                     np.exp(-1j * modes[:, np.newaxis] * theta[np.newaxis, :]),
                     axis=1) / Ntheta
        powersp = np.abs(fft)**2
    else:
        half_angle = np.pi / part
        mask = ((theta >= (np.pi/2 - half_angle)) *
                (theta < (np.pi/2 + half_angle)))
        fft_btm = np.sum(r_dev[np.newaxis, mask] *
                         np.exp(-1j * modes[:, np.newaxis] * part *
                                theta[np.newaxis, mask]), axis=1) / mask.sum()
        mask = ((theta >= (-np.pi/2 - half_angle)) *
                (theta < (-np.pi/2 + half_angle)))
        fft_top = np.sum(r_dev[np.newaxis, mask] *
                         np.exp(-1j * modes[:, np.newaxis] * part *
                                theta[np.newaxis, mask]), axis=1) / mask.sum()
        powersp = (np.abs(fft_top)**2 + np.abs(fft_btm)**2) / 2

    return 2 * np.pi * r * powersp  # rescale with circumference


def epower_spectrum(coords, max_mode, max_r_dev=0.1, mpp=1., part=1,
                    minpx_fullwave=None, show=False):
    """ From an iterable of coordinates, calculates average DFT powerspectrum
    of fluctuations around a circle.

    Parameters
    ----------
    coords :
        iterable of (n, 2) arrays of (y, x) coordinates in pixels
    max_mode :
        fluctuation upto this mode are calculated
    max_r_dev :
        circlefits with with a circle radius that differ more than max_r_dev
        from the ensemble median, are dropped.
    mpp :
        microns per pixel
    minpx_fullwave :
        truncates the produced fft so that each full wave has given minimum of
        pixels, as well as in the original picture and as in the sampling

    Returns
    -------
    qx : wavenumbers in 1 / um
    fft2 : powerspectrum in um^(3/2)
    """
    frame_count = len(coords)
    if minpx_fullwave is not None:
        _, r, theta, _ = circle_deviation(coords[0])
        spacing = np.median(np.diff(theta))
        # pixels in original picture
        maxmode1 = round(2*np.pi*r / minpx_fullwave)
        # sampled pixels
        maxmode2 = round(2*np.pi / spacing / minpx_fullwave)
        max_mode = min(max_mode, maxmode1, maxmode2)

    modes = np.arange(1, max_mode+1)
    fft2 = np.empty((frame_count, max_mode), dtype=np.float)
    radii = np.empty(frame_count, dtype=np.float)

    # spacing = np.empty(frame_count, dtype=np.float)
    for i, coord in enumerate(coords):
        _, r, theta, r_dev = circle_deviation(coord)
        fft2[i] = power_spectrum(theta, r_dev*mpp, r*mpp, modes, part)
        radii[i] = r*mpp
        # spacing[i] = np.median(np.diff(theta))
    avrad = np.median(radii)
    mask = ((radii > (avrad * (1 - max_r_dev))) &
            (radii < (avrad * (1 + max_r_dev))))
    avrad = np.average(radii[mask])

    qx = modes / avrad
    if part > 2:
        qx *= part

    fft2 = np.average(fft2[mask], axis=0)

    if show:
        plt.plot(qx, fft2, marker='.')
        plt.xlabel(r'$q_x [\mu m^{-1}]$')
        plt.ylabel(r'$L \langle|u(q_x)|^2\rangle [\mu m^{3}]$')
        plt.ylim(0,np.max(fft2[6:]))
        plt.grid()
        plt.show()

    return qx, fft2
