from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from trackpy.utils import validate_tuple
from functools import wraps


def wrap_imshow(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        normed = kwargs.pop('normed', True)
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
            def draw_callback(event):
                adjust_imshow(kwargs['ax'], normed)
            kwargs['ax'].figure.canvas.mpl_connect('draw_event', draw_callback)
        return func(*args, **kwargs)
    return wrapper


def wrap_imshow3d(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        aspect = kwargs.pop('aspect', 1.)
        normed = kwargs.pop('normed', True)
        spacing = kwargs.pop('spacing', 0.05)
        if kwargs.get('axs') is None:
            fig = plt.gcf()
            axs = fig.add_subplot(221), fig.add_subplot(222), \
                  fig.add_subplot(223), fig.add_subplot(224)
            axs[3].set_visible(False)
            kwargs['axs'] = axs
            def draw_callback(event):
                adjust_imshow3d(kwargs['axs'], aspect, spacing, normed)
            fig.canvas.mpl_connect('draw_event', draw_callback)
        return func(*args, **kwargs)
    return wrapper


def invert_ax(ax, which='both', invert=True, auto=None):
    """Inverts the x and/or y axes of an axis object."""
    # kwarg auto=None leaves autoscaling unchanged
    if which not in ('x', 'y', 'both'):
        raise ValueError("Parameter `which` must be one of {'x' | 'y' | 'both'}.")
    if which == 'x' or which == 'both':
        low, hi = ax.get_xlim()
        if invert and hi > low:
            ax.set_xlim(hi, low, auto=auto)
        if not invert and low > hi:
            ax.set_xlim(low, hi, auto=auto)
    if which == 'y' or which == 'both':
        low, hi = ax.get_ylim()
        if invert and hi > low:
            ax.set_ylim(hi, low, auto=auto)
        if not invert and low > hi:
            ax.set_ylim(low, hi, auto=auto)
    return ax


def get_visible_clim(ax):
    """Obtains the sliced image displayed on ax"""
    try:
        axim = ax.get_images()[0]
    except IndexError:
        return
    sh_y, sh_x = axim.get_size()
    ext_x_lo, ext_x_hi, ext_y_lo, ext_y_hi = axim.get_extent()
    if ext_y_lo > ext_y_hi:
        ext_y_lo, ext_y_hi = ext_y_hi, ext_y_lo

    mpp = [(ext_y_hi - ext_y_lo) / sh_y,
           (ext_x_hi - ext_x_lo) / sh_x]

    origin = [ext_y_lo / mpp[0] + 0.5,
              ext_x_lo / mpp[0] + 0.5]

    x_lo, x_hi = ax.get_xlim()
    y_hi, y_lo = ax.get_ylim()

    slice_x = slice(max(int(round(x_lo / mpp[1] + 0.5 - origin[1])), 0),
                    min(int(round(x_hi / mpp[1] + 0.5 - origin[1])), sh_x))
    slice_y = slice(max(int(round(y_lo / mpp[0] + 0.5 - origin[0])), 0),
                    min(int(round(y_hi / mpp[0] + 0.5 - origin[0])), sh_y))
    im = axim.get_array()[slice_y, slice_x]
    return im.min(), im.max()


def norm_axesimage(axim, vmin, vmax):
    im = axim.get_array()
    if im.ndim == 3:  # RGB, custom norm
        if vmax - vmin > 0:
            axim.set_array((im - vmin) / (vmax - vmin))
        axim.set_clim(0, 1)  # this is actually ignored
    else:  # use built-in
        axim.set_clim(vmin, vmax)
    return axim


def adjust_imshow(ax, normed=True):
    # disable autoscaling, use tight layout
    ax.autoscale(False, 'both', tight=False)
    # set aspect ratio
    ax.set_aspect('equal', 'box')
    # invert axes
    invert_ax(ax, 'y', invert=True)
    invert_ax(ax, 'x', invert=False)
    # position the ticks
    ax.xaxis.tick_top()
    # hide grid and tickmarks
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(False)
    # get maximum pixel values
    if normed:
        rng = get_visible_clim(ax)
        if rng is not None:
            norm_axesimage(ax.get_images()[0], *rng)
    return ax


def adjust_imshow3d(axs, aspect=1., spacing=0.05, normed=True):
    ax_xy, ax_zy, ax_zx, ax_extra = axs

    # disable autoscaling, use tight layout
    ax_xy.autoscale(False, 'both', tight=False)
    ax_zy.autoscale(False, 'both', tight=False)
    ax_zx.autoscale(False, 'both', tight=False)
    ax_extra.autoscale(False, 'both', tight=False)

    # set aspect ratio
    ax_xy.set_aspect('equal', 'box')
    ax_zy.set_aspect(1/aspect, 'box')
    ax_zx.set_aspect(aspect, 'box')

    # invert axes
    invert_ax(ax_xy, 'y', invert=True)
    invert_ax(ax_xy, 'x', invert=False)
    invert_ax(ax_zy, 'x', invert=False)

    # get x, y, z limits
    x_lo, x_hi = ax_xy.get_xlim()
    y_hi, y_lo = ax_xy.get_ylim()
    z_lo, z_hi = ax_zy.get_xlim()

    # copy axes limits
    ax_zy.set_ylim(y_hi, y_lo)
    ax_zx.set_xlim(x_lo, x_hi)
    ax_zx.set_ylim(z_hi, z_lo)

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[x_hi - x_lo, aspect * (z_hi - z_lo)],
                           height_ratios=[y_hi - y_lo, aspect * (z_hi - z_lo)],
                           wspace=spacing, hspace=spacing)
    ax_xy.set_position(gs[0, 0].get_position(ax_xy.figure))
    ax_zx.set_position(gs[1, 0].get_position(ax_zx.figure))
    ax_zy.set_position(gs[0, 1].get_position(ax_zy.figure))
    ax_extra.set_position(gs[1, 1].get_position(ax_extra.figure))

    # position and hide the correct ticks
    ax_xy.xaxis.tick_top()
    ax_zy.xaxis.tick_top()
    plt.setp(ax_xy.get_xticklabels() + ax_xy.get_yticklabels() +
             ax_zy.get_xticklabels() + ax_zx.get_yticklabels(),
             visible=True)
    plt.setp(ax_zy.get_yticklabels() + ax_zx.get_xticklabels(),
             visible=False)

    # hide grid and tickmarks
    for ax in [ax_xy, ax_zx, ax_zy]:
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(False)

    # get maximum pixel values
    if normed:
        vmin_xy, vmax_xy = get_visible_clim(ax_xy)
        vmin_zy, vmax_zy = get_visible_clim(ax_zy)
        vmin_zx, vmax_zx = get_visible_clim(ax_zx)
        vmin = min(vmin_xy, vmin_zy, vmin_zx)
        vmax = min(vmax_xy, vmax_zy, vmax_zx)
        for ax in [ax_xy, ax_zy, ax_zx]:
            norm_axesimage(ax.get_images()[0], vmin, vmax)
    return axs


@wrap_imshow
def imshow(image, ax=None, mpp=1., origin=(0, 0), **kwargs):
    """Show an image. Origin is in pixels."""
    _imshow_style = dict(origin='lower', interpolation='nearest',
                         cmap=plt.cm.gray, aspect='equal')
    _imshow_style.update(kwargs)
    shape = image.shape[:2]
    mpp = validate_tuple(mpp, ndim=2)
    origin = validate_tuple(origin, ndim=2)

    # extent is defined on the outer edges of the pixels
    # we want the center of the topleft to intersect with the origin
    extent = [(origin[1] - 0.5) * mpp[1],
              (origin[1] + shape[1] - 0.5) * mpp[1],
              (origin[0] - 0.5) * mpp[0],
              (origin[0] + shape[0] - 0.5) * mpp[0]]

    ax.imshow(image, extent=extent, **_imshow_style)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    return ax

@wrap_imshow3d
def imshow3d(image3d, mode='max', center=None, mpp=1.,
             origin=(0, 0, 0), axs=None, **kwargs):
    """Shows the xy, xz, and yz projections of a 3D image.

    Parameters
    ----------
    image3d : ndarray
    mode : {'max' | 'slice'}
    aspect : number
        aspect ratio of pixel size z / xy. Default 1.
    center : tuple
        in pixels
    mpp : tuple
        microns per pixel
    origin : tuple
        coordinate of the (center of the) topleft pixel (in pixels)
    spacing : number
        spacing between images
    axs : t

    Returns
    -------
    fig, (ax_xy, ax_zy, ax_zx, ax_extra)
    """
    imshow_style = dict(origin='lower', interpolation='nearest',
                        cmap=plt.cm.gray, aspect='auto')
    imshow_style.update(kwargs)
    shape = image3d.shape[:3]
    mpp = validate_tuple(mpp, ndim=3)
    origin = validate_tuple(origin, ndim=3)
    ax_xy, ax_zy, ax_zx, ax_extra = axs

    if mode == 'max':
        image_xy = image3d.max(0)
        image_zx = image3d.max(1)
        image_zy = image3d.max(2)
    elif mode == 'slice':
        center_i = [int(round(c - o)) for c, o in zip(center, origin)]
        image_xy = image3d[center_i[0], :, :]
        image_zx = image3d[:, center_i[1], :]
        image_zy = image3d[:, :, center_i[2]]
    else:
        raise ValueError

    if image_zy.ndim == 3:
        image_zy = np.transpose(image_zy, (1, 0, 2))
    else:
        image_zy = image_zy.T

    # extent is defined on the outer edges of the pixels
    # we want the center of the topleft to intersect with the origin
    extent = [(origin[2] - 0.5) * mpp[2],
              (origin[2] + shape[2] - 0.5) * mpp[2],
              (origin[1] - 0.5) * mpp[1],
              (origin[1] + shape[1] - 0.5) * mpp[1],
              (origin[0] - 0.5) * mpp[0],
              (origin[0] + shape[0] - 0.5) * mpp[0]]

    extent_xy = extent[:4]
    extent_zx = extent[:2] + extent[4:6]
    extent_zy = extent[4:6] + extent[2:4]

    ax_xy.imshow(image_xy, extent=extent_xy, **imshow_style)
    ax_zx.imshow(image_zx, extent=extent_zx, **imshow_style)
    ax_zy.imshow(image_zy, extent=extent_zy, **imshow_style)

    ax_xy.set_xlim(extent[0], extent[1], auto=False)
    ax_xy.set_ylim(extent[3], extent[2], auto=False)
    ax_zy.set_xlim(extent[4], extent[5], auto=False)
    ax_zy.set_ylim(extent[3], extent[2], auto=False)
    ax_zx.set_xlim(extent[0], extent[1], auto=False)
    ax_zx.set_ylim(extent[5], extent[4], auto=False)
    return axs


@wrap_imshow
def annotate_ellipse(params, ax=None, crop_radius=1.2, **kwargs):
    """Annotates an ellipse on an image

    Parameters
    ----------
    params : tuple or dict
        either (yr, xr, yc, xc) tuple
        or dict with names ['yr', 'xr', 'yc', 'xc']
    """
    from matplotlib.patches import Ellipse
    ellipse_style = dict(ec='yellow', fill=False)
    ellipse_style.update(kwargs)

    if isinstance(params, tuple):
        yr, xr, yc, xc = params
    else:
        yr = params['yr']
        xr = params['xr']
        yc = params['yc']
        xc = params['xc']

    ax.add_artist(Ellipse(xy=(xc, yc), width=xr*2, height=yr*2,
                          **ellipse_style))

    # crop image around ellipse
    ax.set_xlim(xc - crop_radius * xr, xc + crop_radius * xr)
    ax.set_ylim(yc + crop_radius * yr, yc - crop_radius * yr)
    return ax


@wrap_imshow3d
def annotate_ellipsoid(params, axs=None, crop_radius=1.2, **kwargs):
    """Annotates an ellipse on an image

    Parameters
    ----------
    params : tuple or dict
        either (zr, yr, xr, zc, yc, xc) tuple
        or dict with names ['zr', 'yr', 'xr', 'zc', 'yc', 'xc']
    """
    from matplotlib.patches import Ellipse
    ellipse_style = dict(ec='yellow', fill=False)
    ellipse_style.update(kwargs)
    ax_xy, ax_zy, ax_zx, ax_extra = axs

    if isinstance(params, tuple):
        zr, yr, xr, zc, yc, xc = params
    else:
        zr = params['zr']
        yr = params['yr']
        xr = params['xr']
        zc = params['zc']
        yc = params['yc']
        xc = params['xc']

    ax_xy.add_artist(Ellipse(xy=(xc, yc), width=xr*2, height=yr*2,
                             **ellipse_style))
    ax_zy.add_artist(Ellipse(xy=(zc, yc), width=zr*2, height=yr*2,
                             **ellipse_style))
    ax_zx.add_artist(Ellipse(xy=(xc, zc), width=xr*2, height=zr*2,
                             **ellipse_style))

    # crop image around ellipse
    ax_xy.set_xlim(xc - crop_radius * xr, xc + crop_radius * xr)
    ax_xy.set_ylim(yc - crop_radius * yr, yc + crop_radius * yr)
    ax_zy.set_xlim(zc - crop_radius * zr, zc + crop_radius * zr)
    return axs
