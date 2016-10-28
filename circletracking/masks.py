from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from trackpy.utils import validate_tuple


def get_slice(coords, shape, radius):
    """Returns the slice and origin that belong to ``slice_image``"""
    # interpret parameters
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    coords = np.atleast_2d(np.round(coords).astype(np.int))
    # drop features that have no pixels inside the image
    in_bounds = np.array([(coords[:, i] >= -r) & (coords[:, i] < sh + r)
                         for i, sh, r in zip(range(ndim), shape, radius)])
    coords = coords[np.all(in_bounds, axis=0)]
    # return if no coordinates are left
    if len(coords) == 0:
        return [slice(None, 0)] * ndim, None
    # calculate the box
    lower = coords.min(axis=0) - radius
    upper = coords.max(axis=0) + radius + 1
    # calculate the slices
    origin = [None] * ndim
    slices = [None] * ndim
    for i, sh, low, up in zip(range(ndim), shape, lower, upper):
        lower_bound_trunc = max(0, low)
        upper_bound_trunc = min(sh, up)
        slices[i] = slice(lower_bound_trunc, upper_bound_trunc)
        origin[i] = lower_bound_trunc
    return slices, origin


def slice_image(pos, image, radius):
    """ Slice a box around a group of features from an image.

    The box is the smallest box that contains all coordinates up to `radius`
    from any coordinate.

    Parameters
    ----------
    image : ndarray
        The image that will be sliced
    pos : iterable
        An iterable (e.g. list or ndarray) that contains the feature positions
    radius : number or tuple of numbers
        Defines the size of the slice. Every pixel that has a distance lower or
        equal to `radius` to a feature position is included.

    Returns
    -------
    tuple of:
    - the sliced image
    - the coordinate of the slice origin (top-left pixel)
    """
    slices, origin = get_slice(pos, image.shape,  radius)
    return image[slices], origin


def get_mask(pos, shape, radius, include_edge=True, return_masks=False):
    """ Create a binary mask that masks pixels farther than radius to all
    given feature positions.

    Optionally returns the masks that recover the individual feature pixels from
    a masked image, as follows: ``image[mask][masks_single[i]]``

    Parameters
    ----------
    pos : ndarray (N x 2 or N x 3)
        Feature positions
    shape : tuple
        The shape of the image
    radius : number or tuple
        Radius of the individual feature masks
    include_edge : boolean, optional
        Determine whether pixels at exactly one radius from a position are
        included. Default True.
    return_masks : boolean, optional
        Also return masks that recover the single features from a masked image.
        Default False.

    Returns
    -------
    ndarray containing a binary mask
    if return_masks==True, returns a tuple of [masks, masks_singles]
    """
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    pos = np.atleast_2d(pos)

    if include_edge:
        in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
                   for p in pos]
    else:
        in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
                   for p in pos]
    mask_total = np.any(in_mask, axis=0).T
    if return_masks:
        masks_single = np.empty((len(pos), mask_total.sum()), dtype=np.bool)
        for i, _in_mask in enumerate(in_mask):
            masks_single[i] = _in_mask.T[mask_total]
        return mask_total, masks_single
    else:
        return mask_total
