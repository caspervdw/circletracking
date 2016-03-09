circletracking
==============

[![build status](https://travis-ci.org/caspervdw/circletracking.png?branch=master)](https://travis-ci.org/caspervdw/circletracking) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.47216.svg)](http://dx.doi.org/10.5281/zenodo.47216)

Circletracking is a module for quantitative tracking of circles (and ellipses) in 2D or 3D images.

Overview
--------
Detect bright circles on a dark background with subpixel accuracy. The circle needs to have a Gaussian-like radial profile.
- `find_ellipse` and `find_ellipsoid` crudely find the ellipse/ellipsoid
- `refine_ellipse` and `refine_ellipsoid` interpolate the image lines originating from the estimated center, locates the ellipse/ellipsoid boundary with subpixel accuracy and fits an ellipse/ellipsoid to the obtained points
- `locate_ellipse` and `locate_ellipsoid` combine `find` and `refine` in one routine
- routines in `artificial` generate artificial ellipses/ellipsoids for testing purposes

Testing
-------
Unittests ensure high accuracy:
- on ellipses with 20% random noise: 0.1px for center and 1% for radius
- on ellipsoids with 20% random noise: 0.5px for center and 5% for radius

More information
----------------
Please consult the docstrings for more details.
