import os
import versioneer
from setuptools import setup


try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except IOError:
    descr = ''

try:
    from pypandoc import convert
    descr = convert(descr, 'rst', format='md')
except ImportError:
    pass


# In some cases, the numpy include path is not present by default.
# Let's try to obtain it.
try:
    import numpy
except ImportError:
    ext_include_dirs = []
else:
    ext_include_dirs = [numpy.get_include(),]

setup_parameters = dict(
    name = "circletracking",
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    description = "tracking toolkit for circles and ellipses in 2D and 3D",
    author = "Casper van der Wel",
    author_email = "caspervdw@gmail.com",
    url = "https://github.com/caspervdw/circletracking",
    install_requires = ['numpy>=1.8', 'scipy>=0.13', 'six>=1.8',
	                    'pandas>=0.13', 'scikit-image>=0.9',
                        'matplotlib>=1.3'],
    packages = ['circletracking'],
    long_description = descr,
)

setup(**setup_parameters)
