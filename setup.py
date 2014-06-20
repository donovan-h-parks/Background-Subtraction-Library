from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "pybgs",
	include_dirs = ['/usr/include/opencv/', '/usr/include/opencv2/core', '/usr/include/opencv2/highgui'],
    ext_modules = cythonize('pybgs.pyx')
)

