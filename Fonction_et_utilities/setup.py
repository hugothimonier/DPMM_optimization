from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='cython_dpmm_algorithm',
      ext_modules=cythonize("dpmm_cython_functions.pyx"),
      include_dirs = [numpy.get_include()])
